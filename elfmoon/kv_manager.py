"""KV Cache 永続化マネージャ（ハイブリッドアーキテクチャ対応）。

Qwen3.6 系は full attention（KVCache）と linear attention（ArraysCache＝再帰状態）が
混在する。再帰状態は KV と違って途中位置に切り詰められないため、保存は
「全レイヤーが同一トークン数を処理した整合状態」でのみ行う。

運用フロー（api_server 側）:
  1. prefill 完了直後（プロンプト先頭 len-1 トークン処理時点）に snapshot() で
     状態への参照を捕捉する（この時点では重いコピーをしない）
  2. 生成完了後に save() で捕捉時点の状態をメモリ＋ディスクへ永続化する
  3. 次リクエストの lookup() はプロンプトとの最長プレフィックス一致を返す

ディスク形式は version=2（KV に加えて再帰状態も保存）。旧形式（v1）は
再帰状態を欠く不整合データのため、起動時に削除する。
"""

import hashlib
import json
import os
import struct
import sys
import threading
import time
from collections import OrderedDict
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, trim_prompt_cache

DISK_CACHE_DIR = os.environ.get("ELFMOON_KV_CACHE_DIR") or os.path.expanduser(
    "~/.cache/elfmoon/kv_cache"
)
MAX_DISK_ENTRIES = 4
MIN_SAVE_TOKENS = 20
FORMAT_VERSION = 3
# 退避点（rewind point）関連。
# 会話履歴の**中盤**が書き換わる場合（tool_result 再描画・履歴圧縮・system 追記等）、
# 完全プレフィックス一致では再利用不可になる。prefill のチャンク境界ごとに
# 再帰状態（ArraysCache、trim 不可）を退避しておき、共通 prefix まで巻き戻して再利用する。
# 退避点は「trim できない層の状態」だけで持つ（KV は trim_prompt_cache で戻せる）。
SNAPSHOT_MAX = int(os.environ.get("ELFMOON_SNAPSHOTS", "24"))
SNAPSHOT_TAIL = int(os.environ.get("ELFMOON_SNAPSHOT_TAIL", "256"))
# 情報ログ（hit/save 等）。対話 CLI ではプロンプト表示に割り込むため抑制できる。
# エラーログは本フラグに関わらず常に出す。
KVC_LOG = os.environ.get("ELFMOON_KVC_LOG", "1") != "0"


def _build_cache_objects(
    offset: int, layer_data: list[tuple[str, Any]], n_layers: int
) -> list[Any]:
    """保存データからキャッシュオブジェクトを再構築する。"""
    cache: list[Any] = []
    for tag, data in layer_data:
        if tag == "kv":
            keys, vals = data
            kc = KVCache()
            kc.keys = keys
            kc.values = vals
            kc.offset = offset
            cache.append(kc)
        elif tag == "arr":
            ac = ArraysCache(size=2)
            if data is not None:
                ac.state = [mx.array(x) for x in data]
            cache.append(ac)
    while len(cache) < n_layers:
        cache.append(ArraysCache(size=2))
    return cache


class KVCacheManager:
    """整合スナップショット方式の KV Cache ストア（メモリ＋ディスク）。"""

    def __init__(self, max_entries: int = 4, cache_dir: str = DISK_CACHE_DIR):
        self._caches: OrderedDict = OrderedDict()
        self._max_entries = max_entries
        self._dir = cache_dir
        self._disk_lock = threading.Lock()
        self._namespace = b""
        os.makedirs(self._dir, exist_ok=True)
        self._purge_old_format()
        # 現セッションの巻き戻し用状態（中盤書き換え時の部分再利用に使う）。
        # _session_tokens: 現セッションでキャッシュに投入済みのトークン列
        # _snaps: [(pos, arrays_state)] 退避点。pos はトークン列内の絶対位置。
        # _live_cache: 現セッションの実キャッシュ（巻き戻しはこれに対して行う）。
        self._session_tokens: list[int] = []
        self._snaps: list[tuple[int, Any]] = []
        self._live_cache: list[Any] | None = None

    def set_namespace(self, name: str):
        """キャッシュキーの名前空間（モデル識別子）を設定する。

        キーはトークン列のみから作られるため、異なるモデル間で同一プロンプトが
        衝突し KV 形状不一致でクラッシュする。モデルパス等を渡して分離すること。
        """
        self._namespace = name.encode()

    # ---- 退避点（rewind points） ----

    @staticmethod
    def _arrays_state(cache: list[Any]) -> list[tuple[int, Any]]:
        """trim できない層（ArraysCache）の状態だけを取り出す。

        KV（KVCache）は trim_prompt_cache で巻き戻せるため退避対象にしない。
        """
        return [
            (i, [mx.array(x) for x in c.state])
            for i, c in enumerate(cache)
            if isinstance(c, ArraysCache)
        ]

    def begin_session(self, tokens: list[int], cache: list[Any] | None = None):
        """新セッションを開始する。退避点を蓄積するためのトークン列を設定する。"""
        self._session_tokens = list(tokens)
        self._snaps = []
        self._live_cache = cache
        if cache is not None:
            self.add_snapshot(cache, len(tokens))

    def add_snapshot(self, cache: list[Any], pos: int):
        """prefill チャンク境界で退避点を記録する（ArraysCache 状態のみ）。

        最大 SNAPSHOT_MAX 個まで保持し、位置が均等になるよう間引く。
        末尾近く（SNAPSHOT_TAIL より手前）には必ず 1 個残す。
        """
        arrays = self._arrays_state(cache)
        if not arrays:
            return
        self._snaps.append((pos, arrays))
        while len(self._snaps) > max(1, SNAPSHOT_MAX):
            if len(self._snaps) <= 2:
                del self._snaps[0]
                continue
            # 隣接間隔がもっとも狭い点を落として位置の分布を均す（末尾は残す）
            drop, gap = 1, None
            for i in range(1, len(self._snaps) - 1):
                g = self._snaps[i + 1][0] - self._snaps[i - 1][0]
                if gap is None or g < gap:
                    gap, drop = g, i
            del self._snaps[drop]

    def _restore_cache(self, cache: list[Any], snap, cur_pos: int) -> bool:
        """退避点位置まで cache を巻き戻す。成功したら True。

        ArraysCache（再帰状態）は退避参照に戻し、KV は trim 可能層だけ
        trim_prompt_cache で切り詰める（混在キャッシュでは全層 trim は失敗する）。
        """
        back = cur_pos - snap[0]
        if back < 0:
            return False
        if back > 0:
            trimmable = [c for c in cache if c.is_trimmable()]
            if trim_prompt_cache(trimmable, back) != back:
                return False
        for i, state in snap[1]:
            cache[i].state = state
        return True

    def common_prefix_len(self, cached: list[int], prompt: list[int]) -> int:
        """2 つのトークン列の一致する先頭長を返す。"""
        n = min(len(cached), len(prompt))
        i = 0
        while i < n and cached[i] == prompt[i]:
            i += 1
        return i

    def rewind_to(self, cache: list[Any], prompt_ids: list[int]):
        """現セッションの退避点を利用して共通 prefix まで巻き戻す。

        共通 prefix 以下で最も新しい退避点まで戻り、その位置を返す。
        巻き戻せなければ 0（全捨て）を返す。
        """
        if not self._session_tokens:
            return 0
        common = self.common_prefix_len(self._session_tokens, prompt_ids)
        if common <= 0:
            return 0
        usable = [s for s in self._snaps if 0 < s[0] <= common]
        if not usable:
            return 0
        snap = max(usable, key=lambda s: s[0])
        if not self._restore_cache(cache, snap, len(self._session_tokens)):
            return 0
        # 巻き戻した位置より後ろの退避点は無効になる
        self._snaps = [s for s in self._snaps if s[0] <= snap[0]]
        self._session_tokens = list(prompt_ids[: snap[0]])
        self._live_cache = cache
        return snap[0]

    def set_live_cache(self, cache: list[Any]):
        """現セッションの実キャッシュを登録する（巻き戻し対象）。"""
        self._live_cache = cache

    def live_rewind(self, prompt_ids: list[int]) -> tuple[list[Any] | None, int]:
        """保持中の実キャッシュを共通 prefix まで巻き戻して返す。

        戻り値は (cache, rewind_pos)。巻き戻せなければ (None, 0)。
        """
        cache = self._live_cache
        if cache is None or not self._session_tokens:
            return None, 0
        pos = self.rewind_to(cache, prompt_ids)
        if pos <= 0:
            return None, 0
        return cache, pos

    def mark_fed(self, tokens: list[int]):
        """キャッシュに投入済みのトークン列を現セッションに反映する（生成後）。"""
        self._session_tokens = list(tokens)
        self._snaps = [s for s in self._snaps if s[0] <= len(tokens)]

    # ---- hash ----

    def _hash_prefix(self, tokens: list[int], length: int) -> str:
        packed = (
            self._namespace
            + b"\x00"
            + b"".join(struct.pack("<i", t) for t in tokens[:length])
        )
        return hashlib.sha256(packed).hexdigest()

    # ---- snapshot ----

    def snapshot(self, cache: list[Any]) -> list[tuple[str, Any]] | None:
        """prefill 直後のキャッシュ状態を捕捉する（save() で永続化する）。

        MLX 配列への添字代入は新しいバッキングを生成するため、ArraysCache の
        state は mx.array() でコピーすれば以後の生成に影響されない。
        KVCache のバッファは offset 以降にのみ追記されるので、参照を保持して
        save() 時に offset までスライスすれば捕捉時点の内容が得られる。
        非対応のキャッシュ型や未初期化の再帰状態がある場合は None（保存不可）。
        """
        snap: list[tuple[str, Any]] = []
        for c in cache:
            if isinstance(c, KVCache):
                snap.append(("kv", (c.keys, c.values, c.offset)))
            elif isinstance(c, ArraysCache):
                state = c.state
                if state is None or any(x is None for x in state):
                    return None
                # 上のガードで全要素 non-None を保証済み（if は型絞り込み用）
                snap.append(("arr", [mx.array(x) for x in state if x is not None]))
            else:
                return None
        return snap

    # ---- lookup（メモリ→ディスク、最長プレフィックス一致） ----

    def lookup(self, prompt_ids: list[int], model) -> tuple[list[Any] | None, int]:
        n_layers = len(getattr(model, "layers", None) or model.model.layers)

        # 0) 現セッションの退避点による部分再利用（履歴の中盤書き換え対応）。
        #    完全一致（下記 1/2）が無い場合でも、共通 prefix 以下で最も新しい
        #    退避点まで巻き戻して再利用できる。
        cache, pos = self.live_rewind(prompt_ids)
        if cache is not None:
            return cache, pos

        # 1) メモリ: 一致する中で最長 offset のエントリ
        best_key = None
        best: tuple[int, Any] | None = None
        for key, (offset, layer_data) in self._caches.items():
            if (
                offset <= len(prompt_ids)
                and (best is None or offset > best[0])
                and self._hash_prefix(prompt_ids, offset) == key
            ):
                best_key, best = key, (offset, layer_data)
        if best is not None:
            self._caches.move_to_end(best_key)
            return _build_cache_objects(best[0], best[1], n_layers), best[0]

        # 2) ディスク: offset 降順に一致を試す
        candidates = [
            e
            for e in self._list_disk_entries()
            if e.get("offset", 0) <= len(prompt_ids)
            and self._hash_prefix(prompt_ids, e["offset"]) == e.get("hash", "")
        ]
        candidates.sort(key=lambda e: e["offset"], reverse=True)
        for entry in candidates:
            key, offset = entry["hash"], entry["offset"]
            try:
                layer_data = self._disk_load_arrays(key)
            except Exception as e:
                print(
                    f"[KVC] disk load error, removing corrupt entry: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                self._disk_delete(key)
                continue
            self._caches[key] = (offset, layer_data)
            self._caches.move_to_end(key)
            while len(self._caches) > self._max_entries:
                self._caches.popitem(last=False)
            if KVC_LOG:
                print(
                    f"[KVC] disk→memory: key={key[:12]} offset={offset}",
                    file=sys.stderr,
                    flush=True,
                )
            # 復元したキャッシュを現セッションに登録し、退避点も読み戻す。
            # 復元直後に会話の前方で分岐しても部分再利用できる。
            restored = _build_cache_objects(offset, layer_data, n_layers)
            self.set_live_cache(restored)
            self._session_tokens = list(prompt_ids[:offset])
            self._snaps = self._disk_load_snaps(key)
            return restored, offset

        return None, 0

    # ---- save（メモリ＋ディスク） ----

    def save(
        self,
        token_ids: list[int],
        snap: list[tuple[str, Any]] | None,
        rewind_snaps: list[tuple[int, Any]] | None = None,
    ):
        """snapshot() の捕捉状態を token_ids（処理済み全トークン）キーで保存する。

        rewind_snaps: 退避点リスト（(pos, arrays_state)）。ディスク保存時は
        `.snaps.safetensors` に併せて永続化する。省略時は現セッションの退避点を使う。
        """
        if snap is None:
            return
        offset = len(token_ids)
        if offset < MIN_SAVE_TOKENS:
            return
        if rewind_snaps is None:
            rewind_snaps = self._snaps
        key = self._hash_prefix(token_ids, offset)
        if key in self._caches and self._caches[key][0] == offset:
            # 同一内容が既にある → メモリ/ディスクとも書き直し不要
            self._caches.move_to_end(key)
            return

        layer_data: list[tuple[str, Any]] = []
        to_eval: list[mx.array] = []
        for tag, data in snap:
            if tag == "kv":
                keys, vals, kv_off = data
                end = min(kv_off, offset)
                if keys is not None and end > 0:
                    k = keys[..., :end, :].astype(mx.float16)
                    v = vals[..., :end, :].astype(mx.float16)
                else:
                    k = mx.zeros((1, 1, 0, 1), dtype=mx.float16)
                    v = mx.zeros((1, 1, 0, 1), dtype=mx.float16)
                layer_data.append(("kv", (k, v)))
                to_eval.extend([k, v])
            else:
                layer_data.append(("arr", data))
                to_eval.extend(data)
        # 退避点の配列もメインスレッドで実体化しておく（バックグラウンド保存
        # スレッドには GPU ストリームが無く、lazy 配列は eval できないため）。
        for _pos, _layers in rewind_snaps:
            for _i, _state in _layers:
                to_eval.extend(_state)
        mx.eval(to_eval)

        # メモリ
        self._caches[key] = (offset, layer_data)
        self._caches.move_to_end(key)
        while len(self._caches) > self._max_entries:
            self._caches.popitem(last=False)

        # ディスク（バックグラウンド書込み: 応答終端をブロックしない）
        threading.Thread(
            target=self._disk_save,
            args=(key, offset, layer_data, len(token_ids), rewind_snaps),
            daemon=True,
        ).start()

    # ---- disk persistence ----

    def _disk_path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.safetensors")

    def _meta_path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.json")

    def _disk_save(
        self,
        key: str,
        offset: int,
        layer_data: list[tuple[str, Any]],
        prompt_length: int,
        rewind_snaps: list[tuple[int, Any]] | None = None,
    ):
        try:
            with self._disk_lock:
                arrays: dict[str, mx.array] = {}
                kv_indices: list[int] = []
                arr_indices: dict[str, int] = {}
                for i, (tag, data) in enumerate(layer_data):
                    if tag == "kv":
                        k, v = data
                        arrays[f"l{i}_keys"] = k
                        arrays[f"l{i}_values"] = v
                        kv_indices.append(i)
                    elif data is not None:
                        for j, x in enumerate(data):
                            arrays[f"l{i}_arr{j}"] = x
                        arr_indices[str(i)] = len(data)

                if arrays:
                    mx.save_safetensors(self._disk_path(key), arrays)

                # 退避点（ArraysCache 状態のみ）を別ファイルに保存する。
                # 復元直後に会話の前方で分岐しても部分再利用できるようにするため。
                snap_meta = None
                if rewind_snaps:
                    snap_meta = self._disk_save_snaps(key, rewind_snaps)

                meta = {
                    "version": FORMAT_VERSION,
                    "hash": key,
                    "offset": offset,
                    "num_layers": len(layer_data),
                    "kv_indices": kv_indices,
                    "arr_indices": arr_indices,
                    "prompt_tokens": prompt_length,
                    "created_at": time.time(),
                    "snaps": snap_meta,
                }
                with open(self._meta_path(key), "w") as f:
                    json.dump(meta, f)

                self._cleanup_disk()

            if KVC_LOG:
                print(
                    f"[KVC] disk save: key={key[:12]} offset={offset} "
                    f"kv={len(kv_indices)} arr={len(arr_indices)}/{len(layer_data)}"
                    + (f" snaps={len(rewind_snaps)}" if rewind_snaps else ""),
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as e:
            print(f"[KVC] disk save error: {e}", file=sys.stderr, flush=True)

    def _snaps_path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.snaps.safetensors")

    def _disk_save_snaps(self, key: str, snaps: list[tuple[int, Any]]) -> dict | None:
        """退避点を `<key>.snaps.safetensors` に保存し、メタ情報を返す。

        各退避点は (pos, [(layer_idx, [state, ...]), ...])。全層の ArraysCache
        状態を保持すると大きいため、現行セッションの退避点に限定して保存する。
        """
        arrays: dict[str, mx.array] = {}
        poss: list[int] = []
        for k, (pos, layers) in enumerate(snaps):
            poss.append(pos)
            for i, state in layers:
                for j, a in enumerate(state):
                    arrays[f"{k}:{i}:{j}"] = a
        if not arrays:
            return None
        # 配列はメインスレッドで既に実体化済み（save() 内で eval 済み）なので、
        # バックグラウンドスレッドでは保存のみ行う。
        # safetensors は拡張子を強制するため一時ファイルも同じ拡張子にする
        tmp = self._snaps_path(key)[: -len(".safetensors")] + ".tmp.safetensors"
        mx.save_safetensors(tmp, arrays, {"positions": ",".join(map(str, poss))})
        os.replace(tmp, self._snaps_path(key))
        return {"file": os.path.basename(self._snaps_path(key)), "positions": poss}

    def _disk_load_snaps(self, key: str) -> list[tuple[int, Any]]:
        """保存済み退避点を読み戻す（壊れていれば空リスト）。"""
        path = self._snaps_path(key)
        if not os.path.isfile(path):
            return []
        try:
            arrays, md = mx.load(path, return_metadata=True)
            poss = [int(x) for x in md.get("positions", "").split(",") if x]
            grouped: dict[int, dict[int, dict[int, mx.array]]] = {}
            for name, a in arrays.items():
                k, i, j = (int(x) for x in name.split(":"))
                grouped.setdefault(k, {}).setdefault(i, {})[j] = a
            out: list[tuple[int, Any]] = []
            for k, pos in enumerate(poss):
                layers = grouped.get(k)
                if not layers:
                    continue
                out.append(
                    (
                        pos,
                        [
                            (i, [d[j] for j in sorted(d)])
                            for i, d in sorted(layers.items())
                        ],
                    )
                )
            return sorted(out, key=lambda s: s[0])
        except Exception as exc:
            print(f"[KVC] rewind load error: {exc}", file=sys.stderr, flush=True)
            return []

    def _disk_load_arrays(self, key: str) -> list[tuple[str, Any]]:
        with open(self._meta_path(key)) as f:
            meta = json.load(f)
        if meta.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported cache format: {meta.get('version')}")
        num_layers = meta["num_layers"]
        kv_indices = set(meta.get("kv_indices", []))
        arr_indices = {int(k): v for k, v in meta.get("arr_indices", {}).items()}

        arrays: dict[str, mx.array] = mx.load(self._disk_path(key))  # type: ignore[assignment]
        layer_data: list[tuple[str, Any]] = []
        for i in range(num_layers):
            if i in kv_indices:
                layer_data.append(
                    ("kv", (arrays[f"l{i}_keys"], arrays[f"l{i}_values"]))
                )
            elif i in arr_indices:
                layer_data.append(
                    ("arr", [arrays[f"l{i}_arr{j}"] for j in range(arr_indices[i])])
                )
            else:
                layer_data.append(("arr", None))
        if all(tag == "arr" and d is None for tag, d in layer_data):
            raise ValueError(f"No cache entries found in {self._disk_path(key)}")
        return layer_data

    def _list_disk_entries(self) -> list[dict]:
        if not os.path.isdir(self._dir):
            return []
        entries = []
        for fname in os.listdir(self._dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self._dir, fname)) as f:
                        meta = json.load(f)
                    if meta.get("version") == FORMAT_VERSION:
                        entries.append(meta)
                except (OSError, json.JSONDecodeError):
                    pass
        return entries

    def _purge_old_format(self):
        """旧形式（再帰状態を欠く v1）のエントリを削除する。"""
        if not os.path.isdir(self._dir):
            return
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    meta = json.load(f)
                version = meta.get("version")
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                version = None
            if version != FORMAT_VERSION:
                key = fname[: -len(".json")]
                self._disk_delete(key)
                if KVC_LOG:
                    print(
                        f"[KVC] purged old-format entry: {key[:12]}",
                        file=sys.stderr,
                        flush=True,
                    )

    def _cleanup_disk(self):
        entries = self._list_disk_entries()
        if len(entries) <= MAX_DISK_ENTRIES:
            return
        entries.sort(key=lambda e: e.get("created_at", 0))
        for entry in entries[:-MAX_DISK_ENTRIES]:
            self._disk_delete(entry["hash"])

    def _disk_delete(self, key: str):
        for path in [
            self._disk_path(key),
            self._meta_path(key),
            self._snaps_path(key),
        ]:
            if os.path.exists(path):
                os.remove(path)

    # ---- clear ----

    def clear(self):
        self._caches.clear()
        self._session_tokens = []
        self._snaps = []
        self._live_cache = None

    def clear_disk(self):
        """Remove all disk cache entries."""
        for entry in self._list_disk_entries():
            self._disk_delete(entry["hash"])


kv_manager = KVCacheManager()
