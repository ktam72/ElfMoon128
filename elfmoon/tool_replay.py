"""永続 Tool replay ストア（ElfMoonDeepSeek の ToolReplayStore の移植）。

ds4 サーバーモードの `tool_memory` に相当する。生成済み tool_call の**生テキスト**
（GLM `<tool_call>` / DSML）を call ID ごとに保存し、後続リクエストで同じ ID の
assistant tool_call を render するときに元のサンプルバイト列を復元する。

ElfMoon4 は `apply_chat_template(tools=...)` でツール描画をモデル側に委譲するため、
本ストアは「assistant tool_call メッセージを生ブロックに置換して apply_chat_template に
渡す」形で統合する（_apply_tool_replay）。これにより KV キャッシュの prefix 一致を
壊さない。

## 永続化
- 既定パス: `~/.cache/elfmoon/tool_replay/<model>.json`（0600）
- `ELFMOON_TOOL_REPLAY_FILE` で変更可

## 制限（ds4 との差分）
- ブロック単位の重複排除はせず、call ID → 生テキストの直接マップ（LRU 上限付き）。
- 既定上限は ds4 と同じ 100,000 件 / 512MB。
"""

import json
import os

TOOL_REPLAY_DEFAULT_MAX_ENTRIES = 100000
TOOL_REPLAY_DEFAULT_MAX_BYTES = 512 * 1024 * 1024
_PER_ENTRY_OVERHEAD = 64


def tool_replay_path(model_name: str) -> str:
    """モデルごとの replay ファイルパスを返す。"""
    env = os.environ.get("ELFMOON_TOOL_REPLAY_FILE")
    if env:
        return env
    base = os.environ.get(
        "ELFMOON_TOOL_REPLAY_DIR",
        os.path.expanduser("~/.cache/elfmoon/tool_replay"),
    )
    safe = (model_name or "model").replace("/", "_").replace("\\", "_")
    return os.path.join(base, f"{safe}.json")


def _glm_arg_value(value):
    """GLM arg_value の描画。文字列は引用符なしの生テキスト（ds4 append_glm_arg_value_text）。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


class ToolReplayStore:
    """call ID → 生 tool ブロックの LRU ストア。"""

    def __init__(
        self,
        max_entries: int = TOOL_REPLAY_DEFAULT_MAX_ENTRIES,
        max_bytes: int = TOOL_REPLAY_DEFAULT_MAX_BYTES,
        filepath: str | None = None,
    ):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.filepath = filepath
        self._by_id: dict[str, str] = {}
        self._order: list[str] = []  # LRU 順（先頭が最新）
        self._bytes = 0
        if filepath:
            self.load()

    # ---- 状態 ----

    @property
    def count(self) -> int:
        return len(self._by_id)

    # ---- 記憶（ds4 tool_memory_remember） ----

    def remember(self, calls: list[dict], raw_text: str) -> None:
        """生成直後の raw text を、そのメッセージ内の全 call に紐付けて記憶する。"""
        if not raw_text:
            return
        for call in calls:
            cid = call.get("id") or call.get("call_id")
            if cid:
                self.remember_id(cid, raw_text)

    def remember_id(self, cid: str, raw_text: str) -> None:
        if not cid or not raw_text:
            return
        if self._by_id.get(cid) == raw_text:
            self._touch(cid)
            return
        old = self._by_id.pop(cid, None)
        if old is not None:
            self._bytes -= self._entry_bytes(cid, old)
            if cid in self._order:
                self._order.remove(cid)
        self._by_id[cid] = raw_text
        self._order.insert(0, cid)
        self._bytes += self._entry_bytes(cid, raw_text)
        self._prune()

    def _touch(self, cid: str) -> None:
        if cid in self._order:
            self._order.remove(cid)
            self._order.insert(0, cid)

    def _prune(self) -> None:
        while self._order and (
            len(self._order) > self.max_entries or self._bytes > self.max_bytes
        ):
            last = self._order.pop()
            text = self._by_id.pop(last, None)
            if text is not None:
                self._bytes -= self._entry_bytes(last, text)

    @staticmethod
    def _entry_bytes(cid: str, text: str) -> int:
        return len(cid) + len(text) + _PER_ENTRY_OVERHEAD

    # ---- 参照（ds4 tool_memory_attach_to_messages） ----

    def exact_block(self, calls: list[dict]):
        """メッセージ内の全 call ID が**同一**生ブロックにマップする場合のみそのブロックを返す。

        いずれかが未知・別ブロックなら None（canonical 描画にフォールバック）。
        """
        matched = None
        for call in calls:
            cid = call.get("id") or call.get("call_id")
            text = self._by_id.get(cid) if cid else None
            if text is None:
                return None
            if matched is not None and matched != text:
                return None
            matched = text
        return matched

    def render_tool_calls(self, calls: list[dict]) -> str:
        """メッセージ内の tool_calls を render する。

        - exact 一致時: 生ブロックを**1 回だけ**出力（複数 call 共有でも重複しない）
        - それ以外: GLM canonical 形式を call ごとに出力（call 間は `\\n\\n`）
        """
        block = self.exact_block(calls)
        if block is not None:
            return block
        parts = [self.canonical_glm(c) for c in calls]
        return "\n\n".join(parts)

    @staticmethod
    def canonical_glm(call: dict) -> str:
        """GLM canonical 形式（ds4 append_glm_tool_calls_text 相当）。"""
        fn = call.get("function", call)
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        try:
            obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            if not isinstance(obj, dict):
                obj = {}
        except (ValueError, TypeError):
            obj = {}
        text = f"<tool_call>{name}"
        for k, v in obj.items():
            text += f"<arg_key>{k}</arg_key><arg_value>{_glm_arg_value(v)}</arg_value>"
        return text + "</tool_call>"

    # ---- 永続化（従来形式 {id: raw_text} と互換） ----

    def load(self) -> None:
        if not self.filepath or not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            self._by_id = {}
            self._order = []
            self._bytes = 0
            for cid, text in data.items():
                if isinstance(text, str):
                    self._by_id[cid] = text
                    self._order.insert(0, cid)
                    self._bytes += self._entry_bytes(cid, text)
            self._prune()
        except (OSError, ValueError) as e:
            print(
                f"[TOOL-REPLAY] 読み込み失敗（無視）: {self.filepath}: {e}",
                file=__import__("sys").stderr,
                flush=True,
            )

    def persist(self) -> None:
        if not self.filepath:
            return
        try:
            d = os.path.dirname(self.filepath)
            if d:
                os.makedirs(d, exist_ok=True)
                os.chmod(d, 0o700)
            tmp = self.filepath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._by_id, f, ensure_ascii=False)
            os.replace(tmp, self.filepath)
            os.chmod(self.filepath, 0o600)
        except OSError as e:
            print(
                f"[TOOL-REPLAY] 永続化失敗: {self.filepath}: {e}",
                file=__import__("sys").stderr,
                flush=True,
            )

    # ---- 生ブロック抽出（ds4 parse_*_generated_message_ex 相当） ----

    @staticmethod
    def extract_raw_block(text: str) -> str | None:
        """生成テキストから、最初の tool マーカーから最後の tool 終端までの
        **連続した**生ブロックを取り出す。複数 tool_call を含む場合は全体を返す。

        - GLM: `<tool_call>...</tool_call>`（直前の `\\n\\n` があれば含む）
        - pipe: `<|tool_call|>...<tool_call|>` / `<|tool_call>...<tool_call|>`
        """
        pairs = [
            ("<tool_call>", "</tool_call>"),
            ("<|tool_call|>", "<tool_call|>"),
            ("<|tool_call>", "<tool_call|>"),
        ]
        best = None
        for start, end in pairs:
            idx = text.find(start)
            if idx == -1:
                continue
            if best is None or idx < best[0]:
                best = (idx, start, end)
        if best is None:
            return None
        start_idx, start, end = best

        block_start = start_idx
        if start_idx >= 2 and text[start_idx - 2 : start_idx] == "\n\n":
            block_start = start_idx - 2

        search_from = start_idx + len(start)
        last_end = None
        while True:
            e = text.find(end, search_from)
            if e == -1:
                break
            last_end = e + len(end)
            search_from = last_end
        if last_end is None:
            return None
        return text[block_start:last_end]
