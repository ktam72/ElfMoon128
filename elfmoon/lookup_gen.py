"""prompt-lookup 投機デコード（ElfMoonCoder からの移植）。

コンテキスト内の n-gram を検索して将来トークンを候補とし、1 回の forward で
まとめて検証する。貪欲デコード（temp=0）時のみ出力が通常経路と等価になるため、
既定は temp=0 のときだけ有効。受容率が低いタスクでは自動で通常デコードへ
フォールバックする（外れた候補の検証は純粋な損になるため）。

モデルは全層 trim 可能（純 attention）またはハイブリッド（ArraysCache 混在）
の両方に対応する。ハイブリッドでは検証ブロック開始時の再帰状態を退避し、
外れた候補は退避参照に戻して巻き戻す（KV は trim で戻す）。

パラメータ:
  LOOKUP_NGRAM_MAX / LOOKUP_NGRAM_MIN  候補検索の n-gram 長
  LOOKUP_BLOCK                         1 回の forward で検証する候補トークン数
  LOOKUP_WINDOW / LOOKUP_MIN_ACCEPT    自動フォールバックの判定（直近平均受容数）
  ELFMOON_LOOKUP=0                     無効化
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, trim_prompt_cache

LOOKUP_NGRAM_MAX = 8
LOOKUP_NGRAM_MIN = 3
LOOKUP_BLOCK = 16
LOOKUP_WINDOW = 4
LOOKUP_MIN_ACCEPT = 1.0
PENALTY_CONTEXT = 64


def lookup_candidate(
    seq, n_max=LOOKUP_NGRAM_MAX, n_min=LOOKUP_NGRAM_MIN, k=LOOKUP_BLOCK
):
    """seq の末尾 n-gram を seq 中から探し、続き k トークンを候補として返す。

    長い n-gram を優先し、見つかった中で最も新しい出現を採る（編集タスクでは
    直近の文脈ほど当たりやすい）。候補が無ければ空リスト。
    """
    for n in range(n_max, n_min - 1, -1):
        if len(seq) < n + 1:
            continue
        tail = seq[-n:]
        for i in range(len(seq) - n - 1, -1, -1):
            if seq[i : i + n] == tail:
                cand = seq[i + n : i + n + k]
                if cand:
                    return cand
    return []


@dataclass
class LookupOut:
    """stream_generate 互換の出力オブジェクト。"""

    text: str
    token: int
    from_draft: bool = True
    generation_tokens: int = 0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    finish_reason: Optional[str] = None
    # 追加統計
    lookup_blocks: int = 0
    lookup_accepted: int = 0


def _snapshot_cache(cache):
    """巻き戻し用に trim できない層（ArraysCache）の状態だけを退避する。

    配列は毎ステップ新しいオブジェクトに置換される実装のため、参照の保持で
    コピー不要のスナップショットになる（ElfMoonCoder の実測: 30層で約64MB固定）。
    """
    arrays = [
        (i, list(c.state)) for i, c in enumerate(cache) if isinstance(c, ArraysCache)
    ]
    if not arrays:
        return None
    return arrays


def _restore_cache(cache, arrays, cur_pos: int, target_pos: int) -> bool:
    """cache を target_pos まで巻き戻す（KV trim + ArraysCache 復元）。"""
    back = cur_pos - target_pos
    if back < 0:
        return False
    if back > 0:
        trimmable = [c for c in cache if c.is_trimmable()]
        if trim_prompt_cache(trimmable, back) != back:
            return False
    for i, state in arrays:
        cache[i].state = state
    return True


def _verify_block(model, tid, cand, cache, tokens, pos, *, sampler, procs):
    """[tid] + cand を 1 回の forward で検証する（prompt-lookup の中核）。

    戻り値は (次トークンの配列, 受容トークン列, 投入トークン数)。
    受容判定は argmax 一致そのものなので、出力は貪欲デコードと完全に等価。
    不一致で余分に投入した分は巻き戻す:
    - 全層 trim 可能なモデル → trim_prompt_cache で戻すだけ
    - ハイブリッド（ArraysCache が trim 不可） → ブロック開始時のスナップ
      ショットに戻し、受容分だけを 1 回の forward で再投入する
    """
    snap = None
    if not all(c.is_trimmable() for c in cache):
        snap = _snapshot_cache(cache)
        if snap is None:
            return None, [], 0  # 巻き戻せない構成 → 投機しない
    seq = [tid] + list(cand)
    lg = model(mx.array(seq)[None], cache=cache)[0]
    outs = []
    for i in range(lg.shape[0]):
        row = lg[i : i + 1]
        if procs:
            ctx = tokens + cand[:i]
            arr = mx.array(ctx[-PENALTY_CONTEXT:])
            for pr in procs:
                row = pr(arr, row)
        outs.append(sampler(row - mx.logsumexp(row, keepdims=True)))
    toks = mx.concatenate(outs)
    mx.eval(toks)
    got = toks.tolist()

    k = 0
    while k < len(cand) and got[k] == cand[k]:
        k += 1
    accepted = list(cand[:k])
    extra = len(cand) - k  # 余分に投入したトークン数
    if extra:
        if snap is None:
            trim_prompt_cache([c for c in cache if c.is_trimmable()], extra)
        else:
            _restore_cache(cache, snap, pos + len(seq), pos)
            # 受容分だけ再投入（logits は既に得ているので捨てる）
            mx.eval(model(mx.array(seq[: k + 1])[None], cache=cache))
    return toks[k : k + 1], accepted, k + 1


def stream_generate_lookup(
    model,
    tokenizer,
    prompt,
    max_tokens: int = 16384,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[list] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 4096,
    eos_ids_opt: Optional[list[int]] = None,
    enable_lookup: bool = True,
    kv_manager=None,
):
    """prompt-lookup 投機デコード付きのストリーミング生成。

    prompt はトークン列（list[int]）。prompt_cache は既に履歴分が投入済みの
    キャッシュ（kv_manager から復元したもの）。ここでは残り prompt 分を
    prefill してから生成する。

    yield するのは LookupOut（.text は差分テキスト）。
    """
    assert prompt_cache is not None, (
        "prompt_cache は必須（履歴分を投入済みのキャッシュ）"
    )
    prompt_ids = list(prompt)
    eos_src: list[int] = (
        eos_ids_opt
        if eos_ids_opt is not None
        else (getattr(tokenizer, "eos_token_ids", []) or [tokenizer.eos_token_id])
    )
    eos_ids: list[int] = list(eos_src)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    prev_text = ""

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    procs = logits_processors or []
    use_lookup = bool(enable_lookup)

    # prefill: 残り prompt 分
    x = mx.array(prompt_ids)
    n = len(prompt_ids)
    if n > 1:
        total = n
        processed = 0
        while total - processed > 1:
            remaining = (total - processed) - 1
            step = min(prefill_step_size, remaining)
            model(x[None, processed : processed + step], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            mx.clear_cache()
            processed += step
        y, logprobs = _step(model, x[None, processed:], prompt_cache, sampler, procs)
    else:
        y, logprobs = _step(model, x[None], prompt_cache, sampler, procs)

    mx.eval(y)
    gen: list[int] = []
    tic = time.perf_counter()
    n_out = 0
    n_blocks = 0
    n_accepted = 0
    recent = deque(maxlen=LOOKUP_WINDOW)
    finish: Optional[str] = None
    tid: int = -1

    while len(gen) < max_tokens:
        assert y is not None, "decode ループで y が None（内部状態の破損）"
        tid = y.item()
        if tid in eos_ids:
            finish = "stop"
            break
        gen.append(tid)
        new_tokens = [tid]

        if use_lookup and len(gen) < max_tokens:
            cand = lookup_candidate(
                prompt_ids + gen, k=min(LOOKUP_BLOCK, max_tokens - len(gen))
            )
            if cand:
                pos = len(prompt_ids) + len(gen) - 1
                nxt, accepted, fed = _verify_block(
                    model,
                    tid,
                    cand,
                    prompt_cache,
                    gen,
                    pos,
                    sampler=sampler,
                    procs=procs,
                )
                if nxt is None:  # 巻き戻せない構成 → 通常デコードに戻す
                    use_lookup = False
                    y = _next(model, prompt_cache, sampler, procs, gen)
                else:
                    n_blocks += 1
                    n_accepted += len(accepted)
                    gen.extend(accepted)
                    new_tokens.extend(accepted)
                    y = nxt
                    recent.append(len(accepted))
                    if (
                        len(recent) == recent.maxlen
                        and sum(recent) / len(recent) < LOOKUP_MIN_ACCEPT
                    ):
                        use_lookup = False
            else:
                y = _next(model, prompt_cache, sampler, procs, gen)
        elif len(gen) < max_tokens:
            y = _next(model, prompt_cache, sampler, procs, gen)
        else:
            y = None

        if any(t in eos_ids for t in new_tokens):
            cut = next(i for i, t in enumerate(new_tokens) if t in eos_ids)
            del gen[len(gen) - (len(new_tokens) - cut) :]
            new_tokens = new_tokens[:cut]
            finish = "stop"
            y = None

        for t in new_tokens:
            detokenizer.add_token(t)
        text = detokenizer.text
        if text.endswith("\ufffd"):
            continue
        delta = text[len(prev_text) :] if text.startswith(prev_text) else text
        prev_text = text
        if delta:
            n_out += len(new_tokens)
            yield LookupOut(
                text=delta,
                token=tid,
                from_draft=True,
                generation_tokens=n_out,
                generation_tps=n_out / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
                lookup_blocks=n_blocks,
                lookup_accepted=n_accepted,
            )
        if y is None:
            break

    if not (finish == "stop" or y is None):
        finish = "length"
    detokenizer.finalize()
    # 最終チャンクは detokenizer.text 全体との差分で出す（last_segment は
    # finalize 後の挙動が実装依存で欠落しやすい）。
    text = detokenizer.text
    delta = text[len(prev_text) :] if text.startswith(prev_text) else text
    if not delta.endswith("\ufffd"):
        yield LookupOut(
            text=delta,
            token=tid,
            from_draft=True,
            generation_tokens=n_out,
            generation_tps=n_out / (time.perf_counter() - tic) if tic else 0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=finish,
            lookup_blocks=n_blocks,
            lookup_accepted=n_accepted,
        )


def _step(model, xb, cache, sampler, procs):
    """残り prompt の prefill 後に最初のトークンを選択する。"""
    lg = model(xb, cache=cache)[:, -1, :]
    if procs:
        for pr in procs:
            lg = pr(mx.array([], dtype=mx.int32), lg)
    logprobs = lg - mx.logsumexp(lg, keepdims=True)
    return sampler(logprobs), logprobs


def _next(model, cache, sampler, procs, tokens):
    """次のトークンを 1 ステップ生成する（先読みなしの通常デコード）。"""
    cur = mx.array([tokens[-1]])
    lg = model(cur[None], cache=cache)[:, -1, :]
    if procs:
        arr = mx.array(tokens[-PENALTY_CONTEXT:])
        for pr in procs:
            lg = pr(arr, lg)
    nxt = sampler(lg - mx.logsumexp(lg, keepdims=True))
    mx.eval(nxt)
    return nxt
