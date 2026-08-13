"""prompt cache（退避点）の TTFT 効果を測る（ElfMoonCoder bench_prompt_cache 相当）。

think / no-think 両モードで、ターン2 の TTFT と KVC 再利用率を比較する。
修正（--no-think 反映）がキャッシュ再利用に影響しないことを確認するのが目的。

usage:
  python3 bench/bench_prompt_cache.py [モデル名]
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # elfmoon/
sys.path.insert(0, str(ROOT))
os.environ["ELFMOON_KVC_LOG"] = "0"

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from kv_manager import kv_manager  # noqa: E402
from lookup_gen import stream_generate_lookup  # noqa: E402

CODE = "\n".join(
    f"def handler_{i}(request, context):\n"
    f"    data = request.get('payload', {{}})\n"
    f"    return {{'id': {i}, 'ok': True, 'data': data}}"
    for i in range(500)
)
GEN = dict(max_tokens=32)


def build_prompt(tok, msgs, thinking):
    return tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=thinking
    )


def run(model, tok, prompt_ids, thinking, label, use_lookup=False):
    """KVC 再利用なし（毎回フル prefill）での TTFT と prefill 時間。"""
    sampler = make_sampler(temp=0.0, top_p=1.0, min_p=0.0)
    pc = make_prompt_cache(model)
    t0 = time.perf_counter()
    first = None
    n = 0
    # prefill 全体を投入してから生成開始（TTFT 計測用）
    for r in stream_generate_lookup(
        model,
        tok,
        prompt_ids,
        max_tokens=32,
        sampler=sampler,
        prompt_cache=pc,
        prefill_step_size=4096,
        enable_lookup=use_lookup,
    ):
        if first is None:
            first = time.perf_counter() - t0
        n += len(r.text)
    print(
        f"  {label:24} TTFT {first * 1000:7.0f} ms  gen {n} chars  ({time.perf_counter() - t0:.2f}s)"
    )
    return first


def main():
    from stream_model import MODELS_ROOT

    name = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
    )
    model_path = os.path.join(MODELS_ROOT, name)
    model, tok = load(model_path, lazy=True)
    kv_manager.clear()
    kv_manager.clear_disk()
    kv_manager.set_namespace(os.path.basename(name))

    turn1 = [
        {"role": "user", "content": f"以下のコードを読んでください:\n{CODE}\n概要は？"}
    ]

    print(f"モデル: {name}\n")

    for thinking in (True, False):
        label = "think" if thinking else "no-think"
        print(f"[{label} モード]")
        p1 = build_prompt(tok, turn1, thinking)
        ids1 = tok.encode(p1)
        text1 = ""
        pc = make_prompt_cache(model)
        sampler = make_sampler(temp=0.0, top_p=1.0, min_p=0.0)
        for r in stream_generate_lookup(
            model,
            tok,
            ids1,
            max_tokens=32,
            sampler=sampler,
            prompt_cache=pc,
            prefill_step_size=4096,
            enable_lookup=False,
        ):
            text1 += r.text
        turn2 = turn1 + [
            {"role": "assistant", "content": text1},
            {"role": "user", "content": "2 番目の関数の戻り値は？"},
        ]
        p2 = build_prompt(tok, turn2, thinking)
        ids2 = tok.encode(p2)
        ntok2 = len(ids2)

        # ターン2 をフル prefill（KVC なし）
        run(model, tok, ids2, thinking, "ターン2 (KVCなし)")

        # ターン2 を KVC 経由（履歴 prefill はキャッシュ、残りだけ）
        nogen = tok.apply_chat_template(
            turn2,
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=thinking,
        )
        nogen_ids = tok.encode(nogen)
        boundary = 0
        for i in range(min(len(nogen_ids), len(ids2))):
            if nogen_ids[i] != ids2[i]:
                break
            boundary = i + 1
        cached, cached_len = kv_manager.lookup(ids2, model)
        if cached is not None:
            pc2 = cached
            reuse = cached_len
        else:
            pc2 = make_prompt_cache(model)
            reuse = 0
        kv_manager.set_live_cache(pc2)
        kv_manager._session_tokens = list(ids2[:reuse])
        t0 = time.perf_counter()
        first = None
        acc = reuse
        if reuse < boundary:
            for ci in range(reuse, boundary, 4096):
                chunk = ids2[ci : ci + 4096]
                model(mx.array([chunk]), cache=pc2)
                acc += len(chunk)
                kv_manager.add_snapshot(pc2, acc)
        snap = kv_manager.snapshot(pc2)
        kv_manager.save(ids2[:boundary], snap)
        sampler = make_sampler(temp=0.0, top_p=1.0, min_p=0.0)
        t1 = time.perf_counter()
        for r in stream_generate_lookup(
            model,
            tok,
            ids2[boundary:],
            max_tokens=32,
            sampler=sampler,
            prompt_cache=pc2,
            prefill_step_size=4096,
            enable_lookup=False,
        ):
            if first is None:
                first = time.perf_counter() - t1
        pf_time = time.perf_counter() - t0 - (first or 0)
        print(
            f"  {'ターン2 (KVCあり)':24} TTFT {first * 1000:7.0f} ms  "
            f"再利用 {reuse} tok  prefill {boundary - reuse} tok"
        )
        print()


if __name__ == "__main__":
    main()
