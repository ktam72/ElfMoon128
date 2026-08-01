#!/usr/bin/env python3
"""4bit 部分常駐ストリーミングの本番パス warm A/B 実測。

DeepSeek-V4-Flash-0731 (MLX 4bit, per-expert store) を ResidentCache で
部分常駐させ、実プロンプトの prefill → 462+ トークン decode を実測する。
容量（常駐 expert 数）を変えて速度・品質・ヒット率・メモリを記録する。

使い方:
  python3 -m elfmoon.v4.bench_streaming --capacity 3850 --max-tokens 480
  python3 -m elfmoon.v4.bench_streaming --capacity 8000 --max-tokens 480
  ELFMOON_MEM_BUDGET_GB=108 python3 -m elfmoon.v4.bench_streaming (自動導出)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import mlx.core as mx

sys.path.insert(0, "/Users/ktam/Documents/apps/ElfMoon128")

MLX = "/Volumes/990Pro_2TB/elfmoon128/models/deepseek-v4-flash-0731-mlx"
ENC = "/Volumes/990Pro_2TB/elfmoon128/models/deepseek-v4-flash-0731/encoding"

PROMPTS = {
    "qa": "日本の首都はどこですか?",
    "math": "1+1は?",
    "long": (
        "次のプログラムについて説明してください。"
        "`func gcd(_ a: Int, _ b: Int) -> Int { return b == 0 ? a : gcd(b, a % b) }`"
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MLX)
    ap.add_argument("--prompt", default="long", help="qa/math/long または生文字列")
    ap.add_argument("--max-tokens", type=int, default=480)
    ap.add_argument("--capacity", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from elfmoon.v4.ref_model import build_args
    from elfmoon.v4.mlx_v4 import load_v4_streaming

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    margs = build_args(cfg)
    np.random.seed(args.seed)

    t0 = time.time()
    model, cache, store = load_v4_streaming(args.model, margs, capacity=args.capacity)
    print(f"ロード {time.time() - t0:.1f}s", flush=True)

    sys.path.insert(0, ENC)
    import encoding_dsv4 as enc

    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(args.model)
    content = PROMPTS.get(args.prompt, args.prompt)
    prompt = enc.encode_messages(
        [{"role": "user", "content": content}], thinking_mode="chat"
    )
    ids = tok.encode(prompt)
    print(f"prompt tokens: {len(ids)}", flush=True)

    all_ids = list(ids)
    prompt_len = len(ids)
    t0 = time.time()
    input_ids = mx.array(np.array([all_ids], dtype=np.int32), mx.int32)
    logits = model.forward(input_ids, start_pos=0)
    next_tok = int(mx.argmax(logits[0, -1]).item())
    all_ids.append(next_tok)
    t_prefill = time.time() - t0
    print(f"prefill {len(ids)}tok {t_prefill:.1f}s", flush=True)

    n = 0
    times = []
    t0 = time.time()
    while n < args.max_tokens - 1:
        cur = mx.array(np.array([[all_ids[-1]]], dtype=np.int32), mx.int32)
        start_pos = len(all_ids) - 1
        logits = model.forward(cur, start_pos=start_pos)
        next_tok = int(mx.argmax(logits[0, -1]).item())
        all_ids.append(next_tok)
        n += 1
        times.append(time.time() - t0)
        t0 = time.time()
        if next_tok == tok.eos_token_id:
            print("[EOS]", flush=True)
            break
    t_total = sum(times)
    warm_times = times[args.warmup :]
    warm_tps = len(warm_times) / sum(warm_times) if warm_times else 0.0
    stats = cache.stats()
    peak = mx.get_peak_memory() / 1024**3
    active = mx.get_active_memory() / 1024**3
    print("=== 結果 ===")
    print(f"capacity      : {stats['capacity']}")
    print(f"decode tokens : {n}")
    print(f"warm t/s      : {warm_tps:.2f}  (全 {n / t_total:.2f} t/s)")
    print(
        f"hit rate      : {stats['hit_rate'] * 100:.1f}%  "
        f"(hit={stats['hits']} miss={stats['misses']})"
    )
    print(f"resident      : {stats['resident']} / {stats['capacity']}")
    print(f"peak mem      : {peak:.1f}GB  active: {active:.1f}GB")
    text = tok.decode(all_ids[prompt_len:])
    print("=== 応答 ===")
    print(text)
    print("===")
    with open("/tmp/v4_bench_result.json", "w") as f:
        json.dump(
            {
                "capacity": stats["capacity"],
                "n_tokens": n,
                "warm_tps": warm_tps,
                "all_tps": n / t_total if t_total else 0,
                "hit_rate": stats["hit_rate"],
                "hits": stats["hits"],
                "misses": stats["misses"],
                "resident": stats["resident"],
                "peak_gb": peak,
                "active_gb": active,
                "prefill_s": t_prefill,
                "text": text,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
