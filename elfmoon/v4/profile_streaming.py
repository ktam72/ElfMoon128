#!/usr/bin/env python3
"""部分常駐ストリーミング decode の層内・フェーズ別タイミング計測。

ELFMOON_PROFILE=1 で MLXBlock / MLXStreamingMoE の累積タイマーを有効化し、
warm 状態での decode を 30 トークン計測してボトルネックを層単位で表示する。

使い方:
  ELFMOON_PROFILE=1 python3 -m elfmoon.v4.profile_streaming --capacity 2000
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


def main():
    os.environ["ELFMOON_PROFILE"] = "1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MLX)
    ap.add_argument("--capacity", type=int, default=2000)
    ap.add_argument("--tokens", type=int, default=30)
    args = ap.parse_args()

    from elfmoon.v4.ref_model import build_args
    from elfmoon.v4.mlx_v4 import load_v4_streaming

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    margs = build_args(cfg)
    np.random.seed(0)

    t0 = time.time()
    model, cache, store = load_v4_streaming(args.model, margs, capacity=args.capacity)
    print(f"ロード {time.time() - t0:.1f}s", flush=True)

    sys.path.insert(0, ENC)
    import encoding_dsv4 as enc

    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(args.model)
    content = (
        "次のプログラムについて説明してください。"
        "`func gcd(_ a: Int, _ b: Int) -> Int { return b == 0 ? a : gcd(b, a % b) }`"
    )
    prompt = enc.encode_messages(
        [{"role": "user", "content": content}], thinking_mode="chat"
    )
    ids = tok.encode(prompt)
    print(f"prompt tokens: {len(ids)}", flush=True)

    all_ids = list(ids)
    input_ids = mx.array(np.array([all_ids], dtype=np.int32), mx.int32)
    logits = model.forward(input_ids, start_pos=0)
    next_tok = int(mx.argmax(logits[0, -1]).item())
    all_ids.append(next_tok)
    print("prefill done", flush=True)

    for _ in range(8):
        cur = mx.array(np.array([[all_ids[-1]]], dtype=np.int32), mx.int32)
        logits = model.forward(cur, start_pos=len(all_ids) - 1)
        next_tok = int(mx.argmax(logits[0, -1]).item())
        all_ids.append(next_tok)
    print("warmup done", flush=True)

    for b in model.blocks:
        b.acc_attn = 0.0
        b.acc_ffn = 0.0
        b.ffn._acc_load = 0.0
        b.ffn._acc_matmul = 0.0
        b.ffn._acc_dispatch = 0.0
        b.ffn._acc_shared = 0.0

    times = []
    t0 = time.time()
    for _ in range(args.tokens):
        cur = mx.array(np.array([[all_ids[-1]]], dtype=np.int32), mx.int32)
        logits = model.forward(cur, start_pos=len(all_ids) - 1)
        next_tok = int(mx.argmax(logits[0, -1]).item())
        all_ids.append(next_tok)
        times.append(time.time() - t0)
        t0 = time.time()
    t_total = sum(times)
    print(
        f"=== decode {args.tokens}tok: {len(times) / t_total:.2f} t/s "
        f"({t_total / len(times) * 1000:.1f} ms/tok) ==="
    )

    def acc(p):
        parts = p.split(".")
        s = 0.0
        for b in model.blocks:
            v = b
            for part in parts:
                v = getattr(v, part)
            s += v
        return s

    attn = acc("acc_attn")
    ffn = acc("acc_ffn")
    load = acc("ffn._acc_load")
    matmul = acc("ffn._acc_matmul")
    dispatch = acc("ffn._acc_dispatch")
    shared = acc("ffn._acc_shared")
    ms = lambda s: s / args.tokens * 1000
    print(f"block 合計   : {ms(attn + ffn):6.2f} ms/tok")
    print(f"  attention : {ms(attn):6.2f} ms/tok ({attn / (attn + ffn) * 100:.1f}%)")
    print(f"  ffn       : {ms(ffn):6.2f} ms/tok ({ffn / (attn + ffn) * 100:.1f}%)")
    tot = load + matmul + dispatch + shared
    if tot:
        print(f"  ffn 内訳 (matmul は load+stack と shared の間の同期区間):")
        print(f"    load+stack : {ms(load):6.2f} ms/tok ({load / tot * 100:.1f}%)")
        print(f"    matmul区間 : {ms(matmul):6.2f} ms/tok ({matmul / tot * 100:.1f}%)")
        print(
            f"    dispatch  : {ms(dispatch):6.2f} ms/tok ({dispatch / tot * 100:.1f}%)"
        )
        print(f"    shared    : {ms(shared):6.2f} ms/tok ({shared / tot * 100:.1f}%)")

    print("\n=== 層別 (attn / ffn / load+stack / matmul  ms per tok) ===")
    for i, b in enumerate(model.blocks):
        l = b.ffn._acc_load / args.tokens * 1000
        m = b.ffn._acc_matmul / args.tokens * 1000
        print(
            f"  L{i:2d}  attn={b.acc_attn / args.tokens * 1000:5.2f}  "
            f"ffn={b.acc_ffn / args.tokens * 1000:5.2f}  "
            f"load={l:5.2f}  matmul={m:5.2f}"
        )


if __name__ == "__main__":
    main()
