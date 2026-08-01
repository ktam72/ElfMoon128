#!/usr/bin/env python3
"""部分常駐ストリーミング MoE（MLXStreamingMoE）の数値パリティ検証。

融合テンソル経路（MLXFusedMoE, gather_qmm）と per-expert ストリーミング経路
（MLXStreamingMoE, quantized_matmul）が、同一 4bit store 重みで同じ FFN 出力を
返すかを層単位で比較する（ノイズフロア = bf16 丸め差程度を期待）。

使い方: python3 -m elfmoon.v4.verify_streaming --layers 4,5,6
"""

import argparse
import json
import os
import sys

import numpy as np

import mlx.core as mx

sys.path.insert(0, "/Users/ktam/Documents/apps/ElfMoon128")

MLX = "/Volumes/990Pro_2TB/elfmoon128/models/deepseek-v4-flash-0731-mlx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MLX)
    ap.add_argument("--layers", default="4,5,6")
    ap.add_argument("--capacity", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from elfmoon.v4.ref_model import build_args
    from elfmoon.v4.mlx_v4 import (
        MLXV4Model,
        load_v4_fused,
        load_v4_streaming,
    )

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    margs = build_args(cfg)
    np.random.seed(args.seed)

    layer_ids = [int(x) for x in args.layers.split(",")]

    fused_model = load_v4_fused(args.model, margs, layer_ids=layer_ids, streaming=False)
    stream_model = load_v4_streaming(
        args.model, margs, layer_ids=layer_ids, capacity=args.capacity
    )
    s_model: MLXV4Model = stream_model[0]

    mx.eval(fused_model.blocks[0].ffn.shared.w1)
    mx.eval(s_model.blocks[0].ffn.shared.w1)

    # 全層 FFN に同じ入力を与えて出力を比較
    T = args.tokens
    input_ids = mx.array(np.random.randint(100, 1000, (1, T)), mx.int32)
    x = mx.array(np.random.normal(0, 1, (1, T, 4, 4096)), mx.bfloat16)

    print(f"層: {layer_ids}  capacity: {args.capacity}  tokens: {T}")
    fblocks = {b: fb for b, fb in zip(layer_ids, fused_model.blocks)}
    sblocks = {b: sb for b, sb in zip(layer_ids, s_model.blocks)}
    worst = 0.0
    for lid in layer_ids:
        fb = fblocks[lid]
        sb = sblocks[lid]
        yf = fb.ffn.forward(x, input_ids)
        ys = sb.ffn.forward(x, input_ids)
        mx.eval(yf, ys)
        yf_n = np.asarray(yf.astype(mx.float32))
        ys_n = np.asarray(ys.astype(mx.float32))
        diff = np.abs(yf_n - ys_n)
        denom = max(1e-6, np.abs(yf_n).max())
        md = float(diff.max())
        mrel = float(md / denom)
        worst = max(worst, mrel)
        print(
            f"  layer {lid}: max_abs_diff={md:.4f}  rel={mrel:.4%}  "
            f"|y|max={np.abs(yf_n).max():.4f}"
        )
    # ルーティング一致確認
    if "indices" in sblocks[layer_ids[0]].ffn.debug:
        fi = fblocks[layer_ids[0]].ffn.debug["indices"]
        si = sblocks[layer_ids[0]].ffn.debug["indices"]
        print(
            "  routing indices 一致:",
            bool(np.array_equal(np.asarray(fi), np.asarray(si))),
        )
    print(f"判定: worst rel diff = {worst:.4%}")
    print("合格" if worst < 0.02 else "要調査", flush=True)


if __name__ == "__main__":
    main()
