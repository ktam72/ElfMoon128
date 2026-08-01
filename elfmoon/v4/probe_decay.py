#!/usr/bin/env python3
"""decode の内訳プローブ: FFN / Attention を無効化して本番パスの基底コストを測る。

MLX の遅延実行では、各層の mx.eval(flat) が全上流を同期実行するため、
フェーズ別タイマーは信頼できない。代わりに部品を無効化した差分で測る。

使い方:
  python3 -m elfmoon.v4.probe_decay --ffn-off
  python3 -m elfmoon.v4.probe_decay --attn-off
  python3 -m elfmoon.v4.probe_decay            (フル)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MLX)
    ap.add_argument("--capacity", type=int, default=2000)
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--ffn-off", action="store_true")
    ap.add_argument("--attn-off", action="store_true")
    ap.add_argument("--indexer-off", action="store_true")
    ap.add_argument("--compressor-off", action="store_true")
    ap.add_argument("--no-barrier", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    from elfmoon.v4.ref_model import build_args
    from elfmoon.v4.mlx_v4 import load_v4_streaming

    if args.bf16:
        import elfmoon.v4.mlx_v4 as M

        def _bf16_call(self, x, fp32=True):
            if self.qkind is not None:
                x = M.mlx_act_quant(x, 128, pow2_scale=True)
            return (x @ self.weight.T).astype(M.mx.bfloat16)

        M.MLXLinear.__call__ = _bf16_call

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    margs = build_args(cfg)
    np.random.seed(0)

    model, cache, store = load_v4_streaming(args.model, margs, capacity=args.capacity)

    for b in model.blocks:
        if args.ffn_off:
            b.ffn.forward = lambda x, ids, b=b: (
                setattr(b.ffn, "debug", {}),
                x,
            )[1]
        if args.attn_off:
            b.attn.forward = lambda x, start_pos, b=b: (
                setattr(b.attn, "debug", {}),
                x,
            )[1]
        if args.indexer_off:
            b.attn.indexer = None
        if args.compressor_off:
            if hasattr(b.attn, "compressor"):
                b.attn.compressor.forward = lambda x, start_pos, b=b: None
            b.attn.indexer = None
        if args.no_barrier:
            from elfmoon.v4.mlx_v4 import _decode_ffn_core

            def _fixed(xf, weights, indices, topk, ffn=b.ffn):
                exps = [ffn._load(e) for e in range(6)]
                w_gu = mx.stack(
                    [e["gate.wq"] for e in exps] + [e["up.wq"] for e in exps]
                )
                s_gu = mx.stack([e["gate.s"] for e in exps] + [e["up.s"] for e in exps])
                b_gu = mx.stack([e["gate.b"] for e in exps] + [e["up.b"] for e in exps])
                w_dw = mx.stack([e["down.wq"] for e in exps])
                s_dw = mx.stack([e["down.s"] for e in exps])
                b_dw = mx.stack([e["down.b"] for e in exps])
                return _decode_ffn_core(
                    xf,
                    w_gu,
                    s_gu,
                    b_gu,
                    w_dw,
                    s_dw,
                    b_dw,
                    weights.flatten(),
                    ffn.swiglu_limit,
                    ffn.group_size,
                    ffn.bits,
                )

            b.ffn._decode = _fixed

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

    all_ids = list(ids)
    input_ids = mx.array(np.array([all_ids], dtype=np.int32), mx.int32)
    logits = model.forward(input_ids, start_pos=0)
    next_tok = int(mx.argmax(logits[0, -1]).item())
    all_ids.append(next_tok)

    for _ in range(8):
        cur = mx.array(np.array([[all_ids[-1]]], dtype=np.int32), mx.int32)
        logits = model.forward(cur, start_pos=len(all_ids) - 1)
        next_tok = int(mx.argmax(logits[0, -1]).item())
        all_ids.append(next_tok)

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
    label = []
    if args.ffn_off:
        label.append("ffn-off")
    if args.attn_off:
        label.append("attn-off")
    print(
        f"[{'/'.join(label) if label else 'full'}] {args.tokens}tok "
        f"{len(times) / t_total:.2f} t/s ({t_total / len(times) * 1000:.1f} ms/tok)"
    )


if __name__ == "__main__":
    main()
