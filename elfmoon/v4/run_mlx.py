#!/usr/bin/env python3
"""層パリティ検証: MLX 実装（mlx_v4.py）を 1 層で実行し、参照データと diff。

使用法:
    python3 -m elfmoon.v4.run_mlx --layer 2 --seqlen 64 [--out refs/layer2_s64]

refs/layer2_s64 に保存された基準テンソル（run_ref.py 生成）と MLX 出力を
比較し、最大差・平均差をテンソルごとに表示する。
"""

import argparse
import json
import os

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

from elfmoon.convert_v4 import decode_fp4, decode_fp8, to_mx
from elfmoon.v4.mlx_v4 import (
    MLXV4Model,
    build_mlx_layer,
)
from elfmoon.v4.ref_model import build_args
from elfmoon.v4.run_ref import decode_shard, keys_map_for_layer, shard_of_key

SRC = "/Volumes/990Pro_2TB/elfmoon128/models/deepseek-v4-flash-0731"


def mx_to_np(a):
    """mx.array -> float32 numpy。bf16 は fp32 へ。"""
    if a.dtype == mx.int64:
        return np.asarray(a, dtype=np.int64)
    return np.asarray(a.astype(mx.float32), dtype=np.float32)


def mx_flat_debug(model):
    """MLXBlock の debug を ref と同じキー名の dict（numpy）へ。"""
    out = {}
    blk = model.blocks[0]
    for k, v in blk.debug.items():
        if k == "attn":
            for ak, av in v.items():
                out[f"attn.{ak}"] = av
        elif k == "ffn":
            for fk, fv in v.items():
                out[f"ffn.{fk}"] = fv
        else:
            out[k] = v
    return {k: mx_to_np(v) for k, v in out.items()}


def diff_report(ref, got, keys):
    """ref と got（numpy dict）を比較してレポートする。"""
    rows = []
    for k in keys:
        if k not in ref or k not in got:
            rows.append((k, "MISSING", "", ""))
            continue
        r, g = ref[k], got[k]
        if r.shape != g.shape:
            rows.append((k, f"SHAPE {r.shape} vs {g.shape}", "", ""))
            continue
        d = np.abs(r.astype(np.float64) - g.astype(np.float64))
        rows.append((k, "ok", f"{d.max():.3e}", f"{d.mean():.3e}"))
    w = max(len(k) for k, _, _, _ in rows) + 1
    print(f"\n{'key':<{w}}{'status':<12}{'max':>12}{'mean':>12}")
    print("-" * (w + 36))
    for k, s, ma, me in rows:
        print(f"{k:<{w}}{s:<12}{ma:>12}{me:>12}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=64)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--expert-quant", action="store_true", help="MLX 4bit(group64) expert"
    )
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(SRC, "config.json")))
    margs = build_args(cfg)

    shard = shard_of_key(f"layers.{args.layer}.ffn.experts.0.w1.weight")
    print(f"load {shard} (layer {args.layer})...", flush=True)
    state = decode_shard(shard)
    try:
        esh = shard_of_key("embed.weight")
        if os.path.exists(os.path.join(SRC, esh)):
            import safetensors.torch as st

            state["embed.weight"] = st.load_file(os.path.join(SRC, esh), device="cpu")[
                "embed.weight"
            ]
            print(f"embed 読込: {esh}")
    except Exception as e:
        print(f"embed 未読込: {e}")

    w = {"layers": {}, "embed": to_mx(state["embed.weight"])}
    for k in state:
        if not k.startswith(f"layers.{args.layer}."):
            continue
        if k in w["layers"]:
            continue
        w["layers"][k] = to_mx(state[k])
    state_mx = {"embed.weight": w["embed"]}
    state_mx.update(w["layers"])
    w["layers"] = {
        args.layer: build_mlx_layer(margs, state_mx, args.layer, args.expert_quant)
    }
    model = MLXV4Model(margs, w, layer_ids=[args.layer])

    outdir = args.out or os.path.join(
        os.path.dirname(__file__), "refs", f"layer{args.layer}_s{args.seqlen}"
    )
    refs = {
        k: np.load(os.path.join(outdir, f"ref_{k}.npy"))
        for k in (
            "input_ids",
            "h_in",
            "h_out",
            "dec_inputs",
            "h_dec",
            "pre_attn",
            "attn_norm",
            "attn_out",
            "attn.q",
            "attn.kv",
            "attn.topk_idxs",
            "attn.o_woa",
            "attn.out",
            "hc_post1",
            "pre_ffn",
            "ffn_norm",
            "ffn_out",
            "ffn.weights",
            "ffn.indices",
            "hc_post2",
        )
    }
    print(f"refs: {outdir}")

    input_ids = mx.array(refs["input_ids"], mx.int32)
    h_in = mx.array(refs["h_in"], mx.bfloat16)

    h_emb = mx.take(model.embed, input_ids, axis=0)
    h_emb = mx.broadcast_to(
        h_emb[:, :, None, :],
        (h_emb.shape[0], h_emb.shape[1], margs.hc_mult, h_emb.shape[2]),
    )
    print("embed diff (h_in):")
    embed_rows = diff_report(
        {"h_in": refs["h_in"]},
        {"h_in": mx_to_np(h_emb)},
        ["h_in"],
    )

    print("\n--- prefill (start_pos=0) ---")
    h_out = model.forward_block(h_in, input_ids, start_pos=0)
    got = mx_flat_debug(model)
    got["h_out"] = mx_to_np(h_out)
    keys = [
        "pre_attn",
        "attn_norm",
        "attn_out",
        "attn.q",
        "attn.kv",
        "attn.topk_idxs",
        "attn.o_woa",
        "attn.out",
        "hc_post1",
        "pre_ffn",
        "ffn_norm",
        "ffn_out",
        "ffn.weights",
        "ffn.indices",
        "hc_post2",
        "h_out",
    ]
    diff_report(refs, got, keys)

    print("\n--- decode (n_dec=4) ---")
    h_dec = []
    for i in range(refs["dec_inputs"].shape[1]):
        tok = mx.array(refs["dec_inputs"][:, i : i + 1], mx.int32)
        h = mx.take(model.embed, tok, axis=0)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], margs.hc_mult, h.shape[2]),
        )
        hc = model.forward_block(h, tok, start_pos=args.seqlen + i)
        h_dec.append(mx_to_np(hc))
    h_dec = np.stack(h_dec, axis=1)
    print(
        f"h_dec  max_diff: {np.abs(refs['h_dec'].astype(np.float64) - h_dec).max():.3e}"
    )


if __name__ == "__main__":
    main()
