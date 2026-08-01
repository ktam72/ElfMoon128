#!/usr/bin/env python3
"""層パリティ用: 公式参照モデル（純 torch）を 1 層で実行し基準テンソルを保存。

使用法:
    python3 -m elfmoon.v4.run_ref --layer 2 --seqlen 64 [--out refs/l2]

基準データ（npy）に prefill 出力・各段の中間値（debug）・decode 出力を保存する。
MLX 実装（run_mlx.py）と diff して収束確認に使う。
"""

import argparse
import json
import os

import numpy as np
import safetensors.torch as st
import torch

from elfmoon.convert_v4 import decode_fp4, decode_fp8
from elfmoon.v4.ref_model import Transformer, build_args, load_layer_state

SRC = "/Volumes/990Pro_2TB/elfmoon128/models/deepseek-v4-flash-0731"


def shard_of_key(key: str) -> str:
    idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    return idx["weight_map"][key]


def decode_shard(shard: str):
    """fp8/fp4 を bf16 へデコードした state dict を返す。"""
    m = st.load_file(os.path.join(SRC, shard), device="cpu")
    out = {}
    for k, t in m.items():
        if k.endswith(".scale"):
            continue
        sk = k.replace(".weight", ".scale")
        if t.dtype in (torch.int8, torch.uint8) and sk in m:
            out[k] = decode_fp4(t, m[sk])
        elif t.dtype == torch.float8_e4m3fn:
            out[k] = decode_fp8(t, m[sk])
        else:
            out[k] = t
    return out


def keys_map_for_layer(layer: int, state: dict):
    """チェックポイントキー -> モデル内パス（layers.0 に置換）。"""
    km = {}
    for k in state:
        if k.startswith(f"layers.{layer}."):
            mp = "layers.0." + k[len(f"layers.{layer}.") :]
            km[mp] = k
    return km


def flatten_debug(model):
    """model.debug をフラットな dict（numpy）へ。"""
    out = {}
    blk = model.layers[0]
    for k, v in blk.debug.items():
        if k == "attn":
            for ak, av in v.items():
                out[f"attn.{ak}"] = av
        elif k == "ffn":
            for fk, fv in v.items():
                out[f"ffn.{fk}"] = fv
        else:
            out[k] = v
    return {k: v.float().numpy() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=64)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--n_dec", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(SRC, "config.json")))
    margs = build_args(cfg)

    shard = shard_of_key(f"layers.{args.layer}.ffn.experts.0.w1.weight")
    print(f"load {shard} (layer {args.layer})...", flush=True)
    state = decode_shard(shard)
    try:
        esh = shard_of_key("embed.weight")
        if os.path.exists(os.path.join(SRC, esh)):
            state["embed.weight"] = st.load_file(os.path.join(SRC, esh), device="cpu")[
                "embed.weight"
            ]
            print(f"embed 読込: {esh}")
    except Exception as e:
        print(f"embed 未読込: {e}")

    torch.set_num_threads(8)
    model = Transformer(margs, layer_ids=[args.layer])
    km = keys_map_for_layer(args.layer, state)
    km["embed.weight"] = "embed.weight"
    load_layer_state(model, state, km)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg["vocab_size"], (args.bsz, args.seqlen))
    refs = {}
    refs["input_ids"] = input_ids.numpy()
    refs["dec_inputs"] = torch.randint(
        0, cfg["vocab_size"], (args.bsz, args.n_dec)
    ).numpy()

    torch.manual_seed(0)
    with torch.no_grad():
        h = model.embed(input_ids).unsqueeze(2).repeat(1, 1, margs.hc_mult, 1)
        refs["h_in"] = h.float().numpy()
        h_out = model.forward_block(h, input_ids, start_pos=0)
        refs["h_out"] = h_out.float().numpy()
        refs.update(flatten_debug(model))

        dec_out = []
        hcur = None
        for i in range(args.n_dec):
            tok = torch.from_numpy(refs["dec_inputs"][:, i : i + 1])
            h = model.embed(tok).unsqueeze(2).repeat(1, 1, margs.hc_mult, 1)
            hcur = model.forward_block(h, tok, start_pos=args.seqlen + i)
            dec_out.append(hcur.float().numpy())
        refs["h_dec"] = np.stack(dec_out, axis=1)

    outdir = args.out or os.path.join(
        os.path.dirname(__file__), "refs", f"layer{args.layer}_s{args.seqlen}"
    )
    os.makedirs(outdir, exist_ok=True)
    for k, v in refs.items():
        np.save(os.path.join(outdir, f"ref_{k}.npy"), v)
    print(f"基準保存: {outdir}")
    for k, v in refs.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
