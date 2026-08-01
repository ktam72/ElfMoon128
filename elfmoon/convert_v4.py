#!/usr/bin/env python3
"""DeepSeek-V4-Flash-0731 -> ElfMoon128 store/融合形式 変換器。

ソース（公式チェックポイント, fp4/fp8）を読み、
- 全テンソルを bf16/fp32/int へデコード
- MoE expert のみ MLX 4bit (group_size=64) に再量子化
- 融合形式 `{prefix}.layers.{l}.ffn.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`
- store `l{l}_e{e}.safetensors`（ExpertStore 互換, gate=w1, up=w3, down=w2）

を直接出力する。integrate.py の事後実行は不要。
"""

import argparse
import json
import os
import re
import time

import numpy as np
import safetensors.torch as st
import torch

import mlx.core as mx

PROJ = ("w1", "w2", "w3")
FP4_TABLE = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)
_LOW = FP4_TABLE[
    (np.arange(256, dtype=np.uint16) & 0x0F).astype(np.uint8).astype(np.int64)
]
_HIGH = FP4_TABLE[
    ((np.arange(256, dtype=np.uint16) >> 4) & 0x0F).astype(np.uint8).astype(np.int64)
]


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """e8m0 scale -> float32 (2^(E-127))。"""
    e = scale.view(torch.uint8).float()
    return torch.pow(2, e - 127)


def decode_fp4(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp4 (int8 2nibbleパック) -> bf16 [out, in]。

    nibble順: low が先、high が後（公式 convert.py と一致）。
    scale: e8m0 [out, in/32]、K 方向 32 要素毎。
    """
    out, inp = w.shape
    in_dim = inp * 2
    b = w.view(torch.uint8).numpy()
    vals = np.empty((out, in_dim), dtype=np.float32)
    vals[:, 0::2] = _LOW[b]
    vals[:, 1::2] = _HIGH[b]
    s = e8m0_to_float(scale).numpy()
    s32 = np.repeat(s, 32, axis=1)
    return torch.from_numpy(vals * s32).to(torch.bfloat16)


def decode_fp8(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """e4m3 + e8m0(128x128ブロック) -> bf16 [out, in]。"""
    b_out, b_in = w.shape[0] // 128, w.shape[1] // 128
    s = e8m0_to_float(scale)
    x = w.float().view(b_out, 128, b_in, 128) * s[:, None, :, None]
    return x.reshape(w.shape).bfloat16()


def quant_expert(
    w: torch.Tensor | mx.array, bits: int = 4
) -> tuple[mx.array, mx.array, mx.array]:
    """bf16 [out, in] -> MLX N bit (group_size=64) 量子化。torch/mx 両対応。"""
    if isinstance(w, mx.array):
        m = w.astype(mx.bfloat16)
    else:
        m = mx.array(w.float().numpy(), dtype=mx.bfloat16)
    q, s, b = mx.quantize(m, group_size=64, bits=bits)
    return q, s, b


def to_mx(t: torch.Tensor) -> mx.array:
    """torch テンソル（bf16/f32/i64）を mx.array へ。bf16 は fp32 経由で保持。"""
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().numpy(), dtype=mx.bfloat16)
    return mx.array(t.numpy())


class Converter:
    def __init__(self, src, dst, layers=None, bits=4):
        self.src = src
        self.dst = dst
        self.bits = bits
        self.cfg = json.load(open(os.path.join(src, "config.json")))
        self.idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
        self.wm = self.idx["weight_map"]
        self.n_heads = self.cfg["num_attention_heads"]
        self.head_dim = self.cfg["head_dim"]
        self.o_groups = self.cfg["o_groups"]
        self.n_exp = self.cfg["n_routed_experts"]
        self.n_hash = self.cfg["num_hash_layers"]
        self.layers = (
            layers if layers is not None else range(self.cfg["num_hidden_layers"])
        )
        self.wm_out = {}
        os.makedirs(os.path.join(dst, "store"), exist_ok=True)

    def load_shard(self, shard):
        return st.load_file(os.path.join(self.src, shard), device="cpu")

    def shard_of(self, prefix):
        m = re.search(r"layers\.(\d+)\.", prefix)
        layer = int(m.group(1))
        return self.wm[f"layers.{layer}.ffn.experts.0.w1.weight"].rsplit("/", 1)[-1]

    def decode_experts(self, m, layer):
        """256 experts の w1/w2/w3 を fp4 デコード -> MLX 4bit 量子化。

        戻り値: proj -> (fused wq/s/b のリスト[256], store 内容)。
        """
        fused = {p: [] for p in PROJ}
        store_arrays = []
        gate_ws = []
        base = f"layers.{layer}.ffn.experts"
        for e in range(self.n_exp):
            entry = {}
            for p in PROJ:
                w = m[f"{base}.{e}.{p}.weight"]
                sc = m[f"{base}.{e}.{p}.scale"]
                dec = decode_fp4(w, sc)
                q, s, b = quant_expert(dec, self.bits)
                fused[p].append((q, s, b))
                entry[f"{p}"] = (q, s, b)
            store_arrays.append(entry)
        return fused, store_arrays

    def decode_non_experts(self, m, layer):
        out = {}
        base = f"layers.{layer}"
        # attn
        for t in ("wq_a", "wq_b", "wkv", "wo_b"):
            k = f"{base}.attn.{t}.weight"
            out[k] = decode_fp8(m[k], m[f"{base}.attn.{t}.scale"])
        # wo_a は o_groups 分割 (行方向 4096 毎)
        woa = decode_fp8(m[f"{base}.attn.wo_a.weight"], m[f"{base}.attn.wo_a.scale"])
        dg = woa.shape[0] // self.o_groups
        for g in range(self.o_groups):
            out[f"{base}.attn.wo_a.{g}.weight"] = woa[
                g * dg : (g + 1) * dg
            ].contiguous()
        for t in ("q_norm", "kv_norm"):
            out[f"{base}.attn.{t}.weight"] = m[f"{base}.attn.{t}.weight"]
        out[f"{base}.attn.attn_sink"] = m[f"{base}.attn.attn_sink"]
        # compressor / indexer
        for grp in ("compressor", "indexer"):
            prefix = f"{base}.attn.{grp}"
            for k in list(m):
                if k.startswith(prefix + ".") and k.endswith(".weight"):
                    t = m[k]
                    out[k] = (
                        decode_fp8(t, m[k.replace(".weight", ".scale")])
                        if t.dtype == torch.float8_e4m3fn
                        else t
                    )
                elif k.startswith(prefix + ".") and k.endswith(".ape"):
                    out[k] = m[k]
        # indexer.wq_b は fp8
        k = f"{base}.attn.indexer.wq_b.weight"
        if k in m:
            out[k] = decode_fp8(m[k], m[k.replace(".weight", ".scale")])
        # norms / gate / shared_experts / hc
        for t in ("attn_norm", "ffn_norm"):
            out[f"{base}.{t}.weight"] = m[f"{base}.{t}.weight"]
        out[f"{base}.ffn.gate.weight"] = m[f"{base}.ffn.gate.weight"]
        if layer < self.n_hash:
            out[f"{base}.ffn.gate.tid2eid"] = m[f"{base}.ffn.gate.tid2eid"]
        else:
            out[f"{base}.ffn.gate.bias"] = m[f"{base}.ffn.gate.bias"]
        for p in PROJ:
            k = f"{base}.ffn.shared_experts.{p}.weight"
            if k in m:
                out[k] = decode_fp8(m[k], m[k.replace(".weight", ".scale")])
        for suf in (
            "attn_fn",
            "attn_base",
            "attn_scale",
            "ffn_fn",
            "ffn_base",
            "ffn_scale",
        ):
            out[f"{base}.hc_{suf}"] = m[f"{base}.hc_{suf}"]
        return out

    def write_store(self, layer, store_arrays):
        store_dir = os.path.join(self.dst, "store")
        for e, entry in enumerate(store_arrays):
            d = {}
            for p, (q, s, b) in entry.items():
                name = {"w1": "gate", "w3": "up", "w2": "down"}[p]
                d[f"{name}.wq"], d[f"{name}.s"], d[f"{name}.b"] = q, s, b
            mx.save_safetensors(
                os.path.join(store_dir, f"l{layer}_e{e}.safetensors"), d
            )

    def write_layer_shard(self, layer, m, fused, non_exp, shard_out):
        tensors = {k: to_mx(v) for k, v in non_exp.items()}
        base = f"layers.{layer}.ffn.switch_mlp"
        for p, name in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
            qs, ss, bs = zip(*fused[p])
            tensors[f"{base}.{name}_proj.weight"] = mx.stack(list(qs), axis=0)
            tensors[f"{base}.{name}_proj.scales"] = mx.stack(list(ss), axis=0)
            tensors[f"{base}.{name}_proj.biases"] = mx.stack(list(bs), axis=0)
        mx.save_safetensors(os.path.join(self.dst, shard_out), tensors)
        for k in tensors:
            self.wm_out[k] = shard_out

    def convert(self):
        t0 = time.time()
        # トップレベル (embed / head / norm / hc_head)
        if not self.layers or 0 not in self.layers or True:
            pass
        self._convert_top_level()
        for layer in self.layers:
            shard = self.wm[f"layers.{layer}.ffn.experts.0.w1.weight"]
            print(f"[layer {layer}] shard={shard} 読込中...", flush=True)
            m = self.load_shard(shard)
            fused, store_arrays = self.decode_experts(m, layer)
            non_exp = self.decode_non_experts(m, layer)
            self.write_store(layer, store_arrays)
            shard_out = shard
            self.write_layer_shard(layer, m, fused, non_exp, shard_out)
            del m, fused, store_arrays
            mx.clear_cache()
            print(
                f"  layer {layer} 完了 ({time.time() - t0:.0f}s)",
                flush=True,
            )
        self._write_index()
        print(f"変換完了 ({time.time() - t0:.0f}s)")

    def _convert_top_level(self):
        for shard, keys in (
            (self.wm["embed.weight"], ("embed.weight",)),
            (
                self.wm["head.weight"],
                (
                    "head.weight",
                    "norm.weight",
                    "hc_head_fn",
                    "hc_head_base",
                    "hc_head_scale",
                ),
            ),
        ):
            src_path = os.path.join(self.src, shard)
            if not os.path.exists(src_path):
                print(f"  top-level shard 未DLのためスキップ: {shard}", flush=True)
                continue
            m = mx.load(src_path, format="safetensors")
            out = {k: m[k] for k in keys}
            mx.save_safetensors(os.path.join(self.dst, shard), out)
            for k in out:
                self.wm_out[k] = shard
            del m

    def _write_index(self):
        """index を書き出す。既存 index（別セッションで変換済みの層）とマージする。"""
        idx_path = os.path.join(self.dst, "model.safetensors.index.json")
        wm = {}
        if os.path.exists(idx_path):
            try:
                wm = json.load(open(idx_path))["weight_map"]
            except Exception:
                wm = {}
        wm.update(self.wm_out)
        with open(idx_path, "w") as f:
            json.dump({"metadata": {"total_size": 0}, "weight_map": wm}, f)


def write_config(src, dst, layers):
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg.pop("quantization_config", None)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            os.makedirs(dst, exist_ok=True)
            import shutil

            shutil.copyfile(p, os.path.join(dst, fn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layer", type=int, nargs="*", default=None)
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4])
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    write_config(args.src, args.dst, args.layer)
    c = Converter(args.src, args.dst, layers=args.layer, bits=args.bits)
    c.convert()


if __name__ == "__main__":
    main()
