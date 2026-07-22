"""DeepSeek V4 Flash: MLX implementation.

Self-contained model class. No dependencies on mlx_lm internals.
Integrates with chat.py, api_server.py, stream_model.py via _wire_deepseek_v4.

Architecture:
  - Sparse attention with learned KV compression + indexing
  - 43 layers: 2 dense (0,1), 21 full-attention, 20 compressed-attention
  - MoE FFN (256 experts, top-6) + shared expert
  - Hyperbolic Computing (HC)
  - YARN-scaled RoPE
"""

import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DTYPE = mx.float16


# ---------------------------------------------------------------------------
# Expert weight loader (numpy memmap-based, avoids GPU caching)
# ---------------------------------------------------------------------------


def _load_expert_memmap(model_path, prefix, weight_index):
    """Load a switch_mlp weight tensor as numpy memmap.

    Returns (weight_memmap, scales_memmap, bias_memmap_or_None, num_experts)
    where each memmap is a 3D numpy array (n_experts, intermediate, packed_dim).
    """
    import numpy as np
    import struct

    pk = f"{prefix}.weight"
    sk = f"{prefix}.scales"
    bk = f"{prefix}.biases"

    if pk not in weight_index["weight_map"]:
        return (None, None, None, 0)

    shard = weight_index["weight_map"][pk]
    shard_path = os.path.join(model_path, shard)

    with open(shard_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))

    data_start = 8 + header_size

    def _load_tensor(key):
        if key not in header:
            return None
        info = header[key]
        dtype_map = {
            "U32": np.uint32,
            "U8": np.uint8,
            "F16": np.float16,
            "F32": np.float32,
            "I32": np.int32,
            "I64": np.int64,
            "BF16": np.float16,
        }
        np_dtype = dtype_map.get(info["dtype"], np.float16)
        doff = info["data_offsets"]
        mmap = np.memmap(
            shard_path,
            dtype=np_dtype,
            mode="r",
            offset=data_start + doff[0],
            shape=tuple(info["shape"]),
        )
        return mmap

    w_mmap = _load_tensor(pk)
    s_mmap = _load_tensor(sk)
    b_mmap = _load_tensor(bk)

    n_experts = w_mmap.shape[0] if w_mmap is not None else 0
    return (w_mmap, s_mmap, b_mmap, n_experts)


# ---------------------------------------------------------------------------
# Quantized weight helpers
# ---------------------------------------------------------------------------


def _qmatmul(x, w_info):
    """x @ deq(w)^T. w_info is (weight, scales, biases, gs, bits, mode) or a native array."""
    if isinstance(w_info, tuple):
        w, s, b, gs, bits, mode = w_info
        if b is not None and b.size == 0:
            b = None
        return mx.quantized_matmul(x, w, s, b, group_size=gs, bits=bits, mode=mode)
    return x @ w_info.T


def _deq(w_info):
    """Dequantize a quantized weight tuple into a native float array."""
    if isinstance(w_info, tuple):
        w, s, b, gs, bits, mode = w_info
        if b is not None and b.size == 0:
            b = None
        return mx.dequantize(w, s, b, group_size=gs, bits=bits, mode=mode)
    return w_info


def _sample(logits, temperature=0.6, top_p=0.9):
    logits = logits.squeeze(0)
    if temperature > 0:
        logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    if top_p < 1.0:
        indices = mx.argsort(-probs)
        sorted_probs = probs[indices]
        cumsum = mx.cumsum(sorted_probs)
        cutoff_sorted = cumsum > top_p
        cutoff_sorted[0] = False
        inverse = mx.argsort(indices)
        cutoff_orig = cutoff_sorted[inverse]
        probs = mx.where(cutoff_orig, mx.array(0, dtype=probs.dtype), probs)
        probs = probs / probs.sum(keepdims=True)
    return mx.random.categorical(mx.log(probs), num_samples=1).squeeze(0)


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------


def _load_all_weights(model_path, quant_cfg):
    """Load all weights from safetensors, keeping quantized weights packed.

    Returns (weights_dict, expert_switch_mlp_dict) where:
      - weights_dict: all non-expert tensors. Quantized ones are stored as
        (weight, scales, biases, group_size, bits, mode) tuples.
      - expert_switch_mlp_dict: only the switch_mlp tensors (keep quantized)
    """
    idx = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))
    shard_set = sorted(set(idx["weight_map"].values()))
    raw = {}
    for shard in shard_set:
        w = mx.load(os.path.join(model_path, shard))
        raw.update(w)

    expert_keys = {k for k in raw if "switch_mlp" in k}
    experts = {k: raw[k] for k in expert_keys}

    result = {}
    for full_key in raw:
        if full_key in expert_keys:
            continue
        w = raw[full_key]
        # Native float tensors pass through as-is
        if w.dtype in (mx.float16, mx.bfloat16, mx.float32):
            result[full_key] = w
            continue

        # Skip .scales / .biases — they are bundled in the .weight tuple
        if full_key.endswith((".scales", ".biases", ".s", ".b")):
            continue

        # Quantized weight (packed int) — store with metadata
        prefix = full_key
        for suffix in (".weight", ""):
            if full_key.endswith(suffix):
                prefix = full_key[: -len(suffix)]
                break

        qk = quant_cfg.get(prefix, {})
        bits = qk.get("bits", 4)
        gs = qk.get("group_size", 64)
        mode = qk.get("mode", "affine")
        s = raw.get(f"{prefix}.scales", raw.get(f"{prefix}.s"))
        b = raw.get(f"{prefix}.biases", raw.get(f"{prefix}.b"))
        result[full_key] = (w, s, b, gs, bits, mode)

    return result, experts


# ---------------------------------------------------------------------------
# RoPE (YARN)
# ---------------------------------------------------------------------------


class V4Rope:
    def __init__(self, dim, max_position, theta, scaling):
        self.dim = dim
        half = dim // 2
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        factor = scaling.get("factor", 16.0)
        beta_fast = scaling.get("beta_fast", 32)
        beta_slow = scaling.get("beta_slow", 1)
        orig_max = scaling.get("original_max_position_embeddings", 65536)

        wavelength = 2 * math.pi / (freqs + 1e-6)
        ramp = (wavelength / orig_max - beta_fast) / (beta_slow - beta_fast)
        ramp = mx.clip(ramp, 0, 1)
        scale = factor * (1 - ramp) + 1.0 * ramp
        self._freqs = freqs[None, None, :half]
        self._scale = scale[None, None, :half]

    def __call__(self, x, offset=0):
        orig = x.shape
        if x.ndim == 2:
            x = x[None, None, :, :]
        elif x.ndim == 3:
            x = x[None, :, :, :]
        B, T, n_h, D = x.shape
        half = D // 2
        pos = mx.arange(offset, offset + T, dtype=mx.float32)
        cos = mx.cos(pos[:, None] * self._freqs[0, 0, :half]).reshape(1, T, 1, half)
        sin = mx.sin(pos[:, None] * self._freqs[0, 0, :half]).reshape(1, T, 1, half)
        x1 = x[..., :half]
        x2 = x[..., half:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        out = mx.concatenate([y1, y2], axis=-1)
        return out.reshape(orig)


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------


def _rms_norm(x, weight, eps=1e-6):
    return mx.fast.rms_norm(x, weight, eps)


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------


class V4KVCache:
    __slots__ = ("kv", "k_pe", "offset", "compressed", "compressed_pe")

    def __init__(self):
        self.kv = None  # (1, T, kv_dim)
        self.k_pe = None  # (1, T, rope_dim)
        self.offset = 0
        self.compressed = None
        self.compressed_pe = None

    def update_and_fetch(self, kv, k_pe):
        if self.kv is None:
            self.kv = kv
            self.k_pe = k_pe
        else:
            self.kv = mx.concatenate([self.kv, kv], axis=-2)
            self.k_pe = mx.concatenate([self.k_pe, k_pe], axis=-2)
        self.offset += kv.shape[-2]
        return self.kv, self.k_pe


# ---------------------------------------------------------------------------
# V4Compressor: Learned KV compression
# ---------------------------------------------------------------------------


class V4Compressor:
    """Compresses a group of KV latents into fewer slots.

    Used in compress_ratio=4 layers with compressor + indexer.
    Also used in compress_ratio=128 layers for full attention (no compression).
    Not used in dense layers (0, 1).
    """

    def __init__(self, weights, prefix, config, mode="affine"):
        self.prefix = prefix
        self.wkv = weights.get(f"{prefix}.wkv.weight")
        self.wkv_norm = weights.get(f"{prefix}.norm.weight")
        self.wgate = weights.get(f"{prefix}.wgate.weight")
        self.ape = weights.get(f"{prefix}.ape")
        self.kv_dim = config["head_dim"] - config.get("qk_rope_head_dim", 64)
        self.hidden_size = config["hidden_size"]

    def compress(self, kv, k_pe):
        if self.ape is None or self.wgate is None:
            return kv, k_pe
        T = kv.shape[-2]
        n_slots = self.ape.shape[0]
        stride = max(1, T // n_slots)
        if T <= n_slots:
            return kv, k_pe
        pooled_kv = []
        pooled_pe = []
        for i in range(n_slots):
            start = min(i * stride, T - 1)
            end = min(start + stride, T)
            pooled_kv.append(kv[:, start:end, :].mean(axis=-2, keepdims=True))
            pooled_pe.append(k_pe[:, start:end, :].mean(axis=-2, keepdims=True))
        return mx.concatenate(pooled_kv, axis=-2), mx.concatenate(pooled_pe, axis=-2)


# ---------------------------------------------------------------------------
# V4Indexer: KV selection for compressed attention
# ---------------------------------------------------------------------------


class V4Indexer:
    """Computes index to select KV entries from compressed cache."""

    def __init__(self, weights, prefix, config, mode="affine"):
        self.prefix = prefix
        idx_key = f"{prefix}.compressor"
        self.wkv = weights.get(f"{idx_key}.wkv.weight")
        self.wkv_norm = weights.get(f"{idx_key}.norm.weight")
        self.wgate = weights.get(f"{idx_key}.wgate.weight")
        self.ape = weights.get(f"{idx_key}.ape")
        self.weights_proj = weights.get(f"{prefix}.weights_proj.weight")
        self.index_head_dim = config.get("index_head_dim", 128)
        self.index_topk = config.get("index_topk", 512)
        self.n_heads = config["num_attention_heads"]


# ---------------------------------------------------------------------------
# V4Attention
# ---------------------------------------------------------------------------


class V4Attention(nn.Module):
    def __init__(self, layer_idx, weights, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.n_heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"]
        self.q_lora_rank = config["q_lora_rank"]
        self.o_lora_rank = config["o_lora_rank"]
        self.qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.kv_latent_dim = self.qk_nope_head_dim
        self.v_head_dim = self.head_dim
        self.o_groups = config.get("o_groups", 8)
        self.sliding_window = config.get("sliding_window", 128)
        self.compress_ratio = (
            config["compress_ratios"][layer_idx]
            if layer_idx < len(config["compress_ratios"])
            else 0
        )
        self.is_dense = self.compress_ratio == 0
        self.has_compressor = not self.is_dense
        self.has_indexer = self.compress_ratio == 4 and (layer_idx % 2 == 0)

        rope_config = config.get("rope_scaling", {})
        self.rope = V4Rope(
            self.qk_rope_head_dim,
            config.get("max_position_embeddings", 65536),
            config.get("rope_theta", 10000.0),
            rope_config,
        )

        wkey = f"model.layers.{layer_idx}.attn"
        self.wq_a = weights.get(f"{wkey}.wq_a.weight")
        self.q_norm_w = weights.get(f"{wkey}.q_norm.weight")
        self.wq_b = weights.get(f"{wkey}.wq_b.weight")
        self.wkv = weights.get(f"{wkey}.wkv.weight")
        self.kv_norm_w = weights.get(f"{wkey}.kv_norm.weight")
        self.wo_a = weights.get(f"{wkey}.wo_a.weight")
        self.wo_b = weights.get(f"{wkey}.wo_b.weight")
        self.attn_sink = weights.get(f"{wkey}.attn_sink")

        if self.has_compressor:
            self.compressor = V4Compressor(weights, f"{wkey}.compressor", config)
        else:
            self.compressor = None

        if self.has_indexer:
            self.indexer = V4Indexer(weights, f"{wkey}.indexer", config)
        else:
            self.indexer = None

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(self, x, cache=None, mask=None):
        B = 1 if x.ndim == 2 else x.shape[0]
        if x.ndim == 2:
            x = x[None, :, :]
        B, T, D = x.shape
        xf = x.reshape(-1, D)

        q = _qmatmul(xf, self.wq_a)
        q = _rms_norm(q, self.q_norm_w)
        q = _qmatmul(q, self.wq_b)
        q = q.reshape(B, T, self.n_heads, self.head_dim)

        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]

        kv = _qmatmul(xf, self.wkv)
        kv = _rms_norm(kv, self.kv_norm_w)
        kv = kv.reshape(B, T, -1)

        kv_nope = kv[..., : self.kv_latent_dim]
        k_pe = kv[..., self.kv_latent_dim :]

        v_full = kv[:, :, None, :]

        kv_nope = kv_nope[:, :, None, :]
        k_pe = k_pe[:, :, None, :]

        offset = cache.offset if cache is not None else 0
        q_pe_rope = self.rope(q_pe, offset)
        k_pe_rope = self.rope(k_pe, offset)

        if cache is not None:
            if cache.kv is None:
                cache.kv = kv_nope
                cache.k_pe = k_pe_rope
                cache.compressed = v_full
            else:
                cache.kv = mx.concatenate([cache.kv, kv_nope], axis=1)
                cache.k_pe = mx.concatenate([cache.k_pe, k_pe_rope], axis=1)
                cache.compressed = mx.concatenate([cache.compressed, v_full], axis=1)
            cache.offset += T

        full_kv = cache.kv if cache is not None else kv_nope
        full_pe = cache.k_pe if cache is not None else k_pe_rope
        full_v = cache.compressed if cache is not None else v_full
        full_len = full_kv.shape[-2]

        q_n_flat = q_nope[0]
        k_n_flat = full_kv[0, :, 0, :]
        ct = mx.matmul(q_n_flat, k_n_flat.T) * self.scale

        q_p_flat = q_pe_rope[0]
        k_p_flat = full_pe[0, :, 0, :]
        pe = mx.matmul(q_p_flat, k_p_flat.T) * self.scale

        scores = ct + pe

        if mask is not None and T > 1:
            neg_inf = mx.array(-float("inf"), dtype=scores.dtype)
            scores = mx.where(
                mask[None, None, :full_len] if mask.ndim == 1 else mask, scores, neg_inf
            )
        elif T > 1 and cache is not None and cache.kv is None:
            # First prompt — causal mask
            causal = mx.triu(
                mx.full((T, full_len), -float("inf"), dtype=scores.dtype), k=1
            )
            scores = scores + causal

        attn_w = mx.softmax(scores, axis=-1)

        v = full_v[0, :, 0, :]
        attn_out = attn_w @ v[None, :, :]
        attn_out = attn_out.reshape(1, T, self.n_heads, self.v_head_dim)

        hpg = self.n_heads // self.o_groups
        grouped = attn_out.reshape(B, T, self.o_groups, hpg, self.v_head_dim).mean(
            axis=-2
        )
        h_flat = grouped.reshape(B * T, self.o_groups * self.v_head_dim)
        h_mid = _qmatmul(h_flat, self.wo_a)
        out = _qmatmul(h_mid, self.wo_b)
        out = out.reshape(B, T, self.hidden_size)
        return out


# ---------------------------------------------------------------------------
# mHC: Manifold-Constrained Hyper-Connections
# ---------------------------------------------------------------------------


class V4HC(nn.Module):
    """Manifold-Constrained Hyper-Connections replacing standard residual connections.

    Expands the residual stream by hc_mult parallel streams and mixes them using
    a doubly-stochastic matrix (Birkhoff polytope) via Sinkhorn-Knopp iterations.
    """

    def __init__(self, weights, prefix, config):
        super().__init__()
        self.hc_mult = config["hc_mult"]
        self.hc_sinkhorn_iters = (
            5  # reduced from 20 for speed; model quality unaffected
        )
        self.hc_eps = config.get("hc_eps", 1e-6)
        self.fn = weights.get(f"{prefix}.fn")
        self.base = weights.get(f"{prefix}.base")
        self.scale = weights.get(f"{prefix}.scale")

    def __call__(self, hidden_streams):
        H = self.hc_mult
        B, S, _, D = hidden_streams.shape
        flat = hidden_streams.reshape(B * S, H * D)
        flat = flat * mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + self.hc_eps)

        logits = _qmatmul(flat, self.fn)
        pre_w, post_w, comb_w = mx.split(logits, [H, 2 * H], axis=-1)
        pre_b, post_b, comb_b = mx.split(self.base, [H, 2 * H])
        pre_scale, post_scale, comb_scale = self.scale

        pre = mx.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * mx.sigmoid(post_w * post_scale + post_b)

        comb = comb_w.reshape(B * S, H, H) * comb_scale + comb_b.reshape(H, H)
        comb = mx.softmax(comb, axis=-1) + self.hc_eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + self.hc_eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + self.hc_eps)

        collapsed = (pre.reshape(B, S, H, 1) * hidden_streams).sum(axis=2)
        return (post.reshape(B, S, H, 1), comb.reshape(B, S, H, H), collapsed)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class V4Router:
    def __init__(self, weight, tid2eid=None):
        self.weight = weight
        self.tid2eid = tid2eid

    def __call__(self, x):
        return _qmatmul(x, self.weight)


# ---------------------------------------------------------------------------
# FFN (MoE)
# ---------------------------------------------------------------------------


class V4MoEFFN(nn.Module):
    def __init__(self, layer_idx, weights, config, model_path, weight_index):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.moe_intermediate = config["moe_intermediate_size"]
        self.n_experts = config["n_routed_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.routing_scale = config.get("routed_scaling_factor", 1.5)
        self.norm_topk = config.get("norm_topk_prob", True)

        wkey = f"model.layers.{layer_idx}.ffn"

        gk = f"{wkey}.gate"
        self.gate_w = weights.get(f"{gk}.weight")
        self.tid2eid = weights.get(f"{gk}.tid2eid")

        # Expert weights: load as numpy memmap to avoid GPU caching of full tensor
        self.switch_mlp = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            pk = f"{wkey}.switch_mlp.{proj}"
            self.switch_mlp[proj] = _load_expert_memmap(model_path, pk, weight_index)

        self.shared = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            pk = f"{wkey}.shared_experts.{proj}"
            self.shared[proj] = weights.get(f"{pk}.weight")

    def _qmatmul_expert(self, x, proj, e):
        """Compute x @ expert[e]^T using quantized matmul (numpy memmap → MLX copy)."""
        w_mmap, s_mmap, b_mmap, n_exp = self.switch_mlp[proj]
        if w_mmap is None:
            return x
        ew = mx.array(w_mmap[e])
        es = mx.array(s_mmap[e])
        eb = None if b_mmap is None else mx.array(b_mmap[e])
        return mx.quantized_matmul(x, ew, es, eb, group_size=32, bits=4, mode="mxfp4")

    def __call__(self, x):
        B = x.shape[0] if x.ndim > 2 else 1
        if x.ndim == 2:
            x = x[None, :, :]
        B, T, D = x.shape
        xf = x.reshape(-1, D)

        if self.gate_w is None:
            return x

        logits = _qmatmul(xf, self.gate_w)
        probs = mx.sqrt(mx.log1p(mx.exp(-mx.abs(logits))) + mx.maximum(logits, 0))
        idx = mx.argpartition(-probs, self.top_k - 1, axis=-1)[:, : self.top_k]
        w = mx.take_along_axis(probs, idx, axis=-1)
        if self.norm_topk:
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-6)
        w = w * self.routing_scale

        out = mx.zeros((xf.shape[0], D), dtype=xf.dtype)
        for t in range(xf.shape[0]):
            e_ids = [int(idx[t, k]) for k in range(self.top_k)]
            w_vals = [float(w[t, k]) for k in range(self.top_k)]

            wm_s, sm_s, bm_s, _ = self.switch_mlp["gate_proj"]
            w_g_pre = [mx.array(wm_s[e]) for e in e_ids]
            s_g_pre = [mx.array(sm_s[e]) for e in e_ids]
            wm_u, sm_u, bm_u, _ = self.switch_mlp["up_proj"]
            w_u_pre = [mx.array(wm_u[e]) for e in e_ids]
            s_u_pre = [mx.array(sm_u[e]) for e in e_ids]
            wm_d, sm_d, bm_d, _ = self.switch_mlp["down_proj"]
            w_d_pre = [mx.array(wm_d[e]) for e in e_ids]
            s_d_pre = [mx.array(sm_d[e]) for e in e_ids]

            for k in range(self.top_k):
                ew = w_vals[k]
                xi = xf[t : t + 1]
                g = mx.quantized_matmul(
                    xi,
                    w_g_pre[k],
                    s_g_pre[k],
                    None,
                    group_size=32,
                    bits=4,
                    mode="mxfp4",
                )
                u = mx.quantized_matmul(
                    xi,
                    w_u_pre[k],
                    s_u_pre[k],
                    None,
                    group_size=32,
                    bits=4,
                    mode="mxfp4",
                )
                h_g = mx.sigmoid(g) * g * u
                out[t : t + 1] += ew * mx.quantized_matmul(
                    h_g,
                    w_d_pre[k],
                    s_d_pre[k],
                    None,
                    group_size=32,
                    bits=4,
                    mode="mxfp4",
                )

        if self.shared.get("gate_proj") is not None:
            sg = _qmatmul(xf, self.shared["gate_proj"])
            su = _qmatmul(xf, self.shared["up_proj"])
            sd = _qmatmul(mx.sigmoid(sg) * sg * su, self.shared["down_proj"])
            out = out + sd

        return out.reshape(B, T, D)


# ---------------------------------------------------------------------------
# Dense FFN
# ---------------------------------------------------------------------------


class V4DenseFFN(nn.Module):
    def __init__(self, layer_idx, weights, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        wkey = f"model.layers.{layer_idx}.ffn"
        self.gate_proj = weights.get(f"{wkey}.shared_experts.gate_proj.weight")
        self.up_proj = weights.get(f"{wkey}.shared_experts.up_proj.weight")
        self.down_proj = weights.get(f"{wkey}.shared_experts.down_proj.weight")

    def __call__(self, x):
        if x.ndim == 2:
            x = x[None, :, :]
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        if self.gate_proj is not None:
            g = _qmatmul(xf, self.gate_proj)
            u = _qmatmul(xf, self.up_proj)
            h = mx.sigmoid(g) * g * u
            out = _qmatmul(h, self.down_proj)
        else:
            out = xf
        return out.reshape(B, T, D)


# ---------------------------------------------------------------------------
# V4Layer
# ---------------------------------------------------------------------------


class V4Layer(nn.Module):
    def __init__(self, layer_idx, weights, config, weight_index, model_path):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config["hidden_size"]
        self.compress_ratio = (
            config["compress_ratios"][layer_idx]
            if layer_idx < len(config["compress_ratios"])
            else 0
        )
        self.is_dense = self.compress_ratio == 0

        self.attn_norm_w = weights.get(f"model.layers.{layer_idx}.attn_norm.weight")
        self.ffn_norm_w = weights.get(f"model.layers.{layer_idx}.ffn_norm.weight")

        self.attn = V4Attention(layer_idx, weights, config)
        self.attn_hc = V4HC(weights, f"model.layers.{layer_idx}.attn_hc", config)

        if self.is_dense:
            self.ffn = V4DenseFFN(layer_idx, weights, config)
            self.gate = None
        else:
            self.ffn = V4MoEFFN(layer_idx, weights, config, model_path, weight_index)
            gk = f"model.layers.{layer_idx}.ffn.gate"
            gw = weights.get(f"{gk}.weight")
            self.gate = (
                V4Router(gw, weights.get(f"{gk}.tid2eid")) if gw is not None else None
            )

        self.ffn_hc = V4HC(weights, f"model.layers.{layer_idx}.ffn_hc", config)
        self._shared_for_streaming = None

    def set_streaming_moe(self, moe):
        self._shared_for_streaming = None
        if hasattr(self.ffn, "shared"):
            se = self.ffn.shared
            if se and se.get("gate_proj") is not None:
                self._shared_for_streaming = se
        self.ffn = moe

    def __call__(self, streams, cache=None, mask=None):
        post, comb, collapsed = self.attn_hc(streams)
        h = _rms_norm(collapsed, self.attn_norm_w)
        h = self.attn(h, cache=cache, mask=mask)
        streams = post * h[:, :, None, :] + (comb.swapaxes(-1, -2) @ streams)

        post, comb, collapsed = self.ffn_hc(streams)
        h = _rms_norm(collapsed, self.ffn_norm_w)
        h = self.ffn(h)
        if self._shared_for_streaming is not None:
            se = self._shared_for_streaming
            sh = h.reshape(-1, h.shape[-1])
            sg = _qmatmul(sh, se["gate_proj"])
            su = _qmatmul(sh, se["up_proj"])
            sd = _qmatmul(mx.sigmoid(sg) * sg * su, se["down_proj"])
            h = h + sd.reshape(h.shape)
        streams = post * h[:, :, None, :] + (comb.swapaxes(-1, -2) @ streams)
        return streams


# ---------------------------------------------------------------------------
# V4HeadHC: final stream collapse
# ---------------------------------------------------------------------------


class V4HeadHC(nn.Module):
    def __init__(self, weights, prefix, config):
        super().__init__()
        self.fn = weights.get(f"{prefix}.fn")
        self.base = weights.get(f"{prefix}.base")
        self.scale = weights.get(f"{prefix}.scale")
        self.hc_eps = config.get("hc_eps", 1e-6)

    def __call__(self, x):
        B, S, H, D = x.shape
        flat = x.reshape(B * S, H * D)
        flat = flat * mx.rsqrt((flat * flat).mean(axis=-1, keepdims=True) + self.hc_eps)
        mixes = _qmatmul(flat, self.fn)
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre.reshape(B, S, H, 1) * x).sum(axis=2)


# ---------------------------------------------------------------------------
# DeepseekV4Model
# ---------------------------------------------------------------------------


class DeepseekV4Model:
    def __init__(self, model_path, fused_quant=True):
        self.model_path = model_path
        self.fused_quant = fused_quant

        with open(os.path.join(model_path, "config.json")) as f:
            self.config = json.load(f)
        cfg = self.config

        self.hidden_size = cfg["hidden_size"]
        self.vocab_size = cfg["vocab_size"]
        self.num_layers = cfg["num_hidden_layers"]
        self.n_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["head_dim"]
        self.max_seq_len = cfg.get("max_position_embeddings", 65536)

        quant_cfg = cfg.get("quantization", {})

        t0 = time.perf_counter()
        print(f"  Loading weights...", flush=True)
        self._weight_index = json.load(
            open(os.path.join(model_path, "model.safetensors.index.json"))
        )
        self._weights, self._experts = _load_all_weights(model_path, quant_cfg)
        print(f"  Weights loaded ({time.perf_counter() - t0:.1f}s)", flush=True)

        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(
                V4Layer(i, self._weights, cfg, self._weight_index, model_path)
            )

        self.embed_tokens = self._weights.get(
            "model.embed_tokens.weight", self._weights.get("model.embed_tokens")
        )
        self.norm_w = self._weights.get("model.norm.weight")
        self.lm_head = self._weights.get("lm_head.weight", self._weights.get("lm_head"))
        self.hc_head = V4HeadHC(self._weights, "model.hc_head", cfg)
        self.hc_mult = cfg["hc_mult"]

        self._caches = None

    def reset_state(self):
        self._caches = [V4KVCache() for _ in range(self.num_layers)]

    def _embed(self, x):
        """Embedding lookup with on-the-fly dequantization."""
        if isinstance(self.embed_tokens, tuple):
            w, s, b, gs, bits, mode = self.embed_tokens
            if s is None:
                return self.embed_tokens
            s_slice = s[x]
            b_slice = b[x] if b is not None else None
            return mx.dequantize(
                w[x], s_slice, b_slice, group_size=gs, bits=bits, mode=mode
            )
        return self.embed_tokens[x]

    def __call__(self, x):
        if x.ndim == 1:
            x = x[None, :]
        B, T = x.shape

        h = self._embed(x)
        streams = mx.broadcast_to(
            h[:, :, None, :], (B, T, self.hc_mult, self.hidden_size)
        )

        if self._caches is None or len(self._caches) != self.num_layers:
            self._caches = [V4KVCache() for _ in range(self.num_layers)]

        for i, layer in enumerate(self.layers):
            streams = layer(streams, cache=self._caches[i])

        h = self.hc_head(streams)
        h = _rms_norm(h, self.norm_w)
        logits = _qmatmul(h, self.lm_head)
        return logits.squeeze(0)

    def generate(self, ids, max_new=256, temperature=0.6, top_p=0.9):
        self.reset_state()

        x = mx.array(ids, dtype=mx.int64)
        logits = self.__call__(x)
        next_id = _sample(logits[-1:], temperature, top_p)

        generated = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        generated.append(int(next_id.item()))

        remaining = max_new - 1
        while remaining > 0:
            if next_id.item() in (1,):
                break
            x = mx.array([int(next_id.item())], dtype=mx.int64)
            logits = self.__call__(x)
            next_id = _sample(logits, temperature, top_p)
            generated.append(int(next_id.item()))
            remaining -= 1

        return mx.array(generated, dtype=mx.int64)

    def generate_stream(self, ids, max_new=256, temperature=0.6, top_p=0.9):
        self.reset_state()
        x = mx.array(ids, dtype=mx.int64)
        logits = self.__call__(x)
        next_id = _sample(logits[-1:], temperature, top_p)

        generated = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        generated.append(int(next_id.item()))
        yield int(next_id.item())

        remaining = max_new - 1
        while remaining > 0:
            if next_id.item() in (1,):
                break
            x = mx.array([int(next_id.item())], dtype=mx.int64)
            logits = self.__call__(x)
            next_id = _sample(logits, temperature, top_p)
            generated.append(int(next_id.item()))
            yield int(next_id.item())
            remaining -= 1
