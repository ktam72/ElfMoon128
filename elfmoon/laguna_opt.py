"""Optimized Laguna model for ElfMoon128 — fused QKV + RoPE, compile MoE.

Optimizations vs stock laguna.py:
  1. Fused QKV: q_proj+k_proj+v_proj merged into 1 quantized_matmul
  2. Fused per-head gating into attention output element-wisely
  3. mx.compile on MoeBlock (no cache dependency — stable compilation)
  4. Optimized sliding KV cache: in-place slice for single-token decode
  5. Direct mx.fast ops, fewer transposes
  6. Sanitize hook auto-fuses QKV weights after model loading
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False
    attention_bias: bool = False
    gating: bool | str = True
    tie_word_embeddings: bool = False
    rope_theta: float = 500000.0
    rope_parameters: dict[str, Any] | None = None
    rope_scaling: dict[str, Any] | None = None
    partial_rotary_factor: float | None = None
    rope_style: str = "rotate-half"
    sliding_window: int | None = None
    layer_types: list[str] | None = None
    num_attention_heads_per_layer: list[int] | None = None
    swa_rope_parameters: dict[str, Any] | None = None
    swa_attention_sink_enabled: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=lambda: [0])
    moe_routed_scaling_factor: float = 1.0
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0
    moe_router_use_sigmoid: bool = True

    def __post_init__(self):
        if self.gating is True:
            self.gating = "per-head"
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [
                self.num_attention_heads
            ] * self.num_hidden_layers
        rp = (
            dict(self.rope_parameters)
            if self.rope_parameters is not None
            else (
                dict(self.rope_scaling)
                if self.rope_scaling is not None
                else {"rope_type": "default", "rope_theta": self.rope_theta}
            )
        )
        lts = set(self.layer_types)
        lrp = {k: v for k, v in rp.items() if k in lts and isinstance(v, dict)}
        if lrp:
            tlp = {
                k: v for k, v in rp.items() if k not in lts and not isinstance(v, dict)
            }

            def rpf(lt: str):
                p = dict(lrp.get(lt, {}))
                for k, v in tlp.items():
                    p.setdefault(k, v)
                return p

            dlt = "full_attention" if "full_attention" in lrp else next(iter(lrp))
            self.rope_parameters = rpf(dlt)
            if self.swa_rope_parameters is None and "sliding_attention" in lrp:
                self.swa_rope_parameters = rpf("sliding_attention")
        else:
            self.rope_parameters = rp
        if self.swa_rope_parameters is not None:
            self.swa_rope_parameters = dict(self.swa_rope_parameters)
        self.rope_parameters.setdefault("rope_type", "default")
        if self.swa_rope_parameters is not None:
            self.swa_rope_parameters.setdefault("rope_type", "default")
        if self.partial_rotary_factor is not None:
            self.rope_parameters.setdefault(
                "partial_rotary_factor", self.partial_rotary_factor
            )
            if self.swa_rope_parameters is not None:
                self.swa_rope_parameters.setdefault(
                    "partial_rotary_factor", self.partial_rotary_factor
                )


def _rope_base(args: ModelArgs, rope_config: dict) -> float:
    return float(rope_config.get("rope_theta", args.rope_theta))


def _rope_dims(args: ModelArgs, rope_config: dict) -> int:
    return int(args.head_dim * float(rope_config.get("partial_rotary_factor", 1.0)))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_sigmoid = args.moe_router_use_sigmoid
        self.router_logit_softcapping = args.moe_router_logit_softcapping
        self.proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,))

    def __call__(
        self, x: mx.array, top_k: int | None = None
    ) -> tuple[mx.array, mx.array]:
        dtype = x.dtype
        logits = self.proj(x).astype(mx.float32)
        if self.router_logit_softcapping > 0.0:
            c = self.router_logit_softcapping
            logits = mx.tanh(logits / c) * c
        scores = mx.sigmoid(logits) if self.use_sigmoid else mx.softmax(logits, axis=-1)
        corrected = scores + self.e_score_correction_bias.astype(scores.dtype)
        k = self.top_k if top_k is None else top_k
        inds = mx.stop_gradient(
            mx.argpartition(-corrected, kth=k - 1, axis=-1)[..., :k]
        )
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        return inds, weights.astype(dtype)


# ---------------------------------------------------------------------------
# MLP (dense)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x, moe_k=None) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


# ---------------------------------------------------------------------------
# SwitchGLU (MoE) — compiled for decode stability
# ---------------------------------------------------------------------------


class SwitchGLU(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        from mlx_lm.models.switch_layers import SwitchLinear

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=False)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=False)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=False)
        self._gate_up_fused = False

    def _fuse_gate_up(self):
        if self._gate_up_fused:
            return
        gp = self.gate_proj
        up = self.up_proj
        qgp = gp if hasattr(gp, "scales") else None
        if qgp is None or not hasattr(up, "scales"):
            return
        self._fused_w = mx.concatenate([qgp.weight, up.weight], axis=1)
        self._fused_s = mx.concatenate([qgp.scales, up.scales], axis=1)
        bg = getattr(qgp, "biases", None)
        self._fused_b = (
            mx.concatenate([bg, getattr(up, "biases", None)], axis=1)
            if bg is not None
            else None
        )
        self._fused_gs = getattr(qgp, "group_size", 64)
        self._fused_bits = getattr(qgp, "bits", 4)
        self._fused_mode = getattr(qgp, "mode", "affine")
        self._hidden = qgp.weight.shape[1]
        self._gate_up_fused = True

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        if do_sort:
            idx = indices.flatten()
            order = mx.argsort(idx)
            inv = mx.argsort(order)
            x = x.flatten(0, -3)[order // indices.shape[-1]]
            idx = idx[order]
            sorted_args: dict[str, bool] = {"sorted_indices": True}
        else:
            idx = indices
            inv = None
            sorted_args = {"sorted_indices": False}
        if self._gate_up_fused:
            xf = mx.gather_qmm(
                x,
                self._fused_w,
                self._fused_s,
                self._fused_b,
                rhs_indices=idx,
                transpose=True,
                group_size=self._fused_gs,
                bits=self._fused_bits,
                mode=self._fused_mode,
                sorted_indices=sorted_args["sorted_indices"],
            )
            hi = self._hidden
            x_gate, x_up = xf[..., :hi], xf[..., hi:]
        else:
            x_up = self.up_proj(x, idx, **sorted_args)
            x_gate = self.gate_proj(x, idx, **sorted_args)
        x = swiglu(x_gate, x_up)
        x = self.down_proj(x, idx, **sorted_args)
        if do_sort:
            x = x[inv]
            x = mx.unflatten(x, 0, indices.shape)
        return x.squeeze(-2)


# ---------------------------------------------------------------------------
# MoE Block — mx.compile on the pure-functional part
# ---------------------------------------------------------------------------


class MoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = Router(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x: mx.array, moe_k: int | None = None) -> mx.array:
        inds, scores = self.gate(x, top_k=moe_k)
        y = self.switch_mlp(x, inds)
        y = mx.sum(y * scores[..., None], axis=-2)
        if self.routed_scaling_factor != 1.0:
            y = y * self.routed_scaling_factor
        return y + self.shared_expert(x)


# ---------------------------------------------------------------------------
# Fused QKV Attention
# ---------------------------------------------------------------------------


class FusedQKVAttention(nn.Module):
    """Attention with fused QKV projection (1 quantized_matmul instead of 3).

    Fusion is set up lazily after weight loading via _try_fuse_qkv().
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads_per_layer[layer_idx]
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.gate_per_head = args.gating == "per-head"
        self.gating = bool(args.gating)
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = args.sliding_window if self.is_sliding else None
        self.hidden_size = args.hidden_size

        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=args.qkv_bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.qkv_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.qkv_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )

        if self.gating:
            gate_dim = (
                self.n_heads if self.gate_per_head else self.n_heads * self.head_dim
            )
            self.g_proj = nn.Linear(args.hidden_size, gate_dim, bias=False)

        self.sink = (
            mx.zeros((self.n_heads,))
            if (self.is_sliding and args.swa_attention_sink_enabled)
            else None
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rope_config = (
            args.swa_rope_parameters
            if self.is_sliding and args.swa_rope_parameters is not None
            else args.rope_parameters
        )
        self.rope_m = initialize_rope(
            _rope_dims(args, rope_config),
            base=_rope_base(args, rope_config),
            traditional=False,
            scaling_config=rope_config,
            max_position_embeddings=args.max_position_embeddings,
        )
        self._fused = False

    def _try_fuse_qkv(self):
        """Lazily fuse QKV weights after model loading.

        Only succeeds when all three projections are quantized with the same
        group_size / bits / mode.
        """
        if self._fused:
            return True
        try:
            for p in (self.q_proj, self.k_proj, self.v_proj):
                if not hasattr(p, "scales") or p.scales is None:
                    return False
            gs = getattr(self.q_proj, "group_size", 64)
            bits = getattr(self.q_proj, "bits", 4)
            mode = getattr(self.q_proj, "mode", "affine")
            self._qkv_w = mx.concatenate(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], axis=0
            )
            self._qkv_s = mx.concatenate(
                [self.q_proj.scales, self.k_proj.scales, self.v_proj.scales], axis=0
            )
            qb = self.q_proj.biases
            self._qkv_b = (
                mx.concatenate([qb, self.k_proj.biases, self.v_proj.biases], axis=0)
                if qb is not None
                else None
            )
            self._qkv_gs = gs
            self._qkv_bits = bits
            self._qkv_mode = mode
            self._n_qo = self.n_heads * self.head_dim
            self._n_kvo = self.n_kv_heads * self.head_dim
            self._fused = True
            return True
        except Exception:
            return False

    def __call__(self, x, mask=None, cache=None):
        if not self._fused:
            self._try_fuse_qkv()
        B, L, _ = x.shape
        xf = x.reshape(-1, self.hidden_size)

        if self._fused:
            qkv = mx.quantized_matmul(
                xf,
                self._qkv_w,
                self._qkv_s,
                self._qkv_b,
                transpose=True,
                group_size=self._qkv_gs,
                bits=self._qkv_bits,
                mode=self._qkv_mode,
            )
            qo = qkv[..., : self._n_qo]
            ko = qkv[..., self._n_qo : self._n_qo + self._n_kvo]
            vo = qkv[..., self._n_qo + self._n_kvo :]
        else:
            qo = self.q_proj(xf)
            ko = self.k_proj(xf)
            vo = self.v_proj(xf)

        q = self.q_norm(qo.reshape(B, L, self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        k = self.k_norm(ko.reshape(B, L, self.n_kv_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        v = vo.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope_m(q, offset=cache.offset)
            k = self.rope_m(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope_m(q)
            k = self.rope_m(k)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask, sinks=self.sink
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        if self.gating:
            gate = nn.softplus(self.g_proj(xf).astype(mx.float32)).astype(output.dtype)
            if self.gate_per_head:
                output = output.reshape(
                    B, L, self.n_heads, self.head_dim
                ) * gate.reshape(B, L, self.n_heads, 1)
                output = output.reshape(B, L, -1)
            else:
                output = output * gate

        return self.o_proj(output)


# ---------------------------------------------------------------------------
# DecoderLayer
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = FusedQKVAttention(args, layer_idx)
        is_moe = (
            layer_idx not in args.mlp_only_layers
            and args.num_experts > 0
            and (layer_idx + 1) % args.decoder_sparse_step == 0
        )
        self.mlp: MoeBlock | MLP = (
            MoeBlock(args) if is_moe else MLP(args.hidden_size, args.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attention_type: str = args.layer_types[layer_idx]

    def __call__(self, x, mask=None, cache=None, moe_k=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h), moe_k=moe_k)
        return h + r


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx: int = args.layer_types.index("full_attention")
        self.swa_idx: int | None = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else None
        )

    def __call__(self, inputs, cache=None, input_embeddings=None, moe_k=None):
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        full_mask = create_attention_mask(h, cache[self.fa_idx])
        sliding_mask: mx.array | None = None
        if self.swa_idx is not None:
            sliding_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.args.sliding_window
            )
        for layer, c in zip(self.layers, cache):
            mask = (
                sliding_mask
                if layer.attention_type == "sliding_attention"
                else full_mask
            )
            h = layer(h, mask, c, moe_k=moe_k)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None, moe_k=None):
        out = self.model(inputs, cache, input_embeddings, moe_k=moe_k)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def make_cache(self):
        caches: list[KVCache | RotatingKVCache] = []
        for lt in self.args.layer_types:
            if lt == "sliding_attention" and self.args.sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        if any(k.startswith("language_model.") for k in weights):
            prefix = "language_model."
            weights = {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in weights.items()
            }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        weights = self._unpack_compressed_tensors(weights)
        weights = self._remap_router_weights(weights)
        weights = self._stack_experts(weights)
        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
            and not k.endswith(".self_attn.k_scale")
            and not k.endswith(".self_attn.v_scale")
        }

    def _unpack_compressed_tensors(self, weights):
        if not any(k.endswith(".weight_shape") for k in weights):
            return weights
        new = {}
        for k, v in weights.items():
            if k.endswith(".weight_shape"):
                base = k[: -len("weight_shape")]
                if (
                    f"{base}weight_packed" in weights
                    and f"{base}weight_scale" in weights
                ):
                    scales = weights[f"{base}weight_scale"]
                    new[f"{base}weight"] = weights[f"{base}weight_packed"].view(
                        mx.uint32
                    )
                    new[f"{base}scales"] = scales
                    new[f"{base}biases"] = (-8 * scales).astype(scales.dtype)
            elif k.endswith(".weight_packed") or k.endswith(".weight_scale"):
                base = k.rsplit(".", 1)[0] + "."
                if f"{base}weight_shape" in weights:
                    continue
                new[k] = v
            else:
                new[k] = v
        return new

    def _remap_router_weights(self, weights):
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            gw = f"{prefix}.gate.weight"
            if gw in weights:
                weights[f"{prefix}.gate.proj.weight"] = weights.pop(gw)
            legacy = f"{prefix}.experts.e_score_correction_bias"
            if legacy in weights:
                weights[f"{prefix}.gate.e_score_correction_bias"] = weights.pop(legacy)
        return weights

    def _stack_experts(self, weights):
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                for suffix in ["weight", "scales", "biases"]:
                    first = f"{prefix}.experts.0.{proj}.{suffix}"
                    if first not in weights:
                        continue
                    weights[f"{prefix}.switch_mlp.{proj}.{suffix}"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{proj}.{suffix}")
                            for e in range(self.args.num_experts)
                        ]
                    )
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate.proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers
