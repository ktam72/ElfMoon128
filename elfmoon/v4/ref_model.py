"""DeepSeek-V4-Flash 公式モデル（inference/model.py）の純 torch リファレンス。

tilelang カーネルを ref_kernels.py の純 torch 実装で置き換え、bf16 デコード済み
重みで公式アルゴリズム（QAT シミュレーション含む）を再現する。パリティ検証用。
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .ref_kernels import (
    act_quant,
    apply_rotary_emb,
    fp4_act_quant,
    get_compress_topk_idxs,
    get_window_topk_idxs,
    hc_split_sinkhorn,
    precompute_freqs_cis,
    rotate_activation,
    rms_norm,
    sparse_attn,
)

block_size = 128
fp4_block_size = 32
scale_fmt = None
scale_dtype = torch.float32


@dataclass
class ModelArgs:
    max_batch_size: int = 4
    max_seq_len: int = 4096
    temperature: float = 1
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Optional[Literal["ue8m0"]] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = "fp4"
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 2048
    n_layers: int = 43
    n_hash_layers: int = 3
    n_mtp_layers: int = 1
    n_heads: int = 64
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: Tuple[int] = field(default_factory=tuple)
    dspark_markov_rank: int = 256


def build_args(cfg: dict) -> ModelArgs:
    """公式 config.json から ModelArgs を構築。"""
    rope = cfg.get("rope_scaling", {})
    a = ModelArgs(
        vocab_size=cfg["vocab_size"],
        dim=cfg["hidden_size"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        n_layers=cfg["num_hidden_layers"],
        n_hash_layers=cfg["num_hash_layers"],
        n_heads=cfg["num_attention_heads"],
        n_routed_experts=cfg["n_routed_experts"],
        n_activated_experts=cfg["num_experts_per_tok"],
        score_func=cfg.get("scoring_func", "sqrtsoftplus"),
        route_scale=cfg.get("routed_scaling_factor", 1.5),
        swiglu_limit=cfg.get("swiglu_limit", 0.0),
        q_lora_rank=cfg["q_lora_rank"],
        head_dim=cfg["head_dim"],
        rope_head_dim=cfg["qk_rope_head_dim"],
        norm_eps=cfg["rms_norm_eps"],
        o_groups=cfg["o_groups"],
        o_lora_rank=cfg["o_lora_rank"],
        window_size=cfg.get("sliding_window", 128),
        compress_ratios=tuple(cfg["compress_ratios"]),
        compress_rope_theta=cfg.get("compress_rope_theta", 160000.0),
        original_seq_len=rope.get("original_max_position_embeddings", 0),
        rope_theta=cfg.get("rope_theta", 10000.0),
        rope_factor=rope.get("factor", 1.0),
        beta_fast=rope.get("beta_fast", 32),
        beta_slow=rope.get("beta_slow", 1),
        index_n_heads=cfg.get("index_n_heads", 64),
        index_head_dim=cfg.get("index_head_dim", 128),
        index_topk=cfg.get("index_topk", 512),
        hc_mult=cfg.get("hc_mult", 4),
        hc_sinkhorn_iters=cfg.get("hc_sinkhorn_iters", 20),
        hc_eps=cfg.get("hc_eps", 1e-6),
        expert_dtype=cfg.get("expert_dtype", "fp4"),
        n_shared_experts=cfg.get("n_shared_experts", 1),
        dspark_block_size=cfg.get("dspark_block_size", 0),
        dspark_noise_token_id=cfg.get("dspark_noise_token_id", 0),
        dspark_target_layer_ids=tuple(cfg.get("dspark_target_layer_ids", [])),
        dspark_markov_rank=cfg.get("dspark_markov_rank", 256),
        max_batch_size=1,
    )
    return a


class Linear(nn.Module):
    """bf16 デコード済み重み。qkind が設定されると入力へ QAT 量子化を適用。

    qkind: None=通常 bf16 / "fp8"=入力 fp8 量子化（公式 fp8_gemm 相当）/
           "fp4"=入力 fp8 量子化（公式 fp4_gemm 相当、重みは fp4 デコード済み）。
    """

    def __init__(self, in_features, out_features, qkind=None, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qkind = qkind
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        w = self.weight
        if self.qkind is not None:
            x = act_quant(x, block_size, scale_fmt, scale_dtype, True)
        y = F.linear(x.float(), w.float())
        return y.bfloat16()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16))

    def forward(self, x):
        return F.embedding(x, self.weight.float()).bfloat16()


class Compressor(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32)
        )
        self.wkv = Linear(self.dim, coff * head_dim)
        self.wgate = Linear(self.dim, coff * head_dim)
        self.norm = RMSNorm(head_dim, args.norm_eps)
        self.kv_cache = None
        self.register_buffer(
            "kv_state",
            torch.zeros(
                args.max_batch_size,
                coff * compress_ratio,
                coff * head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (args.max_batch_size, coff * compress_ratio, coff * head_dim),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.freqs_cos = self.freqs_sin = None

    def overlap_transform(self, tensor, value=0):
        ratio, d = self.compress_ratio, self.head_dim
        b, s, _, _ = tensor.size()
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x, start_pos):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = (
            self.compress_ratio,
            self.overlap,
            self.head_dim,
            self.rope_head_dim,
        )
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x).float()
        score = self.wgate(x).float()
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[:bsz, :ratio, :d],
                            self.kv_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cos = self.freqs_cos[:cutoff:ratio]
            freqs_sin = self.freqs_sin[:cutoff:ratio]
        else:
            freqs_cos = self.freqs_cos[start_pos + 1 - self.compress_ratio].unsqueeze(0)
            freqs_sin = self.freqs_sin[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        kv = apply_rotary_emb(kv, freqs_cos, freqs_sin)
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant(kv, fp4_block_size, True)
        else:
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, qkind="fp8")
        self.weights_proj = Linear(self.dim, self.n_heads)
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len // compress_ratio,
                self.head_dim,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )
        self.freqs_cos = self.freqs_sin = None

    def forward(self, x, qr, start_pos, offset):
        bsz, seqlen, _ = x.size()
        freqs_cos = self.freqs_cos[start_pos : start_pos + seqlen]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cos = self.freqs_cos
            self.compressor.freqs_sin = self.freqs_sin
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        q = rotate_activation(q)
        fp4_act_quant(q, fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum(
            "bshd,btd->bsht", q.float(), self.kv_cache[:bsz, : end_pos // ratio].float()
        )
        index_score = (index_score.relu_() * weights.float().unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = (
                torch.arange(seqlen // ratio).repeat(seqlen, 1)
                >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            )
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    def __init__(self, layer_id, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.wq_a = Linear(self.dim, self.q_lora_rank, qkind="fp8")
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim, qkind="fp8")
        self.wkv = Linear(self.dim, self.head_dim, qkind="fp8")
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            qkind="fp8",
        )
        self.wo_b = Linear(self.n_groups * args.o_lora_rank, self.dim, qkind="fp8")
        self.softmax_scale = self.head_dim**-0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        kv_cache_size = args.window_size + (
            args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_size, kv_cache_size, self.head_dim, dtype=torch.bfloat16
            ),
            persistent=False,
        )
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        cos, sin = precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, x, start_pos):
        bsz, seqlen, _ = x.size()
        freqs_cos = self.freqs_cos[start_pos : start_pos + seqlen]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cos = self.freqs_cos
            self.compressor.freqs_sin = self.freqs_sin
            if self.indexer is not None:
                self.indexer.freqs_cos = self.freqs_cos
                self.indexer.freqs_sin = self.freqs_sin
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps)
        q = q.bfloat16()
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv = apply_rotary_emb(kv, freqs_cos, freqs_sin)
        act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset).int()
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(
                q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale
            )
        o = apply_rotary_emb(o, freqs_cos, freqs_sin, inverse=True)

        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float()).bfloat16()
        x = self.wo_b(o.flatten(2))
        self.debug = dict(q=q, kv=kv, topk_idxs=topk_idxs, o_woa=o, out=x)
        return x


class Gate(nn.Module):
    def __init__(self, layer_id, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(
            torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16)
        )
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int64
                ),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(
                torch.empty(args.n_routed_experts, dtype=torch.float32)
            )
            self.register_parameter("tid2eid", None)

    def forward(self, x, input_ids=None):
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    def __init__(self, dim, inter_dim, qkind=None, swiglu_limit=0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, qkind=qkind)
        self.w2 = Linear(inter_dim, dim, qkind=qkind)
        self.w3 = Linear(dim, inter_dim, qkind=qkind)
        self.swiglu_limit = swiglu_limit

    def forward(self, x, weights=None):
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    def __init__(self, layer_id, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = 0
        self.experts_end_idx = args.n_routed_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = "fp4" if args.expert_dtype == "fp4" else None
        self.experts = nn.ModuleList(
            [
                Expert(
                    args.dim,
                    args.moe_inter_dim,
                    qkind=expert_dtype,
                    swiglu_limit=args.swiglu_limit,
                )
                for _ in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = Expert(
            args.dim, args.moe_inter_dim, qkind="fp8", swiglu_limit=args.swiglu_limit
        )

    def forward(self, x, input_ids):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        self.debug = dict(weights=weights, indices=indices)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        y += self.shared_experts(x).float()
        return y.type_as(x).view(shape)


class Block(nn.Module):
    def __init__(self, layer_id, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def hc_head(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def forward(self, x, start_pos, input_ids=None, *attn_args):
        self.debug = {}
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        self.debug["pre_attn"] = x
        x = self.attn_norm(x)
        self.debug["attn_norm"] = x
        x = self.attn(x, start_pos, *attn_args)
        self.debug["attn_out"] = x
        self.debug["attn"] = self.attn.debug
        x = self.hc_post(x, residual, post, comb)
        self.debug["hc_post1"] = x
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        self.debug["pre_ffn"] = x
        x = self.ffn_norm(x)
        self.debug["ffn_norm"] = x
        x = self.ffn(x, input_ids)
        self.debug["ffn_out"] = x
        self.debug["ffn"] = self.ffn.debug
        x = self.hc_post(x, residual, post, comb)
        self.debug["hc_post2"] = x
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, layer_ids=None):
        super().__init__()
        set_qat_globals(args)
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layer_ids = list(range(args.n_layers) if layer_ids is None else layer_ids)
        self.layers = nn.ModuleList(
            [Block(layer_id, args) for layer_id in self.layer_ids]
        )
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    @torch.inference_mode()
    def forward(self, input_ids, start_pos=0, return_hidden=False):
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, input_ids)
        last = self.layers[-1]
        h = last.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        hidden = h
        logits = self.head(self.norm(hidden))
        if return_hidden:
            return logits, hidden
        return logits

    def forward_block(self, x, input_ids, start_pos=0):
        """Block 境界を検証するためのフック。入力は [b,s,hc,d] の h。"""
        h = x
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        return h


class ParallelHead(nn.Module):
    def __init__(self, vocab_size, dim, norm_eps=1e-6, hc_eps=1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def forward(self, x, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        return F.linear(x.float(), self.weight.float())


def load_layer_state(model: nn.Module, state: dict, keys_map: dict):
    """デコード済み state dict（チェックポイントキー名）を model へ割当てる。

    keys_map: {"<model_path>": "<checkpoint_key>", ...}。buffer 等は無視。
    """
    with torch.no_grad():
        for model_path, ckpt_key in keys_map.items():
            t = state[ckpt_key]
            parts = model_path.split(".")
            obj = model
            for p in parts[:-1]:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            try:
                obj._parameters[parts[-1]].copy_(t)
            except RuntimeError as e:
                print(
                    f"load fail: {model_path} <- {ckpt_key} {tuple(t.shape)} -> {tuple(obj._parameters[parts[-1]].shape)}: {e}"
                )
                raise


def set_qat_globals(args: ModelArgs):
    """公式 Transformer.__init__ と同じ QAT スケール設定。"""
    global scale_fmt, scale_dtype
    scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
