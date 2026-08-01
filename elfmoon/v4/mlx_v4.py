"""DeepSeek-V4-Flash の忠実な MLX 実装（公式 inference/model.py の移植）。

公式の sparse attention（window + indexer/compress + sink）、HC（Sinkhorn）、
QAT シミュレーション（fp8/fp4 アクティベーション丸め）を MLX で再現する。
パリティ検証（bf16 expert）と本番（MLX 4bit expert）の両方に対応。

重みはすべて bf16（fp4/fp8 デコード済み）を想定。expert のみ
（q, s, b）形式の MLX 4bit (group64) を指定可能。
"""

import math
import os
import time

import mlx.core as mx

FP8_MAX = 448.0
FP4_VALS = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_BOUNDS = mx.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def mlx_precompute_freqs(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
):
    """YaRN 付き rotary freq（cos/sin、[seqlen, dim//2]）。"""
    idx = mx.arange(0, dim, 2, dtype=mx.float32)
    freqs = 1.0 / (base ** (idx / dim))
    if original_seq_len > 0:
        dimf = dim // 2

        def find_correction_dim(num_rotations):
            return (
                dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = math.floor(find_correction_dim(beta_fast))
        high = math.ceil(find_correction_dim(beta_slow))
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = mx.clip((mx.arange(dimf, dtype=mx.float32) - low) / (high - low), 0, 1)
        freqs = freqs / factor * ramp + freqs * (1 - ramp)
    t = mx.arange(seqlen, dtype=mx.float32)
    f = mx.outer(t, freqs)
    return mx.cos(f), mx.sin(f)


def mlx_rotary(x, cos, sin, inverse=False):
    """x の末尾 rope_head_dim 次元へ rotary 適用。x は [b,s,h,d] か [b,s,d]。"""
    rd = cos.shape[-1] * 2
    xf = x.astype(mx.float32)
    if x.ndim == 3:
        c = cos[None, :, :]
        s = sin[None, :, :]
    else:
        c = cos[None, :, None, :]
        s = sin[None, :, None, :]
    xr = xf[..., -rd:].reshape(*xf.shape[:-1], -1, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    if inverse:
        y0 = x0 * c + x1 * s
        y1 = -x0 * s + x1 * c
    else:
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c
    out = mx.stack([y0, y1], axis=-1).reshape(*xf.shape[:-1], rd)
    y = mx.concatenate([xf[..., :-rd], out], axis=-1)
    return y.astype(mx.bfloat16)


def mlx_rms_norm(x, weight, eps):
    xf = x.astype(mx.float32)
    var = mx.mean(mx.square(xf), axis=-1, keepdims=True)
    return (xf * mx.rsqrt(var + eps) * weight.astype(mx.float32)).astype(x.dtype)


def mlx_round_fp8_e4m3(x):
    """e4m3 グリッドへの丸め（float 値のまま返す）。"""
    a = mx.abs(x)
    th = mx.array(2.0**-6)
    q_sub = mx.round(mx.minimum(a, th) * 512.0) / 512.0
    a_norm = mx.maximum(a, th)
    e = mx.floor(mx.log2(a_norm))
    q_norm = mx.power(2.0, e) * mx.round(8.0 * a / mx.power(2.0, e)) / 8.0
    q = mx.where(a < th, q_sub, q_norm)
    return mx.sign(x) * q


def mlx_act_quant(x, block_size=128, pow2_scale=True):
    """ブロック単位 fp8 量子化（QAT シミュレーション）。bf16 入力→bf16 出力。"""
    shape = x.shape
    N = shape[-1]
    xf = x.astype(mx.float32).reshape(-1, N // block_size, block_size)
    amax = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    amax = mx.maximum(amax, mx.array(1e-4))
    s = amax * (1.0 / FP8_MAX)
    if pow2_scale:
        s = mx.power(2.0, mx.ceil(mx.log2(s)))
    xq = mx.clip(xf / s, -FP8_MAX, FP8_MAX)
    y = mlx_round_fp8_e4m3(xq) * s
    return y.reshape(shape).astype(mx.bfloat16)


def mlx_round_fp4(x):
    a = mx.abs(x)
    idx = mx.sum(a[..., None] > FP4_BOUNDS, axis=-1)
    return mx.sign(x) * FP4_VALS[idx]


def mlx_fp4_quant(x, block_size=32):
    """ブロック単位 fp4 量子化（QAT シミュレーション）。bf16→bf16。"""
    shape = x.shape
    N = shape[-1]
    xf = x.astype(mx.float32).reshape(-1, N // block_size, block_size)
    amax = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    amax = mx.maximum(amax, mx.array(6.0 * 2.0**-126))
    s = mx.power(2.0, mx.ceil(mx.log2(amax * (1.0 / 6.0))))
    xq = mx.clip(xf / s, -6.0, 6.0)
    y = mlx_round_fp4(xq) * s
    return y.reshape(shape).astype(mx.bfloat16)


def mlx_hadamard(x):
    n = x.shape[-1]
    orig = x.shape
    x = x.astype(mx.float32).reshape(-1, n)
    h = 1
    while h < n:
        x = x.reshape(-1, 2, h)
        a, b = x[:, 0], x[:, 1]
        x = mx.concatenate([a + b, a - b], axis=1)
        h *= 2
    return x.reshape(orig) * (n**-0.5)


def mlx_sparse_attn(q, kv, sink, topk_idxs, scale):
    """スパースアテンション。q [b,s,h,d], kv [b,n,d], topk_idxs [b,s,T]。"""
    b, s, h, d = q.shape
    T = topk_idxs.shape[-1]
    valid = topk_idxs >= 0
    idx = mx.clip(topk_idxs, 0, kv.shape[1] - 1)
    bs = mx.broadcast_to(mx.arange(b).reshape(b, 1, 1), (b, s, T))
    kvg = kv[bs, idx]
    kvg = kvg * valid.reshape(b, s, T, 1).astype(kv.dtype)
    qf = q.astype(mx.float32)
    kf = kvg.astype(mx.float32)
    scores = mx.einsum("bshd,bstd->bsht", qf, kf) * scale
    scores = mx.where(valid.reshape(b, s, 1, T), scores, -mx.inf)
    m = mx.max(scores, axis=-1, keepdims=True)
    e = mx.exp(scores - m)
    denom = mx.sum(e, axis=-1, keepdims=True) + mx.exp(
        sink.astype(mx.float32).reshape(1, 1, h, 1) - m
    )
    o = mx.einsum("bsht,bstd->bshd", e, kf) / denom
    return o.astype(mx.bfloat16)


def mlx_hc_split_sinkhorn(
    mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
):
    """HC の pre/post/comb を Sinkhorn 反復で計算。mixes [..., mix_hc]。"""
    mix = mixes.astype(mx.float32).reshape(-1, (2 + hc_mult) * hc_mult)
    hc = hc_mult
    pre = mx.sigmoid(mix[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * mx.sigmoid(mix[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = mix[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb.reshape(-1, hc, hc)
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    return pre, post, comb


def mlx_window_topk(win, bsz, seqlen, start_pos):
    if start_pos >= win - 1:
        sp = start_pos % win
        m = mx.concatenate([mx.arange(sp + 1, win), mx.arange(0, sp + 1)])
        m = m.reshape(1, 1, -1)
        return mx.broadcast_to(m.astype(mx.int32), (bsz, 1, win))
    if start_pos > 0:
        m = mx.where(mx.arange(win) > start_pos, -1, mx.arange(win))
        m = m.reshape(1, 1, -1)
        return mx.broadcast_to(m.astype(mx.int32), (bsz, 1, win))
    base = mx.arange(seqlen).reshape(-1, 1)
    m = mx.clip(base - win + 1, 0, None) + mx.arange(min(seqlen, win))
    m = mx.where(m > base, -1, m)
    m = m.reshape(1, seqlen, -1)
    return mx.broadcast_to(m.astype(mx.int32), (bsz, seqlen, m.shape[-1]))


def mlx_compress_topk(ratio, bsz, seqlen, start_pos, offset):
    if start_pos > 0:
        m = mx.arange(0, (start_pos + 1) // ratio) + offset
        m = m.reshape(1, 1, -1)
        return mx.broadcast_to(m.astype(mx.int32), (bsz, 1, m.shape[-1]))
    nc = seqlen // ratio
    m = mx.broadcast_to(mx.arange(nc), (seqlen, nc))
    pos = mx.arange(1, seqlen + 1).reshape(-1, 1)
    m = mx.where(m >= pos // ratio, -1, m + offset)
    m = m.reshape(1, seqlen, nc)
    return mx.broadcast_to(m.astype(mx.int32), (bsz, seqlen, nc))


_ATTN_Q8 = os.environ.get("ELFMOON_ATTN_Q8", "1") == "1"


class MLXLinear:
    """bf16 重みの線形層。qkind 指定時は入力へ QAT fp8 丸め。"""

    def __init__(self, weight, qkind=None, quant8=False):
        self.qkind = qkind
        self.quant8 = quant8
        if quant8:
            self.wq, self.s, self.b = mx.quantize(weight, group_size=64, bits=8)
            self.weight = None  # int8 化後は bf16 原重みを破棄（メモリ削減）
        else:
            self.weight = weight  # bf16 [out, in]

    def __call__(self, x, fp32=True):
        if self.qkind is not None:
            x = mlx_act_quant(x, 128, pow2_scale=True)
        if self.quant8:
            return mx.quantized_matmul(
                x,
                self.wq,
                self.s,
                self.b,
                transpose=True,
                group_size=64,
                bits=8,
                mode="affine",
            )
        if fp32:
            return (x.astype(mx.float32) @ self.weight.astype(mx.float32).T).astype(
                mx.bfloat16
            )
        return (x @ self.weight.T).astype(mx.bfloat16)


class MLXCompressor:
    def __init__(self, args, compress_ratio, head_dim, rotate, wkv, wgate, ape, norm_w):
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.coff = 1 + self.overlap
        self.wkv = MLXLinear(wkv.astype(mx.float32), quant8=_ATTN_Q8)
        self.wgate = MLXLinear(wgate.astype(mx.float32), quant8=_ATTN_Q8)
        self.ape = ape.astype(mx.float32)
        self.norm_w = norm_w.astype(mx.float32)
        self.eps = args.norm_eps
        self.kv_cache = None
        self.cache_offset = 0
        self.freqs_cos = self.freqs_sin = None
        b = args.max_batch_size
        self.kv_state = mx.zeros(
            (b, self.coff * compress_ratio, self.coff * head_dim), mx.float32
        )
        self.score_state = mx.full(
            (b, self.coff * compress_ratio, self.coff * head_dim), -mx.inf, mx.float32
        )

    def overlap_transform(self, tensor, value):
        ratio, d = self.compress_ratio, self.head_dim
        b, s, _, _ = tensor.shape
        new = mx.full((b, s, 2 * ratio, d), value, tensor.dtype)
        new[:, :, ratio:] = tensor[:, :, :, d:]
        new[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new

    def forward(self, x, start_pos):
        ratio, overlap, d, rd, coff = (
            self.compress_ratio,
            self.overlap,
            self.head_dim,
            self.rope_head_dim,
            self.coff,
        )
        bsz, seqlen, _ = x.shape
        xf = x.astype(mx.float32)
        kv = self.wkv(xf)
        score = self.wgate(xf)
        should_compress = (
            seqlen >= ratio if start_pos == 0 else (start_pos + 1) % ratio == 0
        )
        if start_pos == 0:
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            if remainder > 0:
                kv, kv_tail = kv[:, :cutoff], kv[:, cutoff:]
                self.kv_state[:bsz, offset : offset + remainder] = kv_tail
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            kv = kv.reshape(bsz, -1, ratio, coff * d)
            score = score.reshape(bsz, -1, ratio, coff * d) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, -mx.inf)
            kv = mx.sum(kv * mx.softmax(score, axis=2), axis=2)
        else:
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv[:, 0]
                self.score_state[:bsz, ratio + start_pos % ratio] = score[:, 0]
                if should_compress:
                    kv_state = mx.concatenate(
                        [
                            self.kv_state[:bsz, :ratio, :d],
                            self.kv_state[:bsz, ratio:, d:],
                        ],
                        axis=1,
                    )
                    score_state = mx.concatenate(
                        [
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ],
                        axis=1,
                    )
                    kv = mx.sum(
                        kv_state * mx.softmax(score_state, axis=1),
                        axis=1,
                        keepdims=True,
                    )
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv[:, 0]
                self.score_state[:bsz, start_pos % ratio] = score[:, 0]
                if should_compress:
                    kv = mx.sum(
                        self.kv_state[:bsz]
                        * mx.softmax(self.score_state[:bsz], axis=1),
                        axis=1,
                        keepdims=True,
                    )
        if not should_compress:
            return None
        kv = mlx_rms_norm(kv.astype(mx.bfloat16), self.norm_w, self.eps)
        if start_pos == 0:
            c = self.freqs_cos[:cutoff:ratio]
            s = self.freqs_sin[:cutoff:ratio]
        else:
            c = self.freqs_cos[start_pos + 1 - ratio][None]
            s = self.freqs_sin[start_pos + 1 - ratio][None]
        kv = mlx_rotary(kv, c, s)
        if self.rotate:
            kv = mlx_hadamard(kv).astype(mx.bfloat16)
            kv = mlx_fp4_quant(kv, 32)
        else:
            kv = mx.concatenate(
                [mlx_act_quant(kv[..., :-rd], 64, pow2_scale=True), kv[..., -rd:]],
                axis=-1,
            )
        if start_pos == 0:
            self.kv_cache[
                :bsz, self.cache_offset : self.cache_offset + seqlen // ratio
            ] = kv
        else:
            self.kv_cache[:bsz, self.cache_offset + start_pos // ratio] = kv[:, 0]
        return kv


class MLXIndexer:
    def __init__(self, args, compress_ratio, wq_b, weights_proj, comp_weights):
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.wq_b = MLXLinear(wq_b, qkind="fp8", quant8=_ATTN_Q8)
        self.weights_proj = MLXLinear(weights_proj, quant8=_ATTN_Q8)
        self.softmax_scale = self.head_dim**-0.5
        self.compressor = MLXCompressor(
            args, compress_ratio, self.head_dim, True, *comp_weights
        )
        self.kv_cache = mx.zeros(
            (args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim),
            mx.bfloat16,
        )
        self.freqs_cos = self.freqs_sin = None

    def forward(self, x, qr, start_pos, offset):
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cos = self.freqs_cos
            self.compressor.freqs_sin = self.freqs_sin
        q = self.wq_b(qr)
        q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        q = mlx_rotary(
            q,
            self.freqs_cos[start_pos : start_pos + seqlen],
            self.freqs_sin[start_pos : start_pos + seqlen],
        )
        q = mlx_hadamard(q).astype(mx.bfloat16)
        q = mlx_fp4_quant(q, 32)
        self.compressor.forward(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = mx.einsum(
            "bshd,btd->bsht",
            q.astype(mx.float32),
            self.kv_cache[:bsz, : end_pos // ratio].astype(mx.float32),
        )
        index_score = mx.sum(
            mx.maximum(index_score, 0)
            * weights.astype(mx.float32).reshape(bsz, seqlen, self.n_heads, 1),
            axis=2,
        )
        if start_pos == 0:
            pos = mx.arange(1, seqlen + 1).reshape(-1, 1)
            m = mx.broadcast_to(mx.arange(seqlen // ratio), (seqlen, seqlen // ratio))
            mask = m >= pos // ratio
            index_score = index_score + mx.where(mask, -mx.inf, 0)
        n_avail = min(self.index_topk, end_pos // ratio)
        topk_idxs = mx.argsort(-index_score, axis=-1)[..., :n_avail].astype(mx.int32)
        if start_pos == 0:
            pos = mx.arange(1, seqlen + 1).reshape(-1, 1)
            mask = topk_idxs >= pos // ratio
            topk_idxs = mx.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs

    __call__ = forward


class MLXAttention:
    def __init__(self, args, layer_id, w):
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.softmax_scale = self.head_dim**-0.5
        self.attn_sink = w["attn_sink"].astype(mx.float32)
        self.wq_a = MLXLinear(w["wq_a"], qkind="fp8", quant8=_ATTN_Q8)
        self.q_norm_w = w["q_norm"].astype(mx.float32)
        self.wq_b = MLXLinear(w["wq_b"], qkind="fp8", quant8=_ATTN_Q8)
        self.wkv = MLXLinear(w["wkv"], qkind="fp8", quant8=_ATTN_Q8)
        self.kv_norm_w = w["kv_norm"].astype(mx.float32)
        self.wo_a = w["wo_a"].astype(mx.bfloat16)  # [groups, o_lora, d]
        self.wo_b = MLXLinear(w["wo_b"], qkind="fp8", quant8=_ATTN_Q8)
        if self.compress_ratio:
            self.compressor = MLXCompressor(
                args, self.compress_ratio, self.head_dim, False, *w["compressor"]
            )
            if self.compress_ratio == 4:
                self.indexer = MLXIndexer(args, self.compress_ratio, *w["indexer"])
            else:
                self.indexer = None
        kv_cache_size = args.window_size + (
            args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
        )
        self.kv_cache = mx.zeros(
            (args.max_batch_size, kv_cache_size, self.head_dim), mx.bfloat16
        )
        if self.compress_ratio:
            oseq, theta = args.original_seq_len, args.compress_rope_theta
        else:
            oseq, theta = 0, args.rope_theta
        self.freqs_cos, self.freqs_sin = mlx_precompute_freqs(
            self.rope_head_dim,
            args.max_seq_len,
            oseq,
            theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )

    def forward(self, x, start_pos):
        bsz, seqlen, _ = x.shape
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        fc = self.freqs_cos[start_pos : start_pos + seqlen]
        fs = self.freqs_sin[start_pos : start_pos + seqlen]
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.cache_offset = win
            self.compressor.freqs_cos = self.freqs_cos
            self.compressor.freqs_sin = self.freqs_sin
            if self.indexer is not None:
                self.indexer.freqs_cos = self.freqs_cos
                self.indexer.freqs_sin = self.freqs_sin
        qr = mlx_rms_norm(self.wq_a(x), self.q_norm_w, self.eps)
        q = self.wq_b(qr)
        q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        q = q * mx.rsqrt(
            mx.mean(mx.square(q.astype(mx.float32)), axis=-1, keepdims=True) + self.eps
        )
        q = q.astype(mx.bfloat16)
        q = mlx_rotary(q, fc, fs)

        kv = self.wkv(x)
        kv = mlx_rms_norm(kv, self.kv_norm_w, self.eps)
        kv = mlx_rotary(kv, fc, fs)
        kv = mx.concatenate(
            [mlx_act_quant(kv[..., :-rd], 64, pow2_scale=True), kv[..., -rd:]],
            axis=-1,
        )
        topk_idxs = mlx_window_topk(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.shape[1] if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = mlx_compress_topk(
                    ratio, bsz, seqlen, start_pos, offset
                )
            topk_idxs = mx.concatenate([topk_idxs, compress_topk_idxs], axis=-1)

        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                klast = kv[:, -win:]
                if cutoff > 0:
                    self.kv_cache[:bsz, cutoff:win] = klast[:, : win - cutoff]
                    self.kv_cache[:bsz, :cutoff] = klast[:, win - cutoff :]
            if self.compress_ratio:
                kv_compress = self.compressor.forward(x, start_pos)
                if kv_compress is not None:
                    kv = mx.concatenate([kv, kv_compress], axis=1)
            o = mlx_sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv[:, 0]
            if self.compress_ratio:
                self.compressor.forward(x, start_pos)
            o = mlx_sparse_attn(
                q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale
            )
        o = mlx_rotary(o, fc, fs, inverse=True)

        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        o = mx.einsum(
            "bsgd,grd->bsgr", o.astype(mx.float32), self.wo_a.astype(mx.float32)
        ).astype(mx.bfloat16)
        x = self.wo_b(o.reshape(bsz, seqlen, -1))
        self.debug = dict(q=q, kv=kv, topk_idxs=topk_idxs, o_woa=o, out=x)
        return x


class MLXGate:
    def __init__(self, args, layer_id, weight, bias_or_tid):
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = weight.astype(mx.bfloat16)
        if self.hash:
            self.tid2eid = bias_or_tid.astype(mx.int64)
            self.bias = None
        else:
            self.bias = bias_or_tid.astype(mx.float32)

    def forward(self, x, input_ids):
        scores = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        if self.score_func == "softmax":
            scores = mx.softmax(scores, axis=-1)
        elif self.score_func == "sigmoid":
            scores = mx.sigmoid(scores)
        else:
            scores = mx.sqrt(mx.logaddexp(mx.zeros_like(scores), scores))
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = mx.argsort(-scores, axis=-1)[..., : self.topk].astype(mx.int32)
        weights = mx.take_along_axis(original_scores, indices, axis=-1)
        if self.score_func != "softmax":
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        weights = weights * self.route_scale
        return weights, indices


class MLXExpert:
    def __init__(self, w1, w2, w3, swiglu_limit=0.0, quant=None, bits=4):
        self.swiglu_limit = swiglu_limit
        self.bits = bits
        if quant is not None:
            self.w1 = ("q", quant[0])
            self.w2 = ("q", quant[1])
            self.w3 = ("q", quant[2])
        else:
            self.w1 = ("b", w1)
            self.w2 = ("b", w2)
            self.w3 = ("b", w3)

    def _matmul(self, kind, w, x):
        if kind == "q":
            q, s, b = w
            xq = mlx_act_quant(x, 128, pow2_scale=True)
            return mx.quantized_matmul(xq, q, s, b, group_size=64, bits=self.bits)
        xq = mlx_act_quant(x, 128, pow2_scale=True)
        return (xq.astype(mx.float32) @ w.astype(mx.float32).T).astype(mx.bfloat16)

    def forward(self, x, weights=None):
        gate = self._matmul(*self.w1, x).astype(mx.float32)
        up = self._matmul(*self.w3, x).astype(mx.float32)
        if self.swiglu_limit > 0:
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.clip(gate, None, self.swiglu_limit)
        y = mx.sigmoid(gate) * gate * up
        if weights is not None:
            y = weights * y
        return self._matmul(*self.w2, y.astype(mx.bfloat16))


class MLXMoE:
    def __init__(
        self,
        args,
        layer_id,
        gate_w,
        gate_b,
        experts_w,
        shared_w,
        shared_quant=None,
        bits=4,
    ):
        self.gate = MLXGate(args, layer_id, gate_w, gate_b)
        self.experts = [
            MLXExpert(
                experts_w[e]["w1"],
                experts_w[e]["w2"],
                experts_w[e]["w3"],
                args.swiglu_limit,
                experts_w[e].get("quant"),
                bits,
            )
            for e in range(len(experts_w))
        ]
        self.shared = MLXExpert(
            shared_w["w1"],
            shared_w["w2"],
            shared_w["w3"],
            args.swiglu_limit,
            shared_quant,
            bits,
        )

    def forward(self, x, input_ids):
        shape = x.shape
        xf = x.reshape(-1, x.shape[-1])
        weights, indices = self.gate.forward(xf, input_ids.flatten())
        self.debug = dict(weights=weights, indices=indices)
        y = mx.zeros(xf.shape, mx.float32)
        n_exp = len(self.experts)
        for i in range(n_exp):
            m = indices == i
            if not mx.any(m):
                continue
            has = mx.any(m, axis=-1)
            n = has.shape[0]
            num = int(mx.sum(has.astype(mx.int32)))
            idx = mx.sort(mx.where(has, mx.arange(n), n))[:num]
            wsel = mx.sum(mx.where(m, weights, 0), axis=-1)
            y = y.at[idx].add(
                self.experts[i].forward(xf[idx], wsel[idx, None]).astype(mx.float32)
            )
        y = y + self.shared.forward(xf).astype(mx.float32)
        return y.astype(mx.bfloat16).reshape(shape)


def _gather_sort_mlx(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort_mlx(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class MLXFusedMoE:
    """融合 expert テンソル [n_exp, out, in]（MLX 4bit）を gather_qmm で一括計算。

    既存 MLXMoE の per-expert ループ（Python 同期 + 256 回 dispatch）を、
    expert ソート付き 3 カーネルに置き換える。数値は per-expert 経路と
    bf16 丸め差（~1e-3）で一致する。
    """

    def __init__(
        self,
        args,
        layer_id,
        gate_w,
        gate_b,
        fused,
        shared,
        group_size=64,
        bits=4,
        fused_loader=None,
    ):
        self.gate = MLXGate(args, layer_id, gate_w, gate_b)
        self.group_size = group_size
        self.bits = bits
        self.swiglu_limit = args.swiglu_limit
        # fused_loader が指定された場合、expert 重みは forward 毎にロードする
        # （層単位ストリーミング）。そうでなければ fused を直接保持。
        self.fused_loader = fused_loader
        self.gate_w = fused["gate"] if fused is not None else None
        self.up_w = fused["up"] if fused is not None else None
        self.down_w = fused["down"] if fused is not None else None
        # shared expert: bf16（非expert なので常駐）
        self.shared = MLXExpert(
            shared["w1"], shared["w2"], shared["w3"], args.swiglu_limit
        )

    def _load_fused(self):
        if self.fused_loader is not None:
            return self.fused_loader()
        return {"gate": self.gate_w, "up": self.up_w, "down": self.down_w}

    def _gq(self, xin, w, rhs_idx, do_sort):
        return mx.gather_qmm(
            xin,
            w["wq"],
            w["s"],
            w["b"],
            rhs_indices=rhs_idx,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
            sorted_indices=do_sort,
        )

    def forward(self, x, input_ids):
        fused = self._load_fused()
        if self.fused_loader is None:
            return self._forward_fused_direct(x, input_ids, fused)
        return self._forward_fused(x, input_ids, fused)

    def _forward_fused_direct(self, x, input_ids, fused):
        """全常駐用: 融合テンソル全体を GPU に保持し、rhs_indices で直接選択。

        `mx.take` による抽出をスキップし、gather_qmm をそのまま使う。
        expert が全て GPU 常駐のときに高速。
        """
        shape = x.shape
        xf = x.reshape(-1, x.shape[-1])
        weights, indices = self.gate.forward(xf, input_ids.flatten())
        self.debug = dict(weights=weights, indices=indices)
        N = xf.shape[0]
        topk = indices.shape[-1]
        x = mx.expand_dims(xf, (-2, -3))  # [N, 1, 1, DIM]
        xs = mx.broadcast_to(x, (N, topk, 1, xf.shape[-1])).reshape(-1, 1, xf.shape[-1])
        ii = indices.flatten().astype(mx.int32)

        def gq(xin, w):
            xq = mlx_act_quant(xin, 128, pow2_scale=True)
            return mx.gather_qmm(
                xq,
                w["wq"],
                w["s"],
                w["b"],
                rhs_indices=ii,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode="affine",
                sorted_indices=False,
            )

        g = gq(xs, fused["gate"])
        u = gq(xs, fused["up"])
        if self.swiglu_limit > 0:
            u = mx.clip(u, -self.swiglu_limit, self.swiglu_limit)
            g = mx.clip(g, None, self.swiglu_limit)
        h = mx.sigmoid(g) * g * u
        yo = gq(h, fused["down"])
        yo = yo.reshape(N, topk, -1)
        y = mx.sum(weights[:, :, None].astype(yo.dtype) * yo, axis=1)
        y = y + self.shared.forward(xf).astype(mx.float32)
        return y.astype(mx.bfloat16).reshape(shape)

    def _forward_fused(self, x, input_ids, fused):
        shape = x.shape
        xf = x.reshape(-1, x.shape[-1])
        weights, indices = self.gate.forward(xf, input_ids.flatten())
        self.debug = dict(weights=weights, indices=indices)
        # xf: [N, DIM], indices: [N, topk]
        # 選択された expert のユニーク集合を抽出し、その重みだけを GPU に載せる
        # （融合テンソル [256,out,in] 全体のアップロードを回避して高速化）。
        flat = indices.flatten().astype(mx.int32)
        order = mx.argsort(flat).astype(mx.int32)
        sorted_f = flat[order]
        uniq_mask = mx.concatenate([mx.array([True]), sorted_f[1:] != sorted_f[:-1]])
        pos = mx.where(uniq_mask, mx.arange(flat.shape[0]), flat.shape[0])
        n_uniq = int(mx.sum(uniq_mask.astype(mx.int32)))
        uniq = sorted_f[mx.sort(pos)[:n_uniq]]
        cum = mx.cumsum(uniq_mask.astype(mx.int32)) - 1
        sub_idx = mx.zeros(flat.shape, mx.int32)
        sub_idx[order] = cum  # 各要素の uniq 内位置 [N*topk]

        x = mx.expand_dims(xf, (-2, -3))  # [N, 1, 1, DIM]
        # 各トークンの topk 分に展開: [N, topk, 1, DIM]
        N = xf.shape[0]
        topk = indices.shape[-1]
        xs = mx.broadcast_to(x, (N, topk, 1, xf.shape[-1])).reshape(-1, 1, xf.shape[-1])

        def gq(xin, w):
            xq = mlx_act_quant(xin, 128, pow2_scale=True)
            # 選択 expert の重みのみ抽出（全常駐時は GPU 内、ストリーミング時は mmap 行抽出）
            sq = mx.take(w["wq"], uniq, axis=0)
            ss = mx.take(w["s"], uniq, axis=0)
            sb = mx.take(w["b"], uniq, axis=0)
            return mx.gather_qmm(
                xq,
                sq,
                ss,
                sb,
                rhs_indices=sub_idx,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode="affine",
                sorted_indices=True,
            )

        g = gq(xs, fused["gate"])
        u = gq(xs, fused["up"])
        if self.swiglu_limit > 0:
            u = mx.clip(u, -self.swiglu_limit, self.swiglu_limit)
            g = mx.clip(g, None, self.swiglu_limit)
        h = mx.sigmoid(g) * g * u
        yo = gq(h, fused["down"])
        # yo: [N*topk, 1, DIM]（xs の行順）-> [N, topk, DIM]
        yo = yo.reshape(N, topk, -1)
        y = mx.sum(weights[:, :, None].astype(yo.dtype) * yo, axis=1)
        y = y + self.shared.forward(xf).astype(mx.float32)
        return y.astype(mx.bfloat16).reshape(shape)


@mx.compile
def _decode_ffn_core(
    xf, w_gu, s_gu, b_gu, w_dw, s_dw, b_dw, weights, swiglu_limit, group_size, bits
):
    """decode（N=1）の FFN matmul コアを単一コンパイル済みグラフとして実行。

    per-expert の stack 結果（静的シェイプ）を渡し、act_quant + 2 本の
    quantized_matmul + swiglu + 重み付き集約を 1 回のトレースで処理する。
    mx.compile により Python dispatch と要素演算の複数カーネル化を排除する。
    """
    topk = w_gu.shape[0] // 2
    xb = mx.broadcast_to(xf, (2 * topk, 1, xf.shape[-1]))
    xq = mlx_act_quant(xb, 128, pow2_scale=True)
    gu = mx.quantized_matmul(
        xq,
        w_gu,
        s_gu,
        b_gu,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    g, u = gu[:topk], gu[topk:]
    if swiglu_limit > 0:
        u = mx.clip(u, -swiglu_limit, swiglu_limit)
        g = mx.clip(g, None, swiglu_limit)
    h = mx.sigmoid(g) * g * u
    hq = mlx_act_quant(h.astype(mx.bfloat16), 128, pow2_scale=True)
    yo = mx.quantized_matmul(
        hq,
        w_dw,
        s_dw,
        b_dw,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    return mx.sum(yo[:, 0, :] * weights.astype(yo.dtype)[:, None], axis=0)


class MLXStreamingMoE:
    """V4 用 部分常駐 MoE（per-expert 粒度）。
    融合テンソル経路（MLXFusedMoE）は mmap 配列を消費する際にソース全体が
    実メモリに具体化されるため、43 層全部で 137GB に達し 128GB 機では成立
    しない（実測）。本クラスは store の per-expert ファイル（~13MB/個）を
    ResidentCache のバイト予算で常駐させ、ミス分だけストリームする。

    gate / shared expert は常駐。routed expert は (layer, expert) 単位の
    ResidentCache 経由。計算は MLXFusedMoE と同一の act_quant + quantized_matmul
    （swiglu_limit クリップ込み）で数値パリティを維持する。
    """

    def __init__(
        self,
        args,
        layer_id,
        gate_w,
        gate_b,
        shared_w,
        cache,
        store,
        group_size=64,
        bits=4,
    ):
        self.gate = MLXGate(args, layer_id, gate_w, gate_b)
        self.shared = MLXExpert(
            shared_w["w1"], shared_w["w2"], shared_w["w3"], args.swiglu_limit
        )
        self.swiglu_limit = args.swiglu_limit
        self.layer_idx = layer_id
        self._cache = cache
        self._store = store
        self.group_size = group_size
        self.bits = bits
        self._acc_load = 0.0
        self._acc_matmul = 0.0
        self._acc_dispatch = 0.0
        self._acc_shared = 0.0

    def _load(self, e):
        return self._cache.get(
            (self.layer_idx, e),
            lambda e=e: self._store.load(self.layer_idx, e),
        )

    def _gq(self, xin, w):
        xq = mlx_act_quant(xin, 128, pow2_scale=True)
        return mx.quantized_matmul(
            xq,
            w["wq"],
            w["s"],
            w["b"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )

    def forward(self, x, input_ids):
        shape = x.shape
        xf = x.reshape(-1, x.shape[-1])
        weights, indices = self.gate.forward(xf, input_ids.flatten())
        self.debug = dict(weights=weights, indices=indices)
        prof = os.environ.get("ELFMOON_PROFILE") == "1"
        _t0 = _t1 = _t2 = 0.0
        if prof:
            mx.synchronize()
            _t0 = time.perf_counter()
        N = xf.shape[0]
        topk = indices.shape[-1]
        if N == 1:
            y = self._decode(xf, weights, indices, topk)
        else:
            y = self._prefill(xf, weights, indices, topk)
        if prof:
            mx.synchronize()
            _t1 = time.perf_counter()
        y = y + self.shared.forward(xf).astype(mx.float32)
        if prof:
            mx.synchronize()
            _t2 = time.perf_counter()
            self._acc_dispatch += _t1 - _t0
            self._acc_shared += _t2 - _t1
        return y.astype(mx.bfloat16).reshape(shape)

    def _decode(self, xf, weights, indices, topk):
        """1 トークン decode: 選択 expert を cache から取り、stack + gather。"""
        prof = os.environ.get("ELFMOON_PROFILE") == "1"
        _t0 = _t1 = _t2 = 0.0
        flat = indices.flatten().astype(mx.int32)
        mx.eval(flat)
        exps = [self._load(int(e)) for e in flat.tolist()]
        if prof:
            mx.synchronize()
            _t0 = time.perf_counter()
        w_gu = mx.stack([e["gate.wq"] for e in exps] + [e["up.wq"] for e in exps])
        s_gu = mx.stack([e["gate.s"] for e in exps] + [e["up.s"] for e in exps])
        b_gu = mx.stack([e["gate.b"] for e in exps] + [e["up.b"] for e in exps])
        w_dw = mx.stack([e["down.wq"] for e in exps])
        s_dw = mx.stack([e["down.s"] for e in exps])
        b_dw = mx.stack([e["down.b"] for e in exps])
        if prof:
            mx.synchronize()
            _t1 = time.perf_counter()

        y = _decode_ffn_core(
            xf,
            w_gu,
            s_gu,
            b_gu,
            w_dw,
            s_dw,
            b_dw,
            weights.flatten(),
            self.swiglu_limit,
            self.group_size,
            self.bits,
        )
        if prof:
            mx.synchronize()
            _t2 = time.perf_counter()
            self._acc_load += _t1 - _t0
            self._acc_matmul += _t2 - _t1
        return y

    def _prefill(self, xf, weights, indices, topk):
        """N>1 prefill: expert 毎にトークンをバッチ処理（scatter-add 集約）。"""
        mx.eval(indices, weights)
        idx_l = indices.tolist()
        w_l = weights.tolist()
        N = xf.shape[0]
        expert_groups = {}
        for t in range(N):
            for j in range(topk):
                e = int(idx_l[t][j])
                expert_groups.setdefault(e, []).append((t, w_l[t][j]))
        out = mx.zeros((N, xf.shape[-1]), dtype=mx.float32)
        for e, items in expert_groups.items():
            exp = self._load(e)
            ts = mx.array([it[0] for it in items])
            ws = mx.array([it[1] for it in items])
            xb = xf[ts]
            g = self._gq(
                xb, {"wq": exp["gate.wq"], "s": exp["gate.s"], "b": exp["gate.b"]}
            )
            u = self._gq(xb, {"wq": exp["up.wq"], "s": exp["up.s"], "b": exp["up.b"]})
            if self.swiglu_limit > 0:
                u = mx.clip(u, -self.swiglu_limit, self.swiglu_limit)
                g = mx.clip(g, None, self.swiglu_limit)
            h = mx.sigmoid(g) * g * u
            yo = self._gq(
                h.astype(mx.bfloat16),
                {"wq": exp["down.wq"], "s": exp["down.s"], "b": exp["down.b"]},
            )
            out = out.at[ts].add(yo * ws[:, None].astype(yo.dtype))
        return out


class MLXBlock:
    def __init__(
        self,
        args,
        layer_id,
        w,
        use_fused_moe=False,
        fused_loader=None,
        bits=4,
        moe_mode="fused",
        cache=None,
        store=None,
    ):
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        self.acc_attn = 0.0
        self.acc_ffn = 0.0
        self.attn = MLXAttention(args, layer_id, w["attn"])
        if moe_mode == "stream":
            self.ffn = MLXStreamingMoE(
                args,
                layer_id,
                w["ffn"]["gate_w"],
                w["ffn"]["gate_b"],
                w["ffn"]["shared"],
                cache,
                store,
                bits=bits,
            )
        elif use_fused_moe:
            self.ffn = MLXFusedMoE(
                args,
                layer_id,
                w["ffn"]["gate_w"],
                w["ffn"]["gate_b"],
                w["ffn"].get("fused"),
                w["ffn"]["shared"],
                fused_loader=fused_loader,
                bits=bits,
            )
        else:
            self.ffn = MLXMoE(
                args,
                layer_id,
                w["ffn"]["gate_w"],
                w["ffn"]["gate_b"],
                w["ffn"]["experts"],
                w["ffn"]["shared"],
                w["ffn"].get("shared_quant"),
                bits=bits,
            )
        self.attn_norm_w = w["attn_norm"].astype(mx.float32)
        self.ffn_norm_w = w["ffn_norm"].astype(mx.float32)
        self.hc_attn_fn = w["hc_attn_fn"].astype(mx.float32)
        self.hc_ffn_fn = w["hc_ffn_fn"].astype(mx.float32)
        self.hc_attn_base = w["hc_attn_base"].astype(mx.float32)
        self.hc_ffn_base = w["hc_ffn_base"].astype(mx.float32)
        self.hc_attn_scale = w["hc_attn_scale"].astype(mx.float32)
        self.hc_ffn_scale = w["hc_ffn_scale"].astype(mx.float32)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape = x.shape
        xf = x.astype(mx.float32).reshape(shape[0], shape[1], -1)
        rsqrt = mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + self.norm_eps)
        mixes = (xf @ hc_fn.T) * rsqrt
        pre, post, comb = mlx_hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        pre = pre.reshape(shape[0], shape[1], self.hc_mult)
        y = mx.sum(pre[..., None] * xf.reshape(shape), axis=2)
        return y.astype(x.dtype), post, comb

    def hc_post(self, x, residual, post, comb):
        y = post[..., None] * mx.expand_dims(x, -2) + mx.sum(
            comb[..., None] * mx.expand_dims(residual, -2), axis=2
        )
        return y.astype(x.dtype)

    def hc_head(self, x, hc_fn, hc_scale, hc_base):
        shape = x.shape
        xf = x.astype(mx.float32).reshape(shape[0], shape[1], -1)
        rsqrt = mx.rsqrt(mx.mean(mx.square(xf), axis=-1, keepdims=True) + self.norm_eps)
        mixes = (xf @ hc_fn.T) * rsqrt
        pre = mx.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = mx.sum(pre[..., None] * xf.reshape(shape), axis=2)
        return y.astype(x.dtype)

    def forward(self, x, start_pos, input_ids=None):
        self.debug = {}
        prof = os.environ.get("ELFMOON_PROFILE") == "1"
        _t0 = _t_attn = _t_ffn = 0.0
        if prof:
            mx.synchronize()
            _t0 = time.perf_counter()
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        self.debug["pre_attn"] = x
        x = mlx_rms_norm(x, self.attn_norm_w, self.norm_eps)
        self.debug["attn_norm"] = x
        x = self.attn.forward(x, start_pos)
        self.debug["attn_out"] = x
        self.debug["attn"] = self.attn.debug
        if prof:
            mx.synchronize()
            _t_attn = time.perf_counter()
        x = self.hc_post(x, residual, post, comb)
        self.debug["hc_post1"] = x
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        self.debug["pre_ffn"] = x
        x = mlx_rms_norm(x, self.ffn_norm_w, self.norm_eps)
        self.debug["ffn_norm"] = x
        x = self.ffn.forward(x, input_ids)
        self.debug["ffn_out"] = x
        self.debug["ffn"] = self.ffn.debug
        if prof:
            mx.synchronize()
            _t_ffn = time.perf_counter()
        x = self.hc_post(x, residual, post, comb)
        self.debug["hc_post2"] = x
        if prof:
            self.acc_attn += _t_attn - _t0
            self.acc_ffn += _t_ffn - _t_attn
        return x


class MLXV4Model:
    def __init__(
        self,
        args,
        w,
        layer_ids=None,
        expert_quant=False,
        use_fused_moe=False,
        fused_loaders=None,
        bits=4,
        moe_mode="fused",
        cache=None,
        store=None,
    ):
        self.args = args
        self.hc_mult = args.hc_mult
        self.hc_eps = args.hc_eps
        self.norm_eps = args.norm_eps
        self.layer_ids = list(range(args.n_layers) if layer_ids is None else layer_ids)
        self.embed = w["embed"].astype(mx.bfloat16)
        self.blocks = [
            MLXBlock(
                args,
                lid,
                w["layers"][lid],
                use_fused_moe,
                fused_loaders.get(lid) if fused_loaders else None,
                bits,
                moe_mode=moe_mode,
                cache=cache,
                store=store,
            )
            for lid in self.layer_ids
        ]
        self.norm_w = (
            w.get("norm").astype(mx.float32) if w.get("norm") is not None else None
        )
        self.head = (
            w.get("head").astype(mx.float32) if w.get("head") is not None else None
        )
        # head を GPU へ事前ロード（毎回の matmul 評価時の 20s 転送を回避）
        if self.head is not None:
            mx.eval(self.head)
        self.hc_head_fn = (
            w.get("hc_head_fn").astype(mx.float32)
            if w.get("hc_head_fn") is not None
            else None
        )
        self.hc_head_base = (
            w.get("hc_head_base").astype(mx.float32)
            if w.get("hc_head_base") is not None
            else None
        )
        self.hc_head_scale = (
            w.get("hc_head_scale").astype(mx.float32)
            if w.get("hc_head_scale") is not None
            else None
        )

    def forward_block(self, x, input_ids, start_pos=0):
        for block in self.blocks:
            x = block.forward(x, start_pos, input_ids)
        return x

    def forward(self, input_ids, start_pos=0, return_hidden=False):
        h = mx.take(self.embed, input_ids, axis=0)
        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
        )
        h = self.forward_block(h, input_ids, start_pos)
        h = self.blocks[-1].hc_head(
            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden = h
        logits = hidden.astype(mx.float32) @ self.head.T
        if return_hidden:
            return logits, hidden
        return logits


def build_mlx_layer(args, state, layer_id, expert_quant=False):
    """デコード済み state dict から MLXBlock 用の w を構築。"""
    base = f"layers.{layer_id}."
    attn = {}
    for t in ("wq_a", "wq_b", "wkv", "wo_b"):
        attn[t] = state[f"{base}attn.{t}.weight"]
    woa = state[f"{base}attn.wo_a.weight"]
    g, r, d = args.o_groups, args.o_lora_rank, woa.shape[1]
    attn["wo_a"] = woa.reshape(g, r, d)
    for t in ("q_norm", "kv_norm"):
        attn[t] = state[f"{base}attn.{t}.weight"]
    attn["attn_sink"] = state[f"{base}attn.attn_sink"]
    if args.compress_ratios[layer_id]:
        attn["compressor"] = (
            state[f"{base}attn.compressor.wkv.weight"],
            state[f"{base}attn.compressor.wgate.weight"],
            state[f"{base}attn.compressor.ape"],
            state[f"{base}attn.compressor.norm.weight"],
        )
        if args.compress_ratios[layer_id] == 4:
            attn["indexer"] = (
                state[f"{base}attn.indexer.wq_b.weight"],
                state[f"{base}attn.indexer.weights_proj.weight"],
                (
                    state[f"{base}attn.indexer.compressor.wkv.weight"],
                    state[f"{base}attn.indexer.compressor.wgate.weight"],
                    state[f"{base}attn.indexer.compressor.ape"],
                    state[f"{base}attn.indexer.compressor.norm.weight"],
                ),
            )
    ffn = {}
    ffn["gate_w"] = state[f"{base}ffn.gate.weight"]
    if layer_id < args.n_hash_layers:
        ffn["gate_b"] = state[f"{base}ffn.gate.tid2eid"]
    else:
        ffn["gate_b"] = state[f"{base}ffn.gate.bias"]
    ffn["experts"] = []
    for e in range(args.n_routed_experts):
        w1 = state[f"{base}ffn.experts.{e}.w1.weight"]
        w2 = state[f"{base}ffn.experts.{e}.w2.weight"]
        w3 = state[f"{base}ffn.experts.{e}.w3.weight"]
        if expert_quant:
            from elfmoon.convert_v4 import quant_expert

            quant = tuple(quant_expert(t) for t in (w1, w2, w3))
        else:
            quant = None
        ffn["experts"].append(
            {
                "w1": w1,
                "w2": w2,
                "w3": w3,
                "quant": quant,
            }
        )
    ffn["shared"] = {
        "w1": state[f"{base}ffn.shared_experts.w1.weight"],
        "w2": state[f"{base}ffn.shared_experts.w2.weight"],
        "w3": state[f"{base}ffn.shared_experts.w3.weight"],
    }
    return {
        "attn": attn,
        "ffn": ffn,
        "attn_norm": state[f"{base}attn_norm.weight"],
        "ffn_norm": state[f"{base}ffn_norm.weight"],
        "hc_attn_fn": state[f"{base}hc_attn_fn"],
        "hc_ffn_fn": state[f"{base}hc_ffn_fn"],
        "hc_attn_base": state[f"{base}hc_attn_base"],
        "hc_ffn_base": state[f"{base}hc_ffn_base"],
        "hc_attn_scale": state[f"{base}hc_attn_scale"],
        "hc_ffn_scale": state[f"{base}hc_ffn_scale"],
    }


def load_mlx_model(args, state, layer_ids=None, expert_quant=False):
    """MLXV4Model を構築。state はデコード済みトーチテンソル dict。"""
    w = {"layers": {}, "embed": state["embed.weight"]}
    for lid in layer_ids if layer_ids is not None else range(args.n_layers):
        w["layers"][lid] = build_mlx_layer(args, state, lid, expert_quant)
    return MLXV4Model(args, w, layer_ids=layer_ids, expert_quant=expert_quant)


def load_v4_fused(model_path, args, layer_ids=None, streaming=True, bits=4):
    """変換済み safetensors（mmap ロード）からフルモデルを構築する。

    convert_v4.py の出力（非expert は bf16/fp32、expert は switch_mlp 融合 4bit）
    を index.json 経由で読み、MLXFusedMoE を使う MLXV4Model を返す。

    streaming=True（既定）: expert 融合テンソルは保持せず、各層の forward 毎に
    shard を mx.load して gather_qmm、計算後に解放する。常駐メモリは 1 層分のみ。
    streaming=False: 従来どおり全層の expert を保持（RAM 128GB を超える）。

    層 0/1 は旧命名（layers.{l}.mlp.switch_mlp.*）、層 2+ は
    layers.{l}.ffn.switch_mlp.* に保存されているため両対応する。
    """
    import json
    import os

    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    wm = json.load(open(idx_path))["weight_map"]

    if layer_ids is None:
        layer_ids = list(range(args.n_layers))

    def load_tensor(key):
        shard = wm[key]
        data = mx.load(os.path.join(model_path, shard))
        return data[key]

    def has(key):
        return key in wm

    w = {"layers": {}}
    fused_loaders = {}
    for lid in layer_ids:
        base = f"layers.{lid}."
        attn = {}
        for t in ("wq_a", "wq_b", "wkv", "wo_b"):
            attn[t] = load_tensor(f"{base}attn.{t}.weight")
        woa = [load_tensor(f"{base}attn.wo_a.{g}.weight") for g in range(args.o_groups)]
        attn["wo_a"] = mx.stack(woa, axis=0)  # [groups, r, d]
        for t in ("q_norm", "kv_norm"):
            attn[t] = load_tensor(f"{base}attn.{t}.weight")
        attn["attn_sink"] = load_tensor(f"{base}attn.attn_sink")
        if args.compress_ratios[lid]:
            attn["compressor"] = (
                load_tensor(f"{base}attn.compressor.wkv.weight"),
                load_tensor(f"{base}attn.compressor.wgate.weight"),
                load_tensor(f"{base}attn.compressor.ape"),
                load_tensor(f"{base}attn.compressor.norm.weight"),
            )
            if args.compress_ratios[lid] == 4:
                attn["indexer"] = (
                    load_tensor(f"{base}attn.indexer.wq_b.weight"),
                    load_tensor(f"{base}attn.indexer.weights_proj.weight"),
                    (
                        load_tensor(f"{base}attn.indexer.compressor.wkv.weight"),
                        load_tensor(f"{base}attn.indexer.compressor.wgate.weight"),
                        load_tensor(f"{base}attn.indexer.compressor.ape"),
                        load_tensor(f"{base}attn.indexer.compressor.norm.weight"),
                    ),
                )
        ffn = {}
        ffn["gate_w"] = load_tensor(f"{base}ffn.gate.weight")
        if lid < args.n_hash_layers:
            ffn["gate_b"] = load_tensor(f"{base}ffn.gate.tid2eid")
        else:
            ffn["gate_b"] = load_tensor(f"{base}ffn.gate.bias")
        # 融合 expert: 層 0/1 は旧命名 (mlp.switch_mlp)
        switch_base = (
            f"{base}mlp.switch_mlp"
            if has(f"{base}mlp.switch_mlp.gate_proj.weight")
            else f"{base}ffn.switch_mlp"
        )
        fused_keys = {
            p: {
                "wq": f"{switch_base}.{name}_proj.weight",
                "s": f"{switch_base}.{name}_proj.scales",
                "b": f"{switch_base}.{name}_proj.biases",
            }
            for p, name in (("gate", "gate"), ("up", "up"), ("down", "down"))
        }
        shard = wm[f"{switch_base}.gate_proj.weight"]
        shard_path = os.path.join(model_path, shard)

        def make_loader(sp=shard_path, fk=fused_keys):
            cache = {}

            def load():
                # shard の mmap 配列をキャッシュし、毎回の fresh ページインを回避
                if not cache:
                    data = mx.load(sp)
                    cache.update(
                        {p: {k: data[v] for k, v in t.items()} for p, t in fk.items()}
                    )
                return cache

            return load

        if streaming:
            ffn["fused"] = None
            fused_loaders[lid] = make_loader()
        else:
            fused_loaders[lid] = None
            ffn["fused"] = make_loader()()
        ffn["shared"] = {
            "w1": load_tensor(f"{base}ffn.shared_experts.w1.weight"),
            "w2": load_tensor(f"{base}ffn.shared_experts.w2.weight"),
            "w3": load_tensor(f"{base}ffn.shared_experts.w3.weight"),
        }
        w["layers"][lid] = {
            "attn": attn,
            "ffn": ffn,
            "attn_norm": load_tensor(f"{base}attn_norm.weight"),
            "ffn_norm": load_tensor(f"{base}ffn_norm.weight"),
            "hc_attn_fn": load_tensor(f"{base}hc_attn_fn"),
            "hc_ffn_fn": load_tensor(f"{base}hc_ffn_fn"),
            "hc_attn_base": load_tensor(f"{base}hc_attn_base"),
            "hc_ffn_base": load_tensor(f"{base}hc_ffn_base"),
            "hc_attn_scale": load_tensor(f"{base}hc_attn_scale"),
            "hc_ffn_scale": load_tensor(f"{base}hc_ffn_scale"),
        }
    for k, key in (
        ("embed", "embed.weight"),
        ("norm", "norm.weight"),
        ("head", "head.weight"),
        ("hc_head_fn", "hc_head_fn"),
        ("hc_head_base", "hc_head_base"),
        ("hc_head_scale", "hc_head_scale"),
    ):
        if has(key):
            w[k] = load_tensor(key)
    return MLXV4Model(
        args,
        w,
        layer_ids=layer_ids,
        use_fused_moe=True,
        fused_loaders=fused_loaders,
        bits=bits,
    )


def torch_to_mx(t):
    return mx.array(t.detach().cpu().float().numpy(), dtype=mx.bfloat16)


def load_v4_streaming(
    model_path,
    args,
    layer_ids=None,
    capacity=None,
    perf=False,
    store_dir=None,
    bits=4,
):
    """部分常駐ストリーミングモデルを構築する。

    非 expert（attn/norm/hc/gate/shared/embed/head）は常駐し、routed expert は
    store の per-expert ファイル（mmap ~13MB）を ResidentCache のバイト予算で
    常駐 + ミス時ストリーミングする。融合テンソルは一切保持しないため、
    メモリは非 expert + 常駐 expert + 予算内に収まる（128GB 機で成立）。

    戻り値: (model, cache, store)
    """
    import json
    import os

    from elfmoon.expert_store import ExpertStore
    from elfmoon.resident_cache import (
        ResidentCache,
        budget_bytes_from_env,
        plan_cache_experts,
    )

    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    wm = json.load(open(idx_path))["weight_map"]

    if layer_ids is None:
        layer_ids = list(range(args.n_layers))

    def load_tensor(key):
        shard = wm[key]
        data = mx.load(os.path.join(model_path, shard))
        return data[key]

    def has(key):
        return key in wm

    store = ExpertStore(store_dir or os.path.join(model_path, "store"))
    w = {"layers": {}}
    for lid in layer_ids:
        base = f"layers.{lid}."
        attn = {}
        for t in ("wq_a", "wq_b", "wkv", "wo_b"):
            attn[t] = load_tensor(f"{base}attn.{t}.weight")
        woa = [load_tensor(f"{base}attn.wo_a.{g}.weight") for g in range(args.o_groups)]
        attn["wo_a"] = mx.stack(woa, axis=0)  # [groups, r, d]
        for t in ("q_norm", "kv_norm"):
            attn[t] = load_tensor(f"{base}attn.{t}.weight")
        attn["attn_sink"] = load_tensor(f"{base}attn.attn_sink")
        if args.compress_ratios[lid]:
            attn["compressor"] = (
                load_tensor(f"{base}attn.compressor.wkv.weight"),
                load_tensor(f"{base}attn.compressor.wgate.weight"),
                load_tensor(f"{base}attn.compressor.ape"),
                load_tensor(f"{base}attn.compressor.norm.weight"),
            )
            if args.compress_ratios[lid] == 4:
                attn["indexer"] = (
                    load_tensor(f"{base}attn.indexer.wq_b.weight"),
                    load_tensor(f"{base}attn.indexer.weights_proj.weight"),
                    (
                        load_tensor(f"{base}attn.indexer.compressor.wkv.weight"),
                        load_tensor(f"{base}attn.indexer.compressor.wgate.weight"),
                        load_tensor(f"{base}attn.indexer.compressor.ape"),
                        load_tensor(f"{base}attn.indexer.compressor.norm.weight"),
                    ),
                )
        ffn = {}
        ffn["gate_w"] = load_tensor(f"{base}ffn.gate.weight")
        if lid < args.n_hash_layers:
            ffn["gate_b"] = load_tensor(f"{base}ffn.gate.tid2eid")
        else:
            ffn["gate_b"] = load_tensor(f"{base}ffn.gate.bias")
        ffn["shared"] = {
            "w1": load_tensor(f"{base}ffn.shared_experts.w1.weight"),
            "w2": load_tensor(f"{base}ffn.shared_experts.w2.weight"),
            "w3": load_tensor(f"{base}ffn.shared_experts.w3.weight"),
        }
        w["layers"][lid] = {
            "attn": attn,
            "ffn": ffn,
            "attn_norm": load_tensor(f"{base}attn_norm.weight"),
            "ffn_norm": load_tensor(f"{base}ffn_norm.weight"),
            "hc_attn_fn": load_tensor(f"{base}hc_attn_fn"),
            "hc_ffn_fn": load_tensor(f"{base}hc_ffn_fn"),
            "hc_attn_base": load_tensor(f"{base}hc_attn_base"),
            "hc_ffn_base": load_tensor(f"{base}hc_ffn_base"),
            "hc_attn_scale": load_tensor(f"{base}hc_attn_scale"),
            "hc_ffn_scale": load_tensor(f"{base}hc_ffn_scale"),
        }
    for k, key in (
        ("embed", "embed.weight"),
        ("norm", "norm.weight"),
        ("head", "head.weight"),
        ("hc_head_fn", "hc_head_fn"),
        ("hc_head_base", "hc_head_base"),
        ("hc_head_scale", "hc_head_scale"),
    ):
        if has(key):
            w[k] = load_tensor(key)

    cache = ResidentCache(max(1, int(capacity) if capacity else 1))

    def _flatten_arrays(o):
        if isinstance(o, mx.array):
            return [o]
        if isinstance(o, dict):
            out = []
            for v in o.values():
                out += _flatten_arrays(v)
            return out
        if isinstance(o, (list, tuple)):
            out = []
            for v in o:
                out += _flatten_arrays(v)
            return out
        return []

    # 非 expert 重みの実メモリを実測して予算から差し引く（遅延評価のため実体化必須）
    mx.eval(_flatten_arrays(w))
    non_expert = mx.get_active_memory()
    budget = budget_bytes_from_env()
    per_expert = store.per_expert_bytes()
    print(
        f"[streaming] 非expert実測 {non_expert / 1024**3:.1f}GB / "
        f"expert {per_expert / 1024**2:.1f}MB / 予算 {budget / 1024**3:.0f}GB"
    )
    if capacity is not None:
        cache.capacity = max(1, int(capacity))
        print(f"[streaming] 明示容量 {cache.capacity} experts")
    elif budget > 0 and per_expert > 0 and non_expert > 1024**3:
        cache.capacity = plan_cache_experts(
            budget,
            non_expert,
            per_expert,
            max_experts=len(layer_ids) * 256,
            headroom=0.75,
        )
        print(
            f"[streaming] 自動容量 {cache.capacity} experts"
            f"（{cache.capacity * per_expert / 1024**3:.1f}GB）"
        )
    else:
        print(f"[streaming] 予算導出不可: 暫定容量 {cache.capacity}")
    model = MLXV4Model(
        args,
        w,
        layer_ids=layer_ids,
        use_fused_moe=False,
        moe_mode="stream",
        cache=cache,
        store=store,
        bits=bits,
    )
    return model, cache, store
