"""DeepSeek-V4 公式カーネル（tilelang）の純 torch リファレンス実装。

公式 inference/kernel.py の tilelang カーネルを、Apple Silicon 上の torch で
動くように忠実に再実装する。QAT（fp8/fp4 アクティベーション量子化）の丸めを
含む。数値は公式とビット完全を狙う（fp8 はネイティブキャスト、fp4 はテーブル丸め）。
"""

from functools import lru_cache

import math
import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP8_MAX_INV = 1.0 / FP8_MAX
FP4_MAX = 6.0
FP4_MAX_INV = 1.0 / FP4_MAX

# e2m1 の値テーブル（公式 convert.py の FP4_TABLE と同一）
_FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def fast_round_scale(amax: torch.Tensor, fp8_max_inv: float) -> torch.Tensor:
    """s = 2^ceil(log2(amax * fp8_max_inv))（e8m0 パワーオブ2スケール）。"""
    return torch.pow(2.0, torch.ceil(torch.log2(amax * fp8_max_inv)))


def _round_fp4(r: torch.Tensor) -> torch.Tensor:
    """FP4(e2m1) への丸め。r は [-6,6]。"""
    a = r.abs()
    idx = (a.unsqueeze(-1) > _FP4_BOUNDS.to(r.device)).sum(-1)
    q = _FP4_VALS.to(r.device)[idx]
    return torch.sign(r) * q


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt=None,
    scale_dtype=torch.float32,
    inplace: bool = False,
):
    """ブロック単位 fp8 量子化（QAT シミュレーション）。

    公式 act_quant_kernel と同じ: s = max(amax,1e-4)*(1/448)（scale_fmt 指定時は
    2 のべき乗に丸め）、y = fp8_round(clamp(x/s,-448,448))*s。inplace=True で
    bf16 に量子化戻し。
    """
    shape = x.shape
    N = x.shape[-1]
    assert N % block_size == 0
    xf = x.float().reshape(-1, N // block_size, block_size)
    amax = torch.clamp_min(xf.abs().amax(dim=-1, keepdim=True), 1e-4)
    s = amax * FP8_MAX_INV
    if scale_fmt is not None:
        s = fast_round_scale(amax, FP8_MAX_INV)
    xq = torch.clamp(xf / s, -FP8_MAX, FP8_MAX)
    xq = xq.to(torch.float8_e4m3fn).to(torch.float32) * s
    y = xq.reshape(shape)
    if inplace:
        x.copy_(y.bfloat16())
        return x
    return y.bfloat16(), s.reshape(*shape[:-1], N // block_size)


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = True):
    """ブロック単位 fp4 量子化（QAT シミュレーション、indexer 用）。

    公式 fp4_quant_kernel と同じ: s=2^ceil(log2(max(amax,6*2^-126)/6))、
    y = fp4_round(clamp(x/s,-6,6))*s。
    """
    shape = x.shape
    N = x.shape[-1]
    assert N % block_size == 0
    xf = x.float().reshape(-1, N // block_size, block_size)
    amax = torch.clamp_min(xf.abs().amax(dim=-1, keepdim=True), FP4_MAX * (2.0**-126))
    s = fast_round_scale(amax, FP4_MAX_INV)
    xq = _round_fp4(torch.clamp(xf / s, -FP4_MAX, FP4_MAX)) * s
    y = xq.reshape(shape)
    if inplace:
        x.copy_(y.bfloat16())
        return x
    return y.bfloat16(), s.reshape(*shape[:-1], N // block_size)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """スパースアテンション（gather + online softmax + attn_sink）。

    公式 sparse_attn_kernel と同じ数学。kv は [b, n, d] で k 兼 v。
    topk_idxs の -1 はパディング（マスク）。出力は bf16。
    """
    b, s, h, d = q.shape
    T = topk_idxs.shape[-1]
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp(min=0)
    # [b,s,T,d]
    kvg = torch.gather(
        kv.unsqueeze(1).expand(b, s, kv.shape[1], d),
        2,
        idx.unsqueeze(-1).expand(-1, -1, -1, d),
    )
    kvg = kvg * valid.unsqueeze(-1)
    scores = torch.einsum("bshd,bstd->bsht", q.float(), kvg.float()) * softmax_scale
    scores = torch.where(valid[:, :, None, :], scores, float("-inf"))
    m = scores.amax(dim=-1, keepdim=True)
    e = torch.exp(scores - m)
    denom = e.sum(dim=-1, keepdim=True) + torch.exp(
        attn_sink.view(1, 1, h, 1).float() - m
    )
    o = torch.einsum("bsht,bstd->bshd", e, kvg.float()) / denom
    return o.bfloat16()


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """HC の pre/post/comb を Sinkhorn 反復で計算（公式 hc_split_sinkhorn_kernel）。"""
    mixes = mixes.float().reshape(-1, (2 + hc_mult) * hc_mult)
    hc = hc_mult
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = mixes[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb.view(-1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform（正規化済み、自己逆変換）。"""
    n = x.shape[-1]
    orig = x.shape
    x = x.float().reshape(-1, n)
    h = 1
    while h < n:
        x = x.reshape(-1, 2, h)
        a, b = x[:, 0], x[:, 1]
        x = torch.cat([a + b, a - b], dim=1)
        h *= 2
    return x.reshape(orig) * (n**-0.5)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """indexer 用 Hadamard 回転（fast_hadamard_transform と同一）。"""
    assert x.dtype == torch.bfloat16
    return hadamard(x).bfloat16()


def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
):
    """YaRN スケーリング付き rotary freq（実数 cos/sin を返す）。"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return (
                dim
                * math.log(max_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min_, max_, dim):
            if min_ == max_:
                max_ += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32) - min_) / (
                max_ - min_
            )
            return torch.clamp(linear_func, 0, 1)

        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)  # [seqlen, dim//2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x, freqs_cos, freqs_sin, inverse=False):
    """実数 rotary。x の末尾 rope_head_dim 次元へ適用。出力 bf16。"""
    rd = freqs_cos.shape[-1] * 2
    dtype = x.dtype
    xf = x.float()
    if x.ndim == 3:
        cos = freqs_cos.unsqueeze(0)  # [1,s,d/2]
        sin = freqs_sin.unsqueeze(0)
    else:
        cos = freqs_cos.unsqueeze(0).unsqueeze(2)
        sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    xr = xf[..., -rd:].reshape(*xf.shape[:-1], -1, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    if inverse:
        y0 = x0 * cos + x1 * sin
        y1 = -x0 * sin + x1 * cos
    else:
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
    out = torch.stack([y0, y1], dim=-1).flatten(-2)
    y = xf.clone()
    y[..., -rd:] = out
    return y.to(dtype)


def rms_norm(x, weight, eps):
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * weight.float()).to(dtype)


@lru_cache(1)
def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [
                torch.arange(start_pos + 1, window_size),
                torch.arange(0, start_pos + 1),
            ],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1),
            (0, window_size - start_pos - 1),
            value=-1,
        )
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size)
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


@lru_cache(2)
def get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
