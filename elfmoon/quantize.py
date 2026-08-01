"""コードブック量子化（k-means Vector Quantization）実験モジュール。

opacc1ty 流の 2-bit コードブック量子化を再現し、uniform 量子化との
品質（MSE）比較を行うための実験用。本番推論パスには非接続。
"""

import math

import mlx.core as mx
import numpy as np


def kmeans(vectors, n_clusters=4, n_iter=20, seed=0):
    """Simple k-means for codebook learning.

    Args:
        vectors: (N, D) float32 array
        n_clusters: number of codebook entries
        n_iter: number of iterations
        seed: random seed

    Returns:
        centroids: (n_clusters, D) float32
        labels: (N,) int32 cluster assignments
    """
    N, D = vectors.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, n_clusters, replace=False)
    centroids = np.asarray(vectors[idx], dtype=np.float32)

    labels = np.zeros(N, dtype=np.int32)
    for _ in range(n_iter):
        dists = np.zeros((N, n_clusters), dtype=np.float32)
        for c in range(n_clusters):
            diff = np.asarray(vectors, dtype=np.float32) - centroids[c]
            dists[:, c] = np.sum(diff * diff, axis=1)
        labels = np.argmin(dists, axis=1).astype(np.int32)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(n_clusters, dtype=np.int32)
        np.add.at(counts, labels, 1)
        for c in range(n_clusters):
            if counts[c] > 0:
                mask = labels == c
                new_centroids[c] = np.mean(vectors[mask], axis=0)
            else:
                new_centroids[c] = centroids[c]
        centroids = new_centroids

    return centroids, labels


def quantize_codebook(w, bits=2, sub_vector_size=8, outlier_fraction=0.01, n_iter=20):
    """Codebook quantization of a weight matrix.

    Args:
        w: (M, K) float32 weight matrix
        bits: index bits per sub-vector (2 -> 4 codebook entries)
        sub_vector_size: sub-vector dimensionality
        outlier_fraction: fraction of output rows to preserve at full precision
        n_iter: k-means iterations

    Returns:
        dict with keys:
            indices: (M, ceil(K/sub_vector_size)) uint8, 4 indices packed per byte
            codebook: (M, 2**bits, sub_vector_size) float16
            outlier_mask: (M,) bool
            outlier_values: (M, K) float16 (0 for non-outlier rows)
    """
    w = np.asarray(w, dtype=np.float32)
    M, K = w.shape
    n_clusters = 1 << bits
    n_subvec = math.ceil(K / sub_vector_size)

    if outlier_fraction > 0:
        n_outlier = max(1, int(M * outlier_fraction))
        row_variance = np.var(np.asarray(w, dtype=np.float32), axis=1)
        outlier_idx = np.argsort(row_variance)[-n_outlier:]
        outlier_mask = np.zeros(M, dtype=bool)
        outlier_mask[outlier_idx] = True
    else:
        outlier_mask = np.zeros(M, dtype=bool)
        n_outlier = 0

    n_bytes = math.ceil(n_subvec / (8 // bits))
    indices = np.zeros((M, n_bytes), dtype=np.uint8)
    codebook = np.zeros((M, n_clusters, sub_vector_size), dtype=np.float16)

    for row in range(M):
        row_data = np.asarray(w[row], dtype=np.float32)
        if outlier_mask[row]:
            continue

        sub_vectors = []
        for sv in range(n_subvec):
            start = sv * sub_vector_size
            end = min(start + sub_vector_size, K)
            sv_data = row_data[start:end]
            if len(sv_data) < sub_vector_size:
                pad = np.zeros(sub_vector_size - len(sv_data), dtype=np.float32)
                sv_data = np.concatenate([sv_data, pad])
            sub_vectors.append(sv_data)
        sub_vectors = np.array(sub_vectors, dtype=np.float32)

        centroids, labels = kmeans(sub_vectors, n_clusters=n_clusters, n_iter=n_iter)

        for sv in range(n_subvec):
            idx_val = int(labels[sv])
            if idx_val >= (1 << bits):
                idx_val = (1 << bits) - 1
            byte_pos = sv // 4
            shift = (sv % 4) * bits
            current = int(indices[row, byte_pos])
            cleared = current & (0xFF ^ (0x3 << shift))
            indices[row, byte_pos] = np.uint8(cleared | (idx_val << shift))

        codebook[row] = centroids.astype(np.float16)

    outlier_values = np.zeros((M, K), dtype=np.float16)
    if n_outlier > 0:
        w_np = np.asarray(w, dtype=np.float32)
        outlier_values[outlier_mask] = w_np[outlier_mask].astype(np.float16)

    return {
        "indices": mx.array(indices),
        "codebook": mx.array(codebook),
        "outlier_mask": mx.array(outlier_mask),
        "outlier_values": mx.array(outlier_values),
        "sub_vector_size": sub_vector_size,
    }


def dequantize_codebook(q):
    """Reconstruct fp32 weight matrix from codebook format.

    Args:
        q: dict from quantize_codebook()

    Returns:
        (M, K) float32 reconstructed weight matrix
    """
    indices_np = np.asarray(q["indices"])
    codebook_np = np.asarray(q["codebook"])
    outlier_mask_np = np.asarray(q["outlier_mask"])
    outlier_values_np = np.asarray(q["outlier_values"])
    sub_vector_size = q.get("sub_vector_size", 8)

    M, K_actual = outlier_values_np.shape
    n_subvec = math.ceil(K_actual / sub_vector_size)
    bits = 2

    recon = np.zeros((M, K_actual), dtype=np.float32)

    for row in range(M):
        if outlier_mask_np[row]:
            recon[row] = outlier_values_np[row].astype(np.float32)
            continue
        for sv in range(n_subvec):
            byte_pos = sv // (8 // bits)
            shift = (sv % (8 // bits)) * bits
            idx_val = (indices_np[row, byte_pos] >> shift) & ((1 << bits) - 1)
            start = sv * sub_vector_size
            end = min(start + sub_vector_size, K_actual)
            sv_vals = codebook_np[row, idx_val].astype(np.float32)
            recon[row, start:end] = sv_vals[: end - start]

    return recon


def quantize_uniform(w, bits=4, group_size=64):
    """Uniform quantization via MLX.

    Returns:
        dict with wq, s, b, group_size, bits
    """
    wq, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    return {"wq": wq, "s": s, "b": b, "group_size": group_size, "bits": bits}


def dequantize_uniform(q):
    """Dequantize uniform quantized weights back to fp32."""
    gs = q.get("group_size", 64)
    bits = q.get("bits", 4)
    dq = mx.dequantize(q["wq"], q["s"], q.get("b"), group_size=gs, bits=bits).astype(
        mx.float32
    )
    mx.eval(dq)
    return np.asarray(dq)


def uniform_to_fp32(wq, s, b, group_size=64, bits=4):
    """Dequantize uniform MLX weights to numpy float32."""
    dq = mx.dequantize(wq, s, b, group_size=group_size, bits=bits).astype(mx.float32)
    mx.eval(dq)
    return np.asarray(dq)


def mse(a, b):
    """Mean squared error between two arrays."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(diff * diff))


def max_err(a, b):
    """Max absolute error."""
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return float(np.max(diff))


def snr_db(a, b):
    """Signal-to-noise ratio in dB."""
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    noise = a_np - b_np
    signal_power = np.mean(a_np * a_np)
    noise_power = np.mean(noise * noise)
    if noise_power < 1e-30:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)
