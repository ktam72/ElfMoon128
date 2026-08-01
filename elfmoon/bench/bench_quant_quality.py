"""コードブック量子化 vs uniform 量子化の品質比較ベンチマーク。

実モデルの expert 重みを dequantize → 各方式で再量子化 → fp32 参照と MSE 比較。
"""

import argparse
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from expert_store import BITS, GROUP, ExpertStore
from quantize import (
    dequantize_codebook,
    dequantize_uniform,
    max_err,
    mse,
    quantize_codebook,
    snr_db,
    uniform_to_fp32,
)


def load_single_expert(store, layer, expert):
    """Load one expert's weights and dequantize to fp32."""
    w = store.load(layer, expert)
    result = {}
    for name in ("gate", "up", "down"):
        wq = w[f"{name}.wq"]
        s = w[f"{name}.s"]
        b = w.get(f"{name}.b")
        result[name] = uniform_to_fp32(wq, s, b, group_size=GROUP, bits=BITS)
    return result


def eval_method(label, quant_fn, w_ref_dict, **kwargs):
    """Apply quant_fn to each matrix in w_ref_dict, compute metrics."""
    print(f"\n  --- {label} ---")
    total_mse = 0.0
    total_snr = 0.0
    n = 0
    for name in ("gate", "up", "down"):
        ref_np = w_ref_dict[name]
        ref = mx.array(ref_np)
        t0 = time.perf_counter()
        q = quant_fn(ref, **kwargs)
        dt = time.perf_counter() - t0
        if "codebook" in q:
            recon = dequantize_codebook(q)
        else:
            recon = dequantize_uniform(q)
        err = mse(ref_np, recon)
        snr = snr_db(ref_np, recon)
        me = max_err(ref_np, recon)
        orig_size = ref.size * 32  # bits in fp32
        if "codebook" in q:
            idx_np = np.asarray(q["indices"])
            cb_np = np.asarray(q["codebook"])
            ov_np = np.asarray(q["outlier_values"])
            outlier_count = int(np.sum(np.asarray(q["outlier_mask"])))
            stored_bits = (
                idx_np.size * 8 + cb_np.size * 16 + outlier_count * ref.shape[1] * 16
            )
        else:
            wq_np = np.asarray(q["wq"])
            s_np = np.asarray(q["s"])
            stored_bits = wq_np.size * 8 + s_np.size * 16
            if q.get("b") is not None:
                stored_bits += np.asarray(q["b"]).size * 16
            outlier_count = 0
        ratio = stored_bits / orig_size
        total_mse += err
        total_snr += snr
        n += 1
        print(
            f"    {name:6s}: MSE={err:.2e}  SNR={snr:.1f}dB  "
            f"max_err={me:.4f}  bpw={ratio * 32:.2f}  "
            f"({dt * 1000:.0f}ms) outlier={outlier_count}/{ref.shape[0]}"
        )
    avg_mse = total_mse / n
    avg_snr = total_snr / n
    print(f"    {'avg':6s}: MSE={avg_mse:.2e}  SNR={avg_snr:.1f}dB")
    return avg_mse, avg_snr


def main():
    parser = argparse.ArgumentParser(description="量子化品質比較")
    parser.add_argument("--model", default=None, help="モデル名 (MODELS_ROOT 配下)")
    parser.add_argument("--layer", type=int, default=4, help="測定する層番号")
    parser.add_argument("--expert", type=int, default=0, help="測定する expert 番号")
    parser.add_argument(
        "--n-layers",
        type=int,
        default=0,
        help="指定時: 全層の最初の expert を走査（上書き）",
    )
    args = parser.parse_args()

    from stream_model import MODELS_ROOT, resolve_model

    if args.model:
        mp, sd = resolve_model(args.model)
    else:
        name = os.environ.get("ELFMOON_MODEL", None)
        mp, sd = resolve_model(name)

    store = ExpertStore(sd)

    print(f"モデル: {mp}")
    print(f"store : {sd}")
    print(f"既定  : uniform {BITS}-bit, group_size={GROUP}")
    print()

    if args.n_layers > 0:
        # 全層走査
        print(f"全層走査: layer=0..{args.n_layers - 1}, expert={args.expert}")
        print()

        def _uq(gs, bits):
            return lambda w: (
                {"wq": q[0], "s": q[1], "b": q[2], "group_size": gs, "bits": bits}
                if (q := list(mx.quantize(w, group_size=gs, bits=bits)))
                else None
            )

        methods = [
            ("uniform 2-bit gs=64", _uq(64, 2)),
            (
                "cb 2-bit sv=8 outlier=0%",
                lambda w: quantize_codebook(
                    w, bits=2, sub_vector_size=8, outlier_fraction=0.0
                ),
            ),
            (
                "cb 2-bit sv=8 outlier=1%",
                lambda w: quantize_codebook(
                    w, bits=2, sub_vector_size=8, outlier_fraction=0.01
                ),
            ),
            (
                "cb 2-bit sv=8 outlier=2%",
                lambda w: quantize_codebook(
                    w, bits=2, sub_vector_size=8, outlier_fraction=0.02
                ),
            ),
        ]

        header = ["layer"]
        for label, _ in methods:
            header.extend([f"{label}_mse", f"{label}_snr"])
        print("\t".join(header))

        for l in range(args.n_layers):
            w_ref = load_single_expert(store, l, args.expert)
            row = [str(l)]
            for label, fn in methods:
                mse_vals = []
                snr_vals = []
                for name in ("gate", "up", "down"):
                    ref = w_ref[name]
                    q = fn(ref)
                    if "codebook" in q:
                        recon = dequantize_codebook(q)
                    else:
                        recon = dequantize_uniform(q)
                    mse_vals.append(mse(ref, recon))
                    snr_vals.append(snr_db(ref, recon))
                row.append(f"{sum(mse_vals) / 3:.2e}")
                row.append(f"{sum(snr_vals) / 3:.1f}")
            print("\t".join(row))
            sys.stdout.flush()

    else:
        # 単層詳細
        print(f"単層測定: layer={args.layer}, expert={args.expert}")
        w_ref = load_single_expert(store, args.layer, args.expert)
        for name in ("gate", "up", "down"):
            ref = w_ref[name]
            print(
                f"  {name}: shape={ref.shape}, range=[{np.min(ref):.4f}, {np.max(ref):.4f}]"
            )
        print()

        def _uq(gs, bits):
            return lambda w: (
                {"wq": q[0], "s": q[1], "b": q[2], "group_size": gs, "bits": bits}
                if (q := list(mx.quantize(w, group_size=gs, bits=bits)))
                else None
            )

        methods = [
            (f"uniform {BITS}-bit (baseline)", _uq(GROUP, BITS)),
            ("uniform 2-bit gs=64", _uq(64, 2)),
            ("uniform 2-bit gs=32", _uq(32, 2)),
            (
                "cb 2-bit sv=8 outlier=0%",
                lambda w: quantize_codebook(
                    w, bits=2, sub_vector_size=8, outlier_fraction=0.0
                ),
            ),
            (
                "cb 2-bit sv=8 outlier=0.5%",
                lambda w: quantize_codebook(
                    w, bits=2, sub_vector_size=8, outlier_fraction=0.005
                ),
            ),
            (
                "cb 2-bit sv=8 outlier=1%",
                lambda w: quantize_codebook(
                    w, bits=2, sub_vector_size=8, outlier_fraction=0.01
                ),
            ),
            (
                "cb 2-bit sv=8 outlier=2%",
                lambda w: quantize_codebook(
                    w, bits=2, sub_vector_size=8, outlier_fraction=0.02
                ),
            ),
            (
                "cb 2-bit sv=16 outlier=1%",
                lambda w: quantize_codebook(
                    w, bits=2, sub_vector_size=16, outlier_fraction=0.01
                ),
            ),
        ]

        results = []
        for label, fn in methods:
            avg_mse, avg_snr = eval_method(label, fn, w_ref)
            results.append((label, avg_mse, avg_snr))
            mx.clear_cache()

        print(f"\n  {'=' * 55}")
        print(f"  {'方式':30s} {'MSE':>12s} {'SNR(dB)':>10s}")
        print(f"  {'-' * 55}")
        for label, m, s in results:
            print(f"  {label:30s} {m:12.2e} {s:10.1f}")
        print(f"  {'=' * 55}")


if __name__ == "__main__":
    main()
