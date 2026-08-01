#!/usr/bin/env python3
"""DeepSeek-V4-Flash-0731 非ストリーミング生成（フェーズ 3 コヒーネンス検証）。

変換済み mmap safetensors からフルモデルを構築し、チャットプロンプトを
encoding_dsv4.py でエンコードして生成する。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import mlx.core as mx

sys.path.insert(0, "/Users/ktam/Documents/apps/ElfMoon128")

MLX = "/Volumes/990Pro_2TB/elfmoon128/models/deepseek-v4-flash-0731-mlx"
ENC = "/Volumes/990Pro_2TB/elfmoon128/models/deepseek-v4-flash-0731/encoding"


def mx_to_np(a):
    if a.dtype == mx.int64:
        return np.asarray(a, dtype=np.int64)
    return np.asarray(a.astype(mx.float32), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="こんにちは、今日の天気を教えてください。")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MLX)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--no-streaming", action="store_true")
    args = ap.parse_args()

    from elfmoon.v4.ref_model import build_args
    from elfmoon.v4.mlx_v4 import load_v4_fused

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    margs = build_args(cfg)
    np.random.seed(args.seed)

    print("モデルロード中...", flush=True)
    t0 = time.time()
    model = load_v4_fused(
        args.model, margs, streaming=not args.no_streaming, bits=args.bits
    )
    print(f"ロード完了 ({time.time() - t0:.1f}s)", flush=True)

    # チャットエンコード
    sys.path.insert(0, ENC)
    import encoding_dsv4 as enc

    messages = [{"role": "user", "content": args.prompt}]
    prompt = enc.encode_messages(messages, thinking_mode="chat")
    print(f"\n=== prompt ===\n{prompt}\n==============", flush=True)

    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(args.model)

    # BOS は encode_messages が先頭に付与済み
    ids = tok.encode(prompt)
    print(f"prompt tokens: {len(ids)}")

    # 生成ループ: まず全プロンプトを prefill（start_pos=0）、以降は 1 token decode
    all_ids = list(ids)
    prompt_len = len(ids)
    print("\n=== prefill ===", flush=True)
    t0 = time.time()
    input_ids = mx.array(np.array([all_ids], dtype=np.int32), mx.int32)
    logits = model.forward(input_ids, start_pos=0)
    next_tok = int(mx.argmax(logits[0, -1]).item())
    all_ids.append(next_tok)
    n_tokens = 1
    print(f"prefill 完了 ({time.time() - t0:.1f}s), 初トークン: {next_tok}")

    print("\n=== 生成 ===", flush=True)
    while n_tokens < args.max_tokens:
        cur = mx.array(np.array([[all_ids[-1]]], dtype=np.int32), mx.int32)
        start_pos = len(all_ids) - 1
        logits = model.forward(cur, start_pos=start_pos)
        next_tok = int(mx.argmax(logits[0, -1]).item())
        all_ids.append(next_tok)
        n_tokens += 1
        if next_tok == tok.eos_token_id:
            print("\n[EOS]", flush=True)
            break
        if n_tokens % 10 == 0:
            print(f"  ... {n_tokens} tokens", flush=True)
    dt = time.time() - t0
    print(f"生成 {n_tokens} tokens in {dt:.1f}s = {n_tokens / dt:.2f} tok/s")

    text = tok.decode(all_ids[prompt_len:])
    print(f"\n=== 応答 ===\n{text}\n==========")
    print(f"\n=== 全出力 ===\n{tok.decode(all_ids)}\n==========")


if __name__ == "__main__":
    main()
