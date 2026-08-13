"""decode 速度ベンチマーク（ElfMoonCoder bench_decode 相当・ElfMoon128 用）。

think / no-think 両モードで、同一モデルの prefill / decode 速度を測る。
修正（--no-think 反映）が生成速度に影響しないことを確認するのが目的。

usage:
  python3 bench/bench_decode.py [モデル名] [コンテキスト長]
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # elfmoon/
sys.path.insert(0, str(ROOT))
os.environ["ELFMOON_KVC"] = "0"  # 毎回フル prefill（条件を揃える）
os.environ["ELFMOON_KVC_LOG"] = "0"

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from lookup_gen import stream_generate_lookup  # noqa: E402

GEN_TOKENS = 200


def build_prompt_text(tok, target_tokens, thinking):
    """目標トークン数に近いプロンプトを作る（コード反復ベース）。"""
    unit = "def handler_{i}(req, ctx):\n    return {{'id': {i}, 'ok': True}}\n"
    text, n = "", 0
    while True:
        text += unit.format(i=n)
        n += 1
        if n % 50 == 0:
            msgs = [{"role": "user", "content": text + "\n説明してください"}]
            pt = tok.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=thinking,
            )
            if len(tok.encode(pt)) >= target_tokens:
                return tok.encode(pt)


def measure(model, tok, prompt_ids, thinking, label):
    sampler = make_sampler(temp=0.0, top_p=1.0, min_p=0.0)
    # warmup
    pc = make_prompt_cache(model)
    for r in stream_generate_lookup(
        model,
        tok,
        prompt_ids,
        max_tokens=8,
        sampler=sampler,
        prompt_cache=pc,
        prefill_step_size=4096,
        enable_lookup=False,
    ):
        pass
    # 計測
    pc = make_prompt_cache(model)
    t0 = time.perf_counter()
    pf_t0 = time.perf_counter()
    n_out = 0
    last_tps = 0.0
    for r in stream_generate_lookup(
        model,
        tok,
        prompt_ids,
        max_tokens=GEN_TOKENS,
        sampler=sampler,
        prompt_cache=pc,
        prefill_step_size=4096,
        enable_lookup=False,
    ):
        n_out = r.generation_tokens
        last_tps = r.generation_tps
    total = time.perf_counter() - t0
    print(
        f"  {label:22} prompt {len(prompt_ids)} tok  "
        f"decode {n_out} tok / {total:.2f}s = {last_tps:.1f} t/s"
    )
    return last_tps


def main():
    from stream_model import MODELS_ROOT

    name = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
    )
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    model_path = os.path.join(MODELS_ROOT, name)
    model, tok = load(model_path, lazy=True)

    print(f"モデル: {name} / 生成 {GEN_TOKENS} トークン")
    think_ids = build_prompt_text(tok, ctx, thinking=True)
    nothink_ids = build_prompt_text(tok, ctx, thinking=False)
    print(f"think プロンプト {len(think_ids)} tok / no-think {len(nothink_ids)} tok\n")

    print("[think モード]")
    t_think = measure(model, tok, think_ids, True, "think")
    print("[no-think モード]")
    t_nothink = measure(model, tok, nothink_ids, False, "no-think")
    print(
        f"\ndecode 速度: think {t_think:.1f} t/s → no-think {t_nothink:.1f} t/s "
        f"({t_nothink / t_think:.2f}x)"
    )


if __name__ == "__main__":
    main()
