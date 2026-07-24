"""プリフィル動的調整の効果測定: ベースライン vs 動的調整版。

使い方:
    python3 -m elfmoon.bench.bench_prefill_dynamic [--model MODEL_NAME]

出力:
    ベースラインと動的調整版のプリフィル速度(tok/s)を比較表示。
"""

import argparse
import os
import sys
import time

os.environ["ELFMOON_KVC"] = "0"

import mlx.core as mx
from mlx_lm import load as _mlx_load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stream_model import (
    FUSED_MIN_TOKENS,
    resolve_model,
    wire_streaming,
    optimal_prefill_step,
)

BASELINE_STEP = 4096

LONG_PROMPT = """System: You are an expert coding assistant.

User: Create a complete Breakout game as a single HTML file with the following specifications:

DESIGN REQUIREMENTS:
- Theme: Cyberpunk / Dark Neon with deep dark gradients (#0f172a to #1e1b4b) and subtle grid patterns
- Blocks: Each row has vibrant neon colors (pink, cyan, purple, gold, emerald green) with rounded corners and glow effects
- Paddle & Ball: Neon blue/cyan gradient with canvas shadowBlur glow effects
- Typography: Google Fonts ('Press Start 2P', 'Inter') for stylish score display
- Effects: Particle explosions on block destruction, screen shake on hit/bounce

GAME SYSTEM:
- Controls: Mouse/touch + arrow keys (left/right)
- Game loop: START, PLAYING, GAME_OVER, GAME_CLEAR states with overlay UI
- Score, high score (localStorage), lives display
- Combo multiplier for consecutive block destructions
- Ball physics: paddle hit position affects bounce angle

TECHNICAL:
- Pure HTML5 Canvas 2D API + Vanilla JS (no external libraries)
- Responsive canvas with aspect ratio preservation
- Web Audio API for sound effects (block break, paddle hit, game over)
- requestAnimationFrame at 60fps
- Well-commented code for beginners

The game should have 8 columns and 6 rows of blocks, with the ball starting on the paddle. Include power-ups, increasing difficulty, and polished visual effects. Make the code complete and ready to run in a browser.

This is a detailed specification that requires implementing:
1. A complete game loop with delta-time-based updates
2. Collision detection between ball and blocks/paddle/walls
3. A particle system for visual effects
4. Screen shake mechanics
5. Audio synthesis with Web Audio API oscillators
6. State management for different game screens
7. Score tracking with combo system
8. Responsive design
9. Touch input support
10. Local storage for high scores

Implementation notes:
- The canvas should render at a fixed internal resolution but scale to fit the viewport
- Block destruction should trigger a particle burst with random velocities
- Screen shake intensity should decay exponentially
- Sounds should be generated procedurally without any audio files
- The combo multiplier should increase with each consecutive block destroyed and reset on missing the ball
- Paddle deflection should use trigonometric calculations based on hit position

Please provide the complete, working implementation."""


def measure_prefill_speed(
    model, prompt_ids, prefill_step, label, n_warmup=2, n_measure=5
):
    """プレフィル速度を測定する。"""
    prompt_cache = make_prompt_cache(model)

    # ウォームアップ
    for _ in range(n_warmup):
        cache2 = prompt_cache
        prompt_cache = make_prompt_cache(model)
        for i in range(0, len(prompt_ids), prefill_step):
            chunk = prompt_ids[i : i + prefill_step]
            model(mx.array([chunk]), cache=cache2)
        mx.eval()

    # 本測定
    times = []
    for _ in range(n_measure):
        pc = make_prompt_cache(model)
        t0 = time.perf_counter()
        for i in range(0, len(prompt_ids), prefill_step):
            chunk = prompt_ids[i : i + prefill_step]
            model(mx.array([chunk]), cache=pc)
        mx.eval()
        dt = time.perf_counter() - t0
        times.append(dt)
        mx.clear_cache()

    avg_dt = sum(times) / len(times)
    tok_s = len(prompt_ids) / avg_dt
    print(f"  {label}: {len(prompt_ids)}tok in {avg_dt:.2f}s = {tok_s:.0f} tok/s")
    return tok_s


def main():
    parser = argparse.ArgumentParser(description="プリフィル動的調整の効果測定")
    parser.add_argument("--model", default=None, help="モデル名")
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=0,
        help="生成するプロンプト長（0=既定の長文プロンプト）",
    )
    args = parser.parse_args()

    print("=== プリフィル動的調整 効果測定 ===")
    print(
        f"ベースライン: FUSED_MIN_TOKENS={FUSED_MIN_TOKENS}, PREFILL_STEP={BASELINE_STEP}"
    )

    # モデルロード
    model_name = args.model or "qwen3.6-35b-mlx"
    print(f"\nモデルロード中: {model_name}")
    mp, sd = resolve_model(model_name)
    t0 = time.time()
    model, tokenizer = _mlx_load(mp)
    print(f"  ロード完了: {time.time() - t0:.1f}s")

    print("  ストリーミング配線中...")
    wire_streaming(
        model,
        capacity=None,
        store_dir=sd,
        model_path=str(mp),
    )
    print("  配線完了")

    # プロンプト準備
    if args.prompt_len > 0:
        print(f"\nプロンプト生成: {args.prompt_len}トークン")
        base = LONG_PROMPT
        prompt_ids = tokenizer.encode(base)
        while len(prompt_ids) < args.prompt_len:
            prompt_ids += prompt_ids[: args.prompt_len - len(prompt_ids)]
        prompt_ids = prompt_ids[: args.prompt_len]
    else:
        prompt_ids = tokenizer.encode(LONG_PROMPT)

    prompt_len = len(prompt_ids)
    print(f"プロンプト長: {prompt_len}トークン")

    # ---- ベースライン測定 ----
    print(f"\n--- ベースライン測定 ---")
    for layer in getattr(model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "dynamic_fused_min_tokens"):
            mlp.dynamic_fused_min_tokens = FUSED_MIN_TOKENS
    baseline_speed = measure_prefill_speed(
        model, prompt_ids, BASELINE_STEP, "ベースライン"
    )

    # ---- チャンク最適化版測定 ----
    optimal_step = optimal_prefill_step(prompt_len)
    print(f"\n--- チャンク最適化測定 ---")
    print(f"  PREFILL_STEP={BASELINE_STEP} → {optimal_step}")
    for layer in getattr(model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "dynamic_fused_min_tokens"):
            mlp.dynamic_fused_min_tokens = FUSED_MIN_TOKENS
    optimal_speed = measure_prefill_speed(
        model, prompt_ids, optimal_step, "チャンク最適化"
    )

    # ---- 結果表示 ----
    change = ((optimal_speed - baseline_speed) / baseline_speed) * 100
    print(f"\n{'=' * 50}")
    print(f"結果サマリー:")
    print(f"  ベースライン:         {baseline_speed:.0f} tok/s")
    print(f"  チャンク最適化:       {optimal_speed:.0f} tok/s")
    print(f"  変化率:               {change:+.1f}%")
    print(f"  baseline_step={BASELINE_STEP} optimal_step={optimal_step}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
