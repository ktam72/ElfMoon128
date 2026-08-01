# Laguna-S-2.1-MLX-4bit オンメモリ動作（2026-07-22）

機種: Apple M5 Max / 128GB。モデル: Vontra/Laguna-S-2.1-MLX-4bit（62GB, 118B総/8B活性 MoE）。
アーキ: laguna（Poolside独自, LagunaForCausalLM）。256expert top-10 + shared1、48層
（full/sliding-window混在512窓）、per-head softplus attention gating、YARN+partial rotary、
sigmoid ルーター + moe_routed_scaling 2.5。

## 対応方針（当初想定→実際）
- 当初: MLX 実装を自作（~300行, YARN/gating, 数時間）と見積もり。
- 実際: mlx_lm 未マージPR（#1601等）に MLX 実装が存在。advisor 助言で上流確認し発見。
  → PR ブランチ(pierre427/mlx-lm)の mlx_lm/models/laguna.py(526行)を site-packages に配置。
  全依存が現行 mlx_lm 0.31.3 に存在。ElfMoon 側コード変更なし（chat.py 汎用ロード経路）。

## 結果
```
モデル: .../laguna-s-2.1-4bit（type=laguna）
あなた> 日本一高い山は富士山です。
（13 tokens, 69.2 tok/s）
```
- 流暢な日本語・正確・クラッシュなし。62GB オンメモリで 69 tok/s。
- 注: 62GB は 108GiB 上限内でオンメモリ動作。ElfMoon128 のストリーミング機構は不使用
  （＝実質 mlx_lm への一般対応）。ユーザー了承済み。
