# DeepSeek-V4-Flash-0731 対応の技術的特長

- 作成: 2026-08-01
- 機種: Apple M5 Max / 128GB / 内蔵 NVMe（990Pro_2TB）
- 対象: `deepseek-ai/DeepSeek-V4-Flash-0731`（公式 2026-07-31 公開、167GB / 48 shards）
- 変換済みモデル: `deepseek-v4-flash-0731-mlx`（4bit, per-expert store 146GB）
- 実装: `elfmoon/convert_v4.py` / `elfmoon/v4/mlx_v4.py` / `elfmoon/chat.py`（v4 分岐）
- 関連: `deepseek-v4-flash-0731-support.md`（設計）/ `partial-residency-streaming.md` / `attention-int8.md`（実測）

## 1. 対応の全体像

Apple M5 Max（128GB 統一メモリ）の**単一マシン・ローカル実行**で、公式リリースの
DeepSeek-V4-Flash-0731（base 推論）を**品質維持したまま実用会話速度（実測 4.1-4.9 t/s）**
で動作させることを達成した。対応は 5 つの技術レイヤーで構成される。

```
公式チェックポイント (fp8/fp4, 167GB)
  │ ① 実データ型の解明（safetensors ヘッダ一次計測）
  ▼
変換パイプライン (convert_v4.py)
  │ ② fp8/fp4 → bf16 デコード + expert のみ MLX 4bit 再量子化
  │    融合 shard + per-expert store の二重出力（~160GB）
  ▼
推論エンジン (mlx_v4.py)
  │ ③ ストリーミング部分常駐 MoE（mmap 遅延 + LRU キャッシュ）
  │ ④ attention 全経路 int8 量子化（decode 帯域半減）
  ▼
対話フロント (chat.py v4 分岐)
  └ ⑤ 公式チャットエンコード + mode collapse 対策（実用化）
```

## 2. 技術的特長①: チェックポイントの実態解明（一次情報）

safetensors ヘッダを HTTP 範囲取得で直接計測し、公式重みの実データ型を確定した。

| テンソル | 実データ型 | 備考 |
|---|---|---|
| `layers.N.ffn.experts.E.w1/w3.weight` | **I8（FP4 パック e2m1×2/バイト）** | scale は K 方向 32 要素毎の e8m0 |
| `layers.N.attn.*` / 非 expert | **F8_E4M3** | scale は 128×128 ブロック毎 e8m0 |
| `hc_*` / `attn_sink` | F32 | Hyper-Connections は fp32 維持 |
| `ffn.gate.tid2eid` | I64 | 先頭 3 層の hash ルーティング表 |

要点:
- **MLX は fp8 非対応**のため、全テンソルを bf16 にデコードし、expert のみ
  MLX 4bit（group64）に再量子化する方針を採った（量子化の二段化）。
- 重み命名は**フラット**（`layers.N.hc_attn_fn` 等）で、上流 mlx_lm PR #1201 と一致。
  前回（2026-07-22）「対応不可」とした命名不一致は、mlx-community プレビュー版の
  独自変換によるもので、公式チェックポイントには当てはまらないことを確認して解消。
- モデル仕様: 43 層 / 256 routed experts / top-6 / shared 1 / swiglu_limit=10.0、
  MLA 型 attention（head_dim 512, rope 64, window 128）+ 圧縮アテンション
  （compress_ratio 4/128 交互、indexer top-512）、Hyper-Connections（hc_mult=4）。

## 3. 技術的特長②: 自己完結変換パイプライン

transformers 4.57.6 は `deepseek_v4` 非対応のため、**safetensors を直接読む独自変換**
を実装した（`elfmoon/convert_v4.py`）。

- **FP4 デコード**: パック 2 ニブル → e2m1 値表 → `scale.repeat_interleave(32)` で復元。
- **FP8 デコード**: e4m3 → `scale.repeat_interleave(128)` で復元（128×128 ブロック）。
- **二重出力**: 既存 ElfMoon128 store 互換の融合 shard（`switch_mlp.{gate,up,down}_proj`）
  + per-expert store ファイル（`l{l}_e{e}.safetensors`）。`integrate.py` の事後実行不要。
- **数値検証**: 層 0 で fp4/fp8 デコードの真値復元を確認。変換所要 ~14 分（1 層 17-336 秒）。
- 注意点: `mx.quantize` は CPU/GPU で非決定性があるため、本番変換は GPU 実行で統一。
- 既存 index のマージ漏れバグ（部分変換時に層 0/1 のキーが消失）を修正済み。

## 4. 技術的特長③: ストリーミング部分常駐 MoE（メモリ安全性）

137GB 級モデルを 128GB 機で動かすための核。**メモリ安全の粒度は per-expert store
（~13.5MB/個）のみ**であることを実験で確定した上で設計した。

```
mx.load は mmap 遅延ロード → 配列をカーネルで消費するとソース全体が実メモリに具体化
  mx.take(w, idx)  → RSS 40→1087MB（ソース 1.07GB 全体）
  w[0:1] スライス → RSS 40→1067MB（スライスでも全体）
→ 融合テンソル 3.2GB/層 × 43 層 = 137GB は回避不能 → per-expert 粒度のみ安全
```

実装（`mlx_v4.py`）:
- `MLXStreamingMoE`: gate / shared expert は常駐、routed expert は
  `ResidentCache`（バイト予算 LRU）+ `ExpertStore`。
  decode は選択 top-6 を stack → `act_quant` + `quantized_matmul`×3（swiglu_limit 込み）。
- `load_v4_streaming()`: 非 expert（attn/norm/hc/gate/shared/embed/head）を常駐し、
  非 expert 実メモリ実測 → 予算から expert 常駐容量を自動導出（`plan_cache_experts`）。
- 実測: 常駐 13.8GB（非 expert）+ expert 13.5MB/個。自動容量 = 5073 experts（66.9GB）。

## 5. 技術的特長④: attention 全経路 int8 量子化（decode 帯域半減）

decode は**重み読込メモリ帯域律速**（attention bf16 ~7GB + expert 4bit ~7GB = ~14GB/token）。
attention 重みを bf16→int8（`mx.quantize(bits=8, group_size=64)` +
`quantized_matmul(mode="affine")`）に落として帯域を半減した。

- 適用対象: attention 全経路の Linear（wq_a/wq_b/wkv/wo_b 4 主投影 + compressor wkv/wgate
  + indexer wq_b/weights_proj）。量子化後は bf16 原重みを破棄（メモリも削減）。
- 制御: `ELFMOON_ATTN_Q8`（デフォルト "1"、A/B 用に "0" で bf16）。
- **int8（256 レベル）は元の fp8 e4m3 より細かく**、fp8-QAT の頑健性がそのまま活きる。
- 本番 warm A/B（cap2000, long, 480tok）: **4.92 t/s / 39.7GB / hit 91.5%**
  vs ベースライン 4.02 t/s / 42.4GB（速度 +22%、メモリ -2.7GB）。
- 数値パリティ: prefill/decode の top1 全一致、greedy トークン列完全一致。
  品質（qa/math）も bf16 と完全一致（エビデンス: `attention-int8.md`）。

## 6. 技術的特長⑤: 帯域律速の分析（実用上限の確定）

速度目標を「計測で確定する」方針の下、本番パス warm A/B で律速要因を特定した。

- **cap2000（27GB 常駐）で速度が頭打ち（~4.1 t/s）**。ヒット率 91%→95% でも速度不変
  = per-expert 経路のカーネル/オーバーヘッド律速であり、帯域（ヒット率）律速ではない。
- バリア除去（async）は 200ms/tok（5.0 t/s）が上限。ただし正しい expert 選択にはバリアが
  必要で、品質維持のままのバリア除去は不可。
- **結論: decode は重み読込メモリ帯域律速。4bit 部分常駐設計では品質維持での実用上限は
  ~4-5 t/s。** 設計時点の見込み（5-8 t/s）を実測で修正した。

## 7. 技術的特長⑥: 対話統合と mode collapse 対策（実用化）

`chat.py --model deepseek-v4-flash-0731-mlx` を実会話可能にした。要点は以下の 3 つ。

1. **公式チャットエンコードの解決**: `encoding_dsv4` はモデルディレクトリ直下に無く、
   兄弟モデル `deepseek-v4-flash-0731/encoding` にある。`-mlx` サフィックスを除いた
   パスへ sys.path を解決して使用。
2. **生成ループの刷新**: `model.forward(x, start_pos)` の prefill→decode +
   `_sample(TEMP=0.4, top_p=0.9)`。旧 `model_v4.DeepseekV4Model`（旧命名前提）は
   フラット命名 store と非互換のため廃用。
3. **mode collapse 対策**（上位モデル固有の問題）:
   - **根本原因を特定**: 英語 coding-assistant SYSTEM プロンプトが崩壊の引き金。
     repetition penalty 強化では防げないことを A/B で確認。日本語ニュートラル SYSTEM
     （「あなたは優秀なAIアシスタントです。…」）への置換で解消。
   - repetition penalty デフォルト 1.5（`ELFMOON_V4_RP` で変更可）。
   - 同一トークン連続 5 個で崩壊と判定して停止（テール切り落とし）。
   - 早発崩壊（<48 tokens）時は行クリアして rp=1.6 で 1 回だけ再生成。
   - デフォルトは思考なし（chat）モード。`ELFMOON_V4_THINK=1` で thinking に opt-in。

## 8. 実測パフォーマンス総括

| 構成 | 速度 | 常駐メモリ | ヒット率 |
|---|---|---|---|
| 4bit 部分常駐 cap2000 + attention bf16 | 4.02 t/s | 42.4GB | 91.4% |
| **同 + attention 全経路 int8（デフォルト）** | **4.92 t/s** | **39.7GB** | 91.5% |
| 自動容量 5073 experts + int8 | ~4.9 t/s | ~66.9GB+ | 95%+ |
| 参考: 2bit 全常駐 | 5.5 t/s | 102GB | ❌ 品質崩壊 |

- 実会話（pty 検証, 自動容量）: 自己紹介 42 tokens / 4.3 t/s / hit 73%、
  質問応答 69 tokens / 4.3 t/s / hit 86% — 2 ターンとも coherent・崩壊なし。
- ロードはストリーミングで ~1-4 秒。メモリは全ケース 128GB 内、swap なし。
- **2bit 全常駐は速いが品質崩壊**（102GB）のため不採用。4bit で品質を維持した。

## 9. 他モデルとの比較（ElfMoon128 全体の位置づけ）

| モデル | 方式 | 実測速度 |
|---|---|---|
| Qwen3-235B | streaming 全常駐 | 11.0 t/s |
| GLM-4.7 | streaming 部分常駐 | 6.7 t/s |
| Kimi K3 | streaming 部分常駐 | 1.1 t/s |
| **DeepSeek-V4-Flash-0731（本対応）** | **streaming 部分常駐 + int8** | **4.1-4.9 t/s** |

V4 は per-expert 経路のカーネル/オーバーヘッド律速で ~4-5 t/s に留まるが、
公式の大規模 MoE をローカル単機で品質維持しつつ実用圏の速度で動かす点が最大の価値。

## 10. 制約と残課題

- **実用上限 ~4-5 t/s**（帯域律速）。10 t/s は 4bit では物理的に非現実的。
- per-expert 経路のカーネル最適化（Python dispatch 削減 / 常駐 expert の gather_qmm
  バッチ化）で 5-8 t/s を狙える余地は残る。
- 上位モデルの mode collapse は SYSTEM 置換で実用圏まで抑制したが、完全には消えていない
  （崩壊ガード + リトライで防御）。下位モデル（品質 A/B）との差分確認が望ましい。
- MTP / DSpark（訓練・投機用）は base 推論のスコープ外。

## 11. 参考資料（一次情報源）

- モデル: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
  - config.json / model.safetensors.index.json（データ型・構造の実測）
  - inference/model.py（参照実装）/ encoding/encoding_dsv4.py（チャットエンコード）
- mlx_lm PR #1201（未マージ）: https://github.com/ml-explore/mlx-lm/pull/1201
- 実装: `elfmoon/convert_v4.py` / `elfmoon/v4/mlx_v4.py` / `elfmoon/v4/bench_streaming.py` /
  `elfmoon/v4/verify_streaming.py` / `elfmoon/chat.py`
- 関連エビデンス: `deepseek-v4-flash-0731-support.md` / `deepseek-v4-unsupported.md` /
  `partial-residency-streaming.md` / `attention-int8.md`
