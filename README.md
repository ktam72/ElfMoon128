# ElfMoon128🌙

> 128GB RAM の Apple Silicon Mac 上で、**128GB オンメモリでは載り切らない巨大 MoE**（GPU ワーキングセット上限 108GiB 超）を Streaming MoE で実用速度で動かす MLX 推論エンジン

[ElfMoon4](../ElfMoon4/README.md) のフォーク。ElfMoon4 が「24GB で 80B」を狙うのに対し、ElfMoon128 は **128GB でも載り切らないモデル**（Qwen3-235B / GLM-4.7 等、120〜185GB 級）を動かすことをコンセプトとする。

基本アーキテクチャ（デュアルモード推論、ストリーミング MoE、`store/` 分解、OpenAI 互換 API / 対話 CLI）は ElfMoon4 と共通。**本 README は ElfMoon128 固有の差分のみを記載し、共通事項は [ElfMoon4 の README](../ElfMoon4/README.md) を参照する。**

>⚠️ ElfMoon4 で構築したモデルディレクトリ（元モデル + 分解済み `store/`）は ElfMoon128 でそのまま動作可能（store 形式は完全互換・上位互換）

---

## ElfMoon128 の主な変更点（vs ElfMoon4）

| 変更点 | 内容 | 参照 |
|---|---|---|
| **常駐容量の自動導出** | 固定 6144 スロット → メモリ予算から expert 数を自動算出。モデルごとに expert バイト数が大きく変わる大規模 MoE で必須 | 下記「常駐キャッシュ容量」 |
| **GPU ワーキングセット上限で頭打ち** | 予算 = min(物理RAM, `max_recommended_working_set_size`)。128GB 機でも上限は 108GiB のため超過確保の失敗を防ぐ | 同上 |
| **モデル置き場の分離** | `ELFMOON_MODELS_ROOT128` を優先（未設定時 `ELFMOON_MODELS_ROOT`）。ElfMoon4 と同名モデルの取り違え防止 | 下記「セットアップ」 |
| **glm4_moe 複合ルーター対応** | mlx_lm の `MoEGate`（タプル返し）を生 logits アダプタで配線。GLM-4.7 等（n_group=1）が動作 | — |
| **store の内蔵SSD配置を推奨** | decode は expert ストリーミングの I/O 律速。store を内蔵 SSD に置くと随時読みが速い | 下記「推奨高速化設定」 |

> store 形式（`l{層}_e{expert}.safetensors`, GROUP=64/BITS=4）は ElfMoon4 と**完全互換**。`integrate.py` は無改変。ElfMoon4 で分解した同一モデルの store をそのまま流用できる。

---

## 動作要件

| 項目 | 要件 |
|---|---|
| ハードウェア | Apple Silicon Mac、**RAM 128GB 推奨**（M3/M4/M5 Max・Ultra クラス）。検証機は M5 Max 128GB |
| OS | macOS 14 以降 （26.5以降推奨）|
| ディスク空き | 元モデル + 分解済 expert でモデルサイズの約2倍。Qwen3-235B(123GB): ~246GB / GLM-4.7(185GB): ~370GB。高速化のため store は内蔵 SSD 推奨 |
| モデル置き場 | `ELFMOON_MODELS_ROOT128`（未設定時 `ELFMOON_MODELS_ROOT`）配下 |
| Python / 依存 | [ElfMoon4 の README](../ElfMoon4/README.md#依存ファイルのインストール) と同じ（MLX / mlx-lm / **transformers==4.57.6**） |

---

## 動作確認済みモデル（ElfMoon128, M5 Max 128GB 実測）

128GB オンメモリでは載り切らない（GPU ワーキングセット上限 108GiB を超える）巨大 MoE を、ストリーミングで実用速度動作させたもの。計測は下記「推奨高速化設定」下での定常値（warm）。

| モデル | サイズ | 常駐率 | デコード t/s | 備考 |
|---|---|---|---|---|
| **[Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit)**（最推奨） | 123 GB | 82% | **11.0** | qwen3_moe ネイティブ。命中率100%到達。品質・速度の最良バランス |
| **[GLM-4.7](https://huggingface.co/mlx-community/GLM-4.7-4bit)** | 185 GB | 54% | **6.7** | glm4_moe。命中率93%。複合ルーター対応済み（n_group=1） |

> - 未検証だが同経路で動作見込み: Qwen3-235B-A22B-Thinking / Qwen3-Coder-480B-A35B（252GB）/ GLM-4.6（185GB）。
> - n_group>1（Kimi-K2 / Ring-1T 等）は現状のグループルーティング未対応。

### Kimi K3 非刈込ティア（761GB）を 128GB 機で動作

**モデルサイズの 1/6 に満たないメモリで 761GB の MoE を走らせ、一貫した応答を得た。** モデルは [kernelpool/Kimi-K3-2bit-UVMAX](https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX)（896 experts 無刈込・混合精度）。

> 本モデル自体は作者が [PR #1626](https://github.com/ml-explore/mlx-lm/pull/1626) で 3.58 tok/s の生成を報告済み（オンメモリ想定）。ElfMoon128 の主張は **128GB という桁違いに小さいメモリで同じ非刈込ティアを動かした**点にある。

| 項目 | 値 |
|---|---|
| モデル / store | 761 GB / 713 GB（92 MoE 層 × 896 experts = 82,432 ファイル） |
| 常駐率 | 4.7%（3,925 / 83,328 expert、省メモリモード） |
| デコード | **1.1 tok/s**（462 トークン生成時、命中率 52%） |
| プリフィル | 1.0 tok/s（130 トークン） |
| 出力 | 日本語で自然な応答。EOS まで到達しループなし |

> 短い生成（〜70 トークン）ではプリフィルとウォームアップが支配的で 0.4 tok/s 程度。上表は 462 トークン生成時の定常値。

> ⚠️ **実用速度ではない（デモンストレーション）。** top16 × 92 層 ≈ 1,472 expert 読み／トークンに対し常駐率 7.7% のため、decode がほぼ全域 SSD I/O 律速になる。実運用は上表の Qwen3-235B / GLM-4.7 を推奨。

配線の正しさは出力文ではなく数値で裏付けている: 純正 `SwitchGLU` と同一入力を与えた比較で **max abs diff 0.0027**（参照値の絶対平均 1.43 に対し相対 0.02% ＝ fp16 の丸め誤差レベル）。

Kimi K3 は routed expert を hidden(7168) ではなく **latent 空間(3584)** で動かし、独自の group routing を持つ。MoE ブロックごと差し替える従来方式では固有構造を失うため、`SwitchGLU` 互換の `StreamingSwitchGLU` で **`switch_mlp` 属性のみ**を差し替える方式を新設した（モデル固有部分は mlx-lm 純正のまま残る）。詳細・再現手順は [`evidence/elfmoon128/kimi-k3-2bit-uvmax.md`](evidence/elfmoon128/kimi-k3-2bit-uvmax.md)。mlx-lm 未マージの [PR #1626](https://github.com/ml-explore/mlx-lm/pull/1626) の配置が必要。
> - 128GB 超 MoE の候補一覧は [`evidence/elfmoon128/moe-candidates-over-128gb.md`](evidence/elfmoon128/moe-candidates-over-128gb.md) を参照。

### 参考: 108GiB 内でオンメモリ動作するモデル

108GiB に収まるモデルは分解不要でそのまま高速動作する（ストリーミング機構は不使用）。ElfMoon4 の [オンメモリモデル手順](../ElfMoon4/README.md#オンメモリモデル推奨-gemma4-26b) と同じ。カスタムアーキ例:

| モデル | サイズ | t/s | 対応方法 |
|---|---|---|---|
| **[Laguna-S-2.1-MLX-4bit](https://huggingface.co/Vontra/Laguna-S-2.1-MLX-4bit)** | 62 GB | **63〜69** | Poolside 独自アーキ（118B MoE）。mlx_lm 未対応のため [PR の `laguna.py`](https://github.com/ml-explore/mlx-lm/pull/1601) を `site-packages/mlx_lm/models/` に配置 |

---

## 推奨高速化設定（M5 Max 128GB での実測ベース）

decode は expert ストリーミングの I/O 律速のため、命中率と I/O 帯域が速度を決める。

```bash
# 1) GPU ワーキングセット上限を引き上げ（要 sudo・再起動でリセット）
sudo sysctl iogpu.wired_limit_mb=122880          # 120GB

# 2) store（分解済 expert）を内蔵 SSD に置く（外付けより随時読みが ~2倍速い）
ELFMOON_STORE_DIR=/path/to/internal/store \
ELFMOON_MODEL_DIR=/path/to/model \
ELFMOON_MEM_BUDGET_GB=120 \                       # 上限120GBを予算に反映
ELFMOON_TOP_K=6 \                                 # ルーティング top_k 削減（8→6, 品質微減）
  python3 elfmoon/chat.py --no-think --perf
```

- GLM-4.7: 上記の組み合わせで **2.3→6.7 tok/s**（約3倍、命中率88→95%）。
- Qwen3-235B: expert が小型・高常駐率のため命中率100%に達し、追加調整なしで **11 tok/s**。
- ボトルネック分析・レバー別内訳は [`evidence/elfmoon128/`](evidence/elfmoon128/) 参照。

---

## セットアップ

基本手順（`ELFMOON_MODELS_ROOT` 規約、`integrate.py split_all` による分解、各モデルの DL 例、オンメモリ/Heretic/GLM 等）は **[ElfMoon4 の README「セットアップ」](../ElfMoon4/README.md#セットアップ) と共通**。ElfMoon128 固有の差分のみ以下に記す。

### モデル置き場（ELFMOON_MODELS_ROOT128）

ElfMoon128 は `ELFMOON_MODELS_ROOT128` を優先し、未設定時のみ `ELFMOON_MODELS_ROOT` にフォールバックする。ElfMoon4 と同名モデルでも量子化・サイズが異なるため置き場を分離する。

```bash
echo 'export ELFMOON_MODELS_ROOT128=/Volumes/990Pro_2TB/elfmoon128/models' >> ~/.zshrc
```

ディレクトリ規約（`<モデル名>/config.json, *.safetensors, store/`）は ElfMoon4 と同一。

### store を別ドライブに置く（ELFMOON_STORE_ROOT128）

decode は expert ストリーミングの I/O 律速のため、**モデル本体を外付け SSD に置き、`store/` だけ内蔵 SSD に置く**構成が速い。`ELFMOON_STORE_ROOT128` を設定すると `<ルート>/<モデル名>/store` を自動で解決する（実在する場合のみ採用するため、モデル直下 `store/` の既定規約は変わらない）。

```bash
echo 'export ELFMOON_STORE_ROOT128=~/.elfmoon128_store' >> ~/.zshrc
```

単発で指定する場合は従来どおり `ELFMOON_MODEL_DIR` + `ELFMOON_STORE_DIR`（`--model` は付けない）。

> ⚠️ store のパスを取り違えるとストリーミングが無効化され、巨大モデルを丸ごとロードして強制終了(OOM)する。起動時に `実効容量 …` の行が出ていればストリーミングは有効。出ていなければ store が解決できていない。

### 常駐キャッシュ容量（自動導出）

ElfMoon4 は常駐 expert 数を固定値 6144 としていたが、ElfMoon128 では **メモリ予算から自動導出**する（大規模 MoE では 1 expert のバイト数がモデルごとに大きく変わり、スロット数固定では実メモリ使用量が破綻するため）。

```
容量 = (予算 × headroom − 非expert重みの実測値) ÷ 1expertのバイト数     （上限: expert 総数）
```

- **予算**: `ELFMOON_MEM_BUDGET_GB` があればその値。無ければ **物理 RAM と GPU ワーキングセット上限（`max_recommended_working_set_size`、M5 Max 実測 108GiB）の小さい方**。超過すると算術が正しくても確保に失敗するため頭打ちにする。
- **headroom**: 既定 0.75（`--perf` 時 0.85）。KV キャッシュ・活性化・他アプリの取り分を残す。
- **非 expert 重み**: 融合 expert 解放後の `mx.get_active_memory()` の実測値。常に常駐するため必ず差し引く（遅延評価で過小測定した場合は暫定容量にフォールバック）。
- **上限**: `num_hidden_layers × num_experts`。全部載るモデルで無駄なスロットを確保しない。

```bash
python3 elfmoon/chat.py --model qwen3-235b-a22b-instruct-4bit          # 自動（既定）
ELFMOON_MEM_BUDGET_GB=96 python3 elfmoon/chat.py --model <名前>        # 予算を明示
python3 elfmoon/chat.py 6144 --model <名前>                            # expert 数を直接指定（従来互換）
```

起動時に導出の内訳が表示される:

```
  性能モード: 実効容量 9891（97.8GB） ← 予算120GB×0.85 − 非expert4.2GB / expert10.13MB, 上限12032
```

> ロジックは [`test_capacity_plan.py`](elfmoon/test_capacity_plan.py) で単体検証済み。GLM-4.7 / Qwen3-235B で実地検証済み（[`evidence/elfmoon128/`](evidence/elfmoon128/)）。

---

## 使い方・テスト・トラブルシューティング・ディレクトリ構成

いずれも ElfMoon4 と共通。以下を参照:

- [対話 CLI: chat.py](../ElfMoon4/README.md#対話cli-chatpy)
- [API サーバー: api_server.py](../ElfMoon4/README.md#api-サーバー-api_serverpy)
- [テスト・検証](../ElfMoon4/README.md#テスト検証)（ElfMoon128 は加えて `python3 elfmoon/test_capacity_plan.py`）
- [トラブルシューティング](../ElfMoon4/README.md#トラブルシューティング)
- [ディレクトリ構成](../ElfMoon4/README.md#ディレクトリ構成)

差分は「常駐容量が既定で自動導出（`capacity=auto` 表示）」になる点のみ。位置引数や `ELFMOON_MEM_BUDGET_GB` での明示指定は上記「常駐キャッシュ容量」を参照。

---

## ライセンス / クレジット

Apache License 2.0。モデル本体のライセンスは配布元のモデルカードに従うこと。
着想元・基盤は [ElfMoon4 のクレジット](../ElfMoon4/README.md#クレジット)と同じ（[antirez/ds4](https://github.com/antirez/ds4) / [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm)）。
