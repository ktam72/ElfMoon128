# ElfMoon128🌙

> 128GB RAM の Apple Silicon Mac 上で、**128GB オンメモリでは載り切らない巨大 MoE**（GPU ワーキングセット上限 108GiB 超）を Streaming MoE で実用速度で動かす MLX 推論エンジン

[ElfMoon4](../ElfMoon4/README.md) のフォーク。共通事項（依存インストール・API サーバー・トラブルシューティング等）は **[ElfMoon4 の README](../ElfMoon4/README.md) を参照**。本 README は ElfMoon128 で対応モデルを使うための最短手順を記載する。

> ElfMoon4 で構築したモデルディレクトリはそのまま動作可能（store 形式は完全互換）。

---

## クイックスタート（初回 〜 動作確認）

### 1. 環境変数（1 回だけ）

```bash
echo 'export ELFMOON_MODELS_ROOT128=/Volumes/990Pro_2TB/elfmoon128/models' >> ~/.zshrc
source ~/.zshrc
```

- モデル置き場。未設定時は `ELFMOON_MODELS_ROOT` にフォールバック。
- 依存のインストール（MLX / mlx-lm / `transformers==4.57.6`）は [ElfMoon4 README「依存ファイルのインストール」](../ElfMoon4/README.md#依存ファイルのインストール) を参照。

### 2. 対応モデルの準備と実行

対応モデルは「表のモデル名 = `--model` に渡す名前」で統一。**ダウンロード → 分解/変換 → 実行**の 3 ステップ。

#### Qwen3-235B（最推奨・品質/速度の最良バランス）

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit \
  --local-dir $ELFMOON_MODELS_ROOT128/qwen3-235b-a22b-instruct-4bit
python3 elfmoon/integrate.py split_all $ELFMOON_MODELS_ROOT128/qwen3-235b-a22b-instruct-4bit
python3 elfmoon/chat.py --model qwen3-235b-a22b-instruct-4bit
```

#### GLM-4.7

```bash
HF_HUB_DISABLE_XET=1 hf download mlx-community/GLM-4.7-4bit \
  --local-dir $ELFMOON_MODELS_ROOT128/glm-4.7-4bit
python3 elfmoon/integrate.py split_all $ELFMOON_MODELS_ROOT128/glm-4.7-4bit
python3 elfmoon/chat.py --model glm-4.7-4bit
```

#### 上級者向け（前提作業あり・詳細は evidence 参照）

| モデル | 対応方法 | 詳細 |
|---|---|---|
| Kimi K3（761GB・デモ） | mlx_lm PR #1626 の配置 + モデル DL | [`kimi-k3-2bit-uvmax.md`](evidence/elfmoon128/kimi-k3-2bit-uvmax.md) |
| Laguna S-2.1（オンメモリ・分解不要） | mlx_lm PR #1601 の配置 + モデル DL | [`laguna-s-2.1-onmemory.md`](evidence/elfmoon128/laguna-s-2.1-onmemory.md) |

### 3. 動作確認

起動時に下記の「実効容量」行が出ればストリーミングは有効（モデルは分解済み store から随時読込まれる）。

```
性能モード: 実効容量 9891（97.8GB） ← 予算120GB×0.85 − 非expert4.2GB / expert10.13MB, 上限12032
```

> ⚠️ この行が出ないと store が解決できておらず、巨大モデルを丸ごとロードして強制終了（OOM）する。`--model` 名と store のパスを確認する。

---

## 対応モデル（M5 Max 128GB 実測）

| モデル | サイズ | 常駐率 | デコード t/s | 備考 |
|---|---|---|---|---|
| **[Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit)**（最推奨） | 123 GB | 82% | **11.0** | 命中率100%到達。品質・速度の最良バランス |
| **[GLM-4.7](https://huggingface.co/mlx-community/GLM-4.7-4bit)** | 185 GB | 54% | **6.7** | 命中率93% |
| Kimi K3 非刈込（デモ） | 761 GB | 4.7% | 1.1 | 実用速度ではない（SSD I/O 律速） |
| Laguna S-2.1（オンメモリ） | 62 GB | — | 63〜69 | 108GiB 内に収まるため分解不要 |

> 各モデルの数値根拠・ボトルネック分析は [`evidence/elfmoon128/`](evidence/elfmoon128/) を参照。

---

## 推奨高速化設定（M5 Max 128GB 実測ベース）

```bash
# 1) GPU ワーキングセット上限を引き上げ（要 sudo・再起動でリセット）
sudo sysctl iogpu.wired_limit_mb=122880          # 120GB

# 2) store を内蔵 SSD に置き、予算・top_k を指定して実行
ELFMOON_STORE_DIR=/path/to/internal/store \
ELFMOON_MODEL_DIR=/path/to/model \
ELFMOON_MEM_BUDGET_GB=120 \
ELFMOON_TOP_K=6 \
  python3 elfmoon/chat.py --no-think --perf
```

- store を内蔵 SSD に置くと随時読みが外付けより ~2 倍速い（`ELFMOON_STORE_ROOT128` 設定で自動解決。下記セットアップ参照）。
- 効果（実測）: GLM-4.7 は 2.3→6.7 t/s、Qwen3-235B は 11 t/s。

---

## セットアップ（詳細）

基本手順は **[ElfMoon4 README「セットアップ」](../ElfMoon4/README.md#セットアップ)** と共通。ElfMoon128 固有の差分のみ記載する。

### モデル置き場（`ELFMOON_MODELS_ROOT128`）

ElfMoon4 と同名モデルでも量子化・サイズが異なるため置き場を分離する。ディレクトリ規約（`<モデル名>/config.json, *.safetensors, store/`）は ElfMoon4 と同一。

```bash
echo 'export ELFMOON_MODELS_ROOT128=/Volumes/990Pro_2TB/elfmoon128/models' >> ~/.zshrc
```

### store を別ドライブに置く（`ELFMOON_STORE_ROOT128`）

モデル本体を外付け SSD に置き、`store/` だけ内蔵 SSD に置くと decode が速い。設定すると `<ルート>/<モデル名>/store` を自動解決する（実在する場合のみ採用）。

```bash
echo 'export ELFMOON_STORE_ROOT128=~/.elfmoon128_store' >> ~/.zshrc
```

単発で指定する場合は従来どおり `ELFMOON_MODEL_DIR` + `ELFMOON_STORE_DIR`（`--model` は付けない）。

### 常駐キャッシュ容量（自動導出）

ElfMoon4 は常駐 expert 数を固定値 6144 としていたが、ElfMoon128 では **メモリ予算から自動導出**する（1 expert のバイト数がモデルごとに大きく変わるため）。

```
容量 = (予算 × headroom − 非expert重みの実測値) ÷ 1expertのバイト数     （上限: expert 総数）
```

- **予算**: `ELFMOON_MEM_BUDGET_GB` があればその値。無ければ物理 RAM と GPU ワーキングセット上限（M5 Max 実測 108GiB）の小さい方。
- **headroom**: 既定 0.75（`--perf` 時 0.85）。
- **上限**: `num_hidden_layers × num_experts`。

```bash
python3 elfmoon/chat.py --model qwen3-235b-a22b-instruct-4bit          # 自動（既定）
ELFMOON_MEM_BUDGET_GB=96 python3 elfmoon/chat.py --model <名前>        # 予算を明示
python3 elfmoon/chat.py 6144 --model <名前>                            # expert 数を直接指定（従来互換）
```

---

## 使い方・テスト・トラブルシューティング・ディレクトリ構成

いずれも [ElfMoon4 の README](../ElfMoon4/README.md) を参照:

- [対話 CLI: chat.py](../ElfMoon4/README.md#対話cli-chatpy)（`--model` / `--no-think` / `--fast` / `--perf`）
- [API サーバー: api_server.py](../ElfMoon4/README.md#api-サーバー-api_serverpy)
- [テスト・検証](../ElfMoon4/README.md#テスト検証)（ElfMoon128 は加えて `python3 elfmoon/test_capacity_plan.py`）
- [トラブルシューティング](../ElfMoon4/README.md#トラブルシューティング)

ElfMoon128 固有の差分は「常駐容量が既定で自動導出（`capacity=auto`）」になる点のみ。

---

## ライセンス / クレジット

Apache License 2.0。モデル本体のライセンスは配布元のモデルカードに従うこと。
着想元・基盤は [ElfMoon4 のクレジット](../ElfMoon4/README.md#クレジット)と同じ（[antirez/ds4](https://github.com/antirez/ds4) / [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm)）。
