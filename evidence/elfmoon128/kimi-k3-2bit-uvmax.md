# Kimi K3 (2bit UVMAX / 非刈込 896 experts) 動作記録

M5 Max 128GB, ElfMoon128。**モデルサイズ 761GB に対し 128GB のメモリで動作**させ、
一貫した応答を確認した。

> モデル自体は PR #1626 の作者が 3.58 tok/s の生成を報告済み（オンメモリ想定）。
> 本記録の主張は「128GB で非刈込ティアを動かした」点であり、初生成の主張ではない。
> なお [PipeNetwork/kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx) の README が
> 非刈込ティアを "have never produced a token" とするのは同者の別ビルドについての記述。

## 結果

| 項目 | 値 |
|---|---|
| モデル | [kernelpool/Kimi-K3-2bit-UVMAX](https://huggingface.co/kernelpool/Kimi-K3-2bit-UVMAX) 761GB |
| store | 713GB / 82,432 ファイル（92 MoE 層 × 896 experts）内蔵 SSD |
| 常駐 | 3,925 / 83,328 = 4.7%（省メモリ既定）/ 6,394 = 7.7%（`--perf`） |
| デコード | **1.1 tok/s**（chat.py, 462 トークン生成, 命中率 52%, 省メモリ既定） |
| プリフィル | 1.0 tok/s（130 トークン） |
| 出力 | 日本語で自然な応答。EOS 到達・ループなし |

短い生成ではプリフィルとウォームアップが分母を支配し 0.38〜0.41 tok/s に見える
（40〜74 トークン計測時）。462 トークンまで回した 1.1 tok/s が定常値。

chat.py 実測ログ（自己紹介を依頼、462 トークン）:

```
  省メモリモード: 実効容量 3925（34.0GB） ← 予算108GB×0.75 − 非expert46.7GB / expert8.86MB, 上限83328
  ウォームスタート: 3925 experts プライム（16秒）
準備完了（36秒）
...
<|open|>response<|sep|>こんにちは！初めまして、Claude と申します。…
（プリフィル 130tok 1tok/s ／ 出力 462 tokens, 1.1 tok/s, 命中率52%）
```

think チャネル → response チャネル → `<|end_of_msg|>` の構造が正常に出ており、
長文でも破綻しない。なお本ビルドは自己紹介で "Claude" と名乗る（蒸留由来と思われる）。

> 0.38 tok/s は実用速度ではない。top16 × 92層 ≈ 1,472 expert 読み/トークンに対し
> 常駐率 7.7% のため、decode がほぼ全域 SSD I/O 律速になる。デモとしての意義。

## 実装: switch_mlp のみ差し替え（StreamingSwitchGLU）

Kimi K3 の MoE ブロックは routed expert を **latent 空間 3584**（hidden 7168 ではない）で
動かし、`routed_expert_down_proj → experts → routed_expert_norm → routed_expert_up_proj`
という固有構造を持つ。加えてルーティングは group select（`_group_expert_select`）。

従来どおり `layer.mlp` ごと `StreamingMoE` に置換すると latent 射影・shared experts・
group routing をすべて失い、hidden 7168 が latent 前提の expert に流れて shape 不一致で落ちる。

そこで **`mlp.switch_mlp` 属性だけ**を `SwitchGLU` 互換の `StreamingSwitchGLU` に差し替える方式に変更した。
契約は `(x[..., D], indices[..., k]) -> [..., k, D]`（重み付け・総和は呼び出し側）。
これによりモデル固有部分は純正実装のまま残る。

## 数値パリティ検証（配線の正しさの証明）

layer 1 の純正 `SwitchGLU` と `StreamingSwitchGLU` に同一入力
（latent (4, 3584) / 固定 indices (4, 16)）を与えて比較:

```
layer 1: bits=2 gs=128 mode=affine / latent=3584 top_k=16 n_exp=896
shape ref/new: (4, 16, 3584) (4, 16, 3584)
max abs diff : 0.00271      （参照値の絶対平均 1.431 に対し相対 0.02%）
mean abs diff: 0.00025
```

fp16 の丸め誤差レベルで一致。出力文の妥当性ではなく数値で配線を裏付けている。

## 既知の制限

Kimi 経路は `switch_mlp` 差し替え後に `continue` するため、`fused_store` による
gather_qmm 高速プレフィルと最終 MoE 層での `clear_cache` を通らない。プレフィルは
Python の per-expert ループになる。短いプロンプトでは無視できるが、長文投入時は
プレフィルが支配的になる。

## 併せて入れた汎用修正（他モデルにも有効）

- expert の量子化パラメータを `QuantizedSwitchLinear` の `.bits/.group_size/.mode` から動的検出
  （従来は GROUP=64/BITS=4 固定。UVMAX は bits=2/group_size=128）
- `wire_streaming` の層探索を `model.language_model.model.layers`（VLM ラップ構造）まで拡張
- `chat.py`: `tokenizer.json` を持たないモデルは `AutoTokenizer(trust_remote_code=True)` へ直行
  （mlx_lm 内部の対話プロンプトで停止するのを回避）

## 環境状態（リポジトリ外・要注意）

`site-packages/mlx_lm/models/` に mlx-lm PR #1626（`kernelpool/mlx-lm@add-kimi-k3`）から
以下を配置済み。いずれも元ファイルは `.orig` でバックアップ:

- `kimi_k3.py`（新規・mlx_lm 未マージ）
- `gated_delta.py`（`lower_bound`/`beta_scale` 引数追加版が必要）
- `base.py`（mask の次元拡張 3 行）

## 前処理の注意

- `tokenizer_config.json` の `extra_special_tokens` が list のため dict へ変換が必要
  （transformers 4.57.6 要件。`.orig` バックアップあり）
- `split_all` は規模ゆえ OOM で中断する。残層に対し `integrate.split_layer()` を
  直接ループで呼び再開する（本モデルは計 5 回で完了）
