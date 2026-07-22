# 対応記録: プレフィル高速化 と MCP ツール連携（OpenAI 互換）

- 作成日: 2026-07-16
- 対象リポジトリ: `~/Documents/apps/ElfMoon4`
- 対象モデル検証: `qwen3.6-35b-mlx`（プレフィル）、`gemma-4-26b-a4b-it-heretic-4bit`（MCP/opencode）
- 変更ファイル: `elfmoon/stream_model.py`, `elfmoon/api_server.py`, `elfmoon/chat.py`

---

## 0. 概要

本対応は 2 本柱。

1. **プレフィル高速化**: per-expert ループの Python dispatch を融合テンソル + `gather_qmm` に置き換え、**8192 トークンで TTFT 約 3.7x**（2429 トークンで 15.5s→4.6s ≈ 3.3x）。デコード速度・出力は非回帰。
2. **MCP ツール連携の修正**: opencode 等の OpenAI 互換クライアントで、モデルのツール呼び出しが生テキストのまま漏れていた問題を解消。**クライアント側ツール実行方式（tool_calls を返す）**に統一し、加えて gemma のチャンネルマーカー漏れも除去。

> 補足: 投機的デコード（ツイート発の MTP）も検証したが、ElfMoon の compiled 単トークン decode が既に高速で net 高速化にならず**見送り**（詳細は memory `elfmoon-specdecode-feasibility`）。本コミットには含まない。

---

## 1. プレフィル高速化（融合テンソル + gather_qmm）

### 1.1 ボトルネック

warm プレフィルを cProfile で分解した結果、時間の大半が **`StreamingMoE.__call__` の per-expert ループ**（`expert_groups` の Python 構築、expert ごとの小さな `quantized_matmul` を数千回 dispatch、`tolist()` 同期）に費やされていた。単一プレフィルは expert working set（35B で 40 層 × 256 = 10,240）が常駐キャッシュ（6144）を超えるため**キャッシュ命中率 0%**で、ストリーミングキャッシュはプレフィルでは無力。I/O は全体の ~28%、残り ~72% が計算 + Python dispatch。

### 1.2 設計

`stream_model.py` に以下を追加。

- **`FusedPrefillStore`**: 元モデル safetensors の融合 expert テンソル
  `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`（`[n_experts, ...]`）を **mmap 直読み**する。integrate.py が分解する前の形をそのまま使うので store 再生成は不要。層ごとの dict は保持しない（融合テンソル ~450MB/層 × 40 層 = 18GB の蓄積を防ぐ。warm 時の再読込は OS のページキャッシュが担保）。
- **`_prefill_moe_gather`**: per-expert ループを、expert 順ソート付きの **`gather_qmm` 3 カーネル**（gate / up / down）に集約。`mlx_lm.models.switch_layers` の `_gather_sort` / `_scatter_unsort` を利用。トークン 64 個以上でソート有効（`do_sort = idx.size >= 64`）。数値は per-expert 経路と bf16 丸め差（~1e-3）で一致。

`StreamingMoE.__call__` の N>1 分岐を、閾値以上かつ融合テンソルが読める層でこの経路に切り替える:

```python
if fs is not None and N >= FUSED_MIN_TOKENS and self.layer_idx in fs:
    ...  # gather_qmm 経路
```

融合テンソルが読めないモデル・短チャンクは従来の per-expert 経路にフォールバック（短チャンクは ResidentCache が効くため per-expert が有利）。

### 1.3 メモリとパイプライン管理

- **`async_eval` 4 層ごと**（`layer_idx % 4 == 3`）: 全 40 層の融合テンソルを遅延評価すると 18GB 蓄積する。`mx.async_eval` は CPU をブロックせず GPU 実行を開始させるので、次層のロードと計算がオーバーラップする。
- **最終 MoE 層で `mx.eval` + `mx.clear_cache`**（`_is_last_moe`）: プレフィルで使った融合テンソルのバッファ（~18GB）をプールに残すと decode 時のメモリ圧で速度が落ちる（実測 23→17 t/s）ため即返却する。

### 1.4 チャンク幅

`PREFILL_STEP`（env `ELFMOON_PREFILL_STEP`、既定 **4096**）を `api_server.py` / `chat.py` に導入。gather_qmm 経路では融合テンソル読込がチャンク数に比例する固定費のため、チャンクは大きいほど長プロンプトで有利。ただし 8192 は活性化ピークが ~21.7GB に達し 24GB 機で危険なため既定は 4096（ピーク ~5GB）。

- `stream_generate`（chat.py が使用）の `prefill_step_size` 既定は **512** で融合閾値 2048 未満のため、明示的に `PREFILL_STEP` を渡さないと高速化されない点に注意。

### 1.5 実測（M4 Pro 24GB, qwen3.6-35b, 省メモリ 6144）

| プロンプト長 | 旧（per-expert, step 2048） | 新（gather_qmm, step 4096） | 倍率 |
|---|---|---|---|
| 2429 tok | 15.5s | 4.6s | **3.3x** |
| 8192 tok | (200 t/s 相当) | 730 t/s | **~3.7x** |

- デコード速度: 非回帰（定常 ~30 t/s、最終層 `clear_cache` により decode 側のメモリ圧を回避）
- 出力パリティ: greedy argmax サンプル 6/6 一致、80B 動作 OK、slot_cache テスト 5/5

---

## 2. MCP ツール連携（① クライアント側ツール実行）

### 2.1 症状

opencode で「カレントディレクトリの一覧表示」等を依頼すると、応答に
`<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|>` という**生のツール呼び出し文字列**がそのまま返っていた。

### 2.2 原因（2 層）

1. **マーカー不一致**: モデルの開始マーカーは `<|tool_call>`（`|` 片側）だが、コードは `TOOL_CALL_START = "<|tool_call|>"`（両側 `|`）を `find` していたため `_extract_tool_calls` が空を返し、生テキストが素通し。
2. **設計の食い違い**: 既存の `_generate_impl` はツールを**サーバー側で実行**するエージェントループ（`mcp_manager.call_tool`）だった。しかし opencode は標準の OpenAI function-calling＝「サーバーは `tool_calls` を**返す**だけ、実行はクライアント」を期待する。opencode の `bash` を ElfMoon 自前 MCP で実行することはできない。

### 2.3 対応（api_server.py）

**方針: ① クライアント側ツール実行に統一**（サーバーでは実行しない）。

- **マーカー堅牢化**: `_TC_START_RE = re.compile(r"<\|tool_call\|?>")` で `<|tool_call>` / `<|tool_call|>` 両形を許容。
- **エンジンは tool_calls を返す**: `_generate_impl` はツール検出時に `mcp_manager.call_tool` での実行をやめ、`{"tool_calls": [...], "content": ...}` を yield して終了。
- **ハンドラで OpenAI 形式返却**:
  - 非ストリーミング（`_handle_nonstream_tools`）: `message.tool_calls` ＋ `finish_reason: "tool_calls"`。
  - ストリーミング（`_handle_stream_tools`）: `delta.tool_calls`（`index` 付き）＋ 最終チャンク `finish_reason: "tool_calls"`。
- **抽出失敗ログ**: tools 有効時に抽出 0 件だが `"tool_call"` を含む生出力を stderr にダンプし、マーカーずれを即座に発見できるようにした。
- **既存バグ修正**: gemma 形式パース時に `call_end = close`（`}` 位置）で上書きしていたため終了マーカー `<tool_call|>` が cleaned テキストに残っていた不具合を除去（END マーカーの後まで消費）。

> `mcp_manager` の自前 MCP ツール注入フォールバック（クライアントが tools を送らない場合）は残置。opencode は毎回 tools を送るのでこの経路には入らない。

---

## 3. チャンネルマーカー除去（gemma）

### 3.1 症状

ツール呼び出し成功後の最終応答に `<|channel>thought` / `<channel|>` が漏れていた。

### 3.2 原因

ストリーミング出力が生の detokenizer 出力（`last_segment`）をそのまま流しており、`_clean_token_artifacts` もチャンネル除去も通っていなかった。gemma はチャンネル形式 `<|channel>thought\n{思考}\n<channel|>{回答}` で思考を包む。

### 3.3 対応（api_server.py）

- **`_strip_channels()` 追加**: モデルの `chat_template.jinja` 自身の除去ロジックに準拠。`<channel|>` で split し、`<|channel>` を含む part は `<|channel>` より前だけ残す。これで思考チャンネルとマーカーが消え最終回答のみ残る。マーカーを含まないテキスト（Qwen 等）は素通し。
- **非ツール分岐を全文処理化**: `_generate_impl` は生成完了後に全トークンを持っているため、逐次 detokenizer 流しをやめ、`output_text`（`_clean_token_artifacts` 済）を `_strip_channels` してから返す。`no_think` 時は `<think>` も `ThinkStripper` で除去。tool_calls に付随する `content` にも `_strip_channels` を適用。

> 注: 思考内容も（マーカーだけでなく）除去される。opencode 用途では妥当だが、推論を `reasoning_content` として残す場合は別途対応が必要。

---

## 4. 変更ファイル一覧

| ファイル | 主な変更 |
|---|---|
| `elfmoon/stream_model.py` | `FusedPrefillStore`, `_prefill_moe_gather`, `FUSED_MIN_TOKENS`, N>1 分岐の gather_qmm 化, `async_eval`/`clear_cache` によるメモリ管理, `_is_last_moe` 配線 |
| `elfmoon/api_server.py` | `PREFILL_STEP`, `_TC_START_RE`, `_strip_channels`, tool_calls 返却（stream/non-stream）, 抽出失敗ログ, END マーカー修正 |
| `elfmoon/chat.py` | `PREFILL_STEP` を `stream_generate` の `prefill_step_size` に伝搬 |

## 5. 環境変数

| 変数 | 既定 | 説明 |
|---|---|---|
| `ELFMOON_PREFILL_STEP` | 4096 | プレフィルのチャンク幅。大きいほど長プロンプトで高速だがメモリ増（8192 で ~21.7GB） |
| `ELFMOON_FUSED_MIN_TOKENS` | 2048 | この token 数以上の N>1 で gather_qmm 融合経路を使う閾値 |

## 6. 検証

- **プレフィル**: 2429tok 3.3x / 8192tok TTFT 3.7x、decode 非回帰、出力 argmax パリティ一致、80B 動作、slot_cache 5/5。
- **MCP**: `_extract_tool_calls` を観測フォーマットで単体検証（両マーカー形・gemma `call:` 形・OpenAI JSON 形・マーカー残留なし）。ハンドラの tool_calls 返却をモックで end-to-end 検証（stream/non-stream とも OpenAI 形式）。opencode + gemma-4-26b で実動作確認（`ls`, ファイル読取が正常動作、マーカー漏れなし）。
- **チャンネル**: `_strip_channels` を 5 ケースで検証（観測フォーマット・思考あり・前置き・Qwen 素通し・think 混在）。

## 7. 既知の制約 / TODO

- **非ツールの legacy 経路（`_generate_legacy`）はチャンネル未対応**。opencode は毎回 tools を送るため影響しないが、gemma で「ツール無しの素のチャット」をすると同じ漏れが起きる。対応するには真の逐次ストリーミング用のチャンネルストリッパーが必要。
- **思考内容の扱い**: 現状は除去。`reasoning_content` として surface する要望が出たら別途対応。
- **投機的デコード**: 本リポジトリには不採用（memory 参照）。
