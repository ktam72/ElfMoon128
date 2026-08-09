"""API サーバーの OpenAI 互換機能の静的・純関数検査。

api_server.py のモジュール import はモデルロードを伴わないため、純関数
（reasoning 分割 / content ブロック正規化 / tool replay）と AST 配線を検査する。

usage:
  python3 elfmoon/test/test_api_openai.py
"""

import ast
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from api_server import (  # noqa: E402
    _reasoning_enabled,
    _split_reasoning,
    _normalize_message,
    _first_tool_marker,
    _tool_call_complete,
    _TOOL_LOOKAHEAD,
    ReasoningSplitter,
)
from tool_replay import ToolReplayStore  # noqa: E402

API = HERE.parent / "api_server.py"


def check_pure():
    results = []

    # ---- reasoning 分割 ----
    r, c = _split_reasoning("<think>考え中</think>こんにちは")
    results.append(
        (
            "_split_reasoning 先頭think",
            r == "考え中" and c == "こんにちは",
            f"r={r!r} c={c!r}",
        )
    )
    r, c = _split_reasoning("回答")
    results.append(
        ("_split_reasoning thinkなし", r == "" and c == "回答", f"r={r!r} c={c!r}")
    )
    r, c = _split_reasoning("<think>未完")
    results.append(
        ("_split_reasoning 未終了think", r == "未完" and c == "", f"r={r!r} c={c!r}")
    )

    # ---- _reasoning_enabled ----
    e1 = _reasoning_enabled(None, None, "none")
    e2 = _reasoning_enabled(None, None, "low")
    e3 = _reasoning_enabled(True, None, None)
    e4 = _reasoning_enabled(False, None, None)
    e5 = _reasoning_enabled({"type": "disabled"}, None, None)
    e6 = _reasoning_enabled(None, False, None)
    results.append(("reasoning_effort none → off", e1 is False, str(e1)))
    results.append(("reasoning_effort low → on", e2 is True, str(e2)))
    results.append(("thinking true → on", e3 is True, str(e3)))
    results.append(("thinking false → off", e4 is False, str(e4)))
    results.append(("thinking disabled → off", e5 is False, str(e5)))
    results.append(("think false → off", e6 is False, str(e6)))

    # ---- content ブロック正規化（tool_use / tool_result / thinking） ----
    nm = _normalize_message(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "考え中"},
                {"type": "text", "text": "回答"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "Tokyo"},
                },
            ],
        }
    )
    ok = (
        nm["content"] == "回答"
        and nm["reasoning"] == "考え中"
        and len(nm["tool_calls"]) == 1
        and nm["tool_calls"][0]["id"] == "call_1"
        and nm["tool_calls"][0]["function"]["name"] == "get_weather"
        and json.loads(nm["tool_calls"][0]["function"]["arguments"])
        == {"city": "Tokyo"}
    )
    results.append(("content ブロック（tool_use/thinking/text）", ok, str(nm)))

    nm2 = _normalize_message(
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "晴れ",
                }
            ],
        }
    )
    results.append(
        (
            "content ブロック（tool_result）",
            nm2["content"] == "晴れ" and nm2["tool_call_id"] == "call_1",
            str(nm2),
        )
    )

    # ---- ReasoningSplitter ----
    sp = ReasoningSplitter(in_think=False)
    pieces = []
    for p in ["<thi", "nk>考え", "中</think>回答"]:
        r, c = sp.feed(p)
        if r:
            pieces.append(("r", r))
        if c:
            pieces.append(("c", c))
    joined_r = "".join(t for k, t in pieces if k == "r")
    joined_c = "".join(t for k, t in pieces if k == "c")
    results.append(
        (
            "ReasoningSplitter 分割",
            joined_r == "考え中" and joined_c == "回答",
            f"r={joined_r!r} c={joined_c!r}",
        )
    )

    print(f"{'項目':<34} {'OK':<4} 備考")
    print("-" * 80)
    all_ok = True
    for name, ok, note in results:
        all_ok = all_ok and ok
        print(f"{name:<34} {'✅' if ok else '❌':<4} {note}")
    print("-" * 80)
    return all_ok, results


def check_streaming():
    """ストリーミング化の純関数検査（tool 領域検出・後ろ倒し送出のシミュレーション）。"""
    results = []

    # _first_tool_marker: 各種マーカー検出
    m = _first_tool_marker("考えます<tool_call>get")
    results.append(("first_tool_marker GLM", m == 4, f"m={m}"))
    m = _first_tool_marker("<|tool_call|>get_time")
    results.append(("first_tool_marker pipe", m == 0, f"m={m}"))
    m = _first_tool_marker("こんにちは")
    results.append(("first_tool_marker なし", m is None, f"m={m}"))

    # _tool_call_complete
    results.append(
        (
            "tool_call_complete GLM",
            _tool_call_complete("<tool_call>a</tool_call>") is True,
            "",
        )
    )
    results.append(
        ("tool_call_complete 未完了", _tool_call_complete("<tool_call>a") is False, "")
    )
    results.append(
        (
            "tool_call_complete pipe",
            _tool_call_complete('<|tool_call|>{"name":"x"}<tool_call|>') is True,
            "",
        )
    )

    # 後ろ倒し送出のシミュレーション
    chunks = []  # 送出済み content 断片
    sent = 0
    text = ""
    tool_started = False

    def nonlocal_sent_update(v):
        nonlocal sent
        sent = v

    def feed(tok):
        nonlocal text, tool_started
        text += tok
        if tool_started:
            return
        if _first_tool_marker(text) is not None:
            tool_started = True
            limit = _first_tool_marker(text)
            if limit > sent:
                chunks.append(text[sent:limit])
                nonlocal_sent_update(limit)
            return
        limit = max(0, len(text) - _TOOL_LOOKAHEAD)
        if limit > sent:
            chunks.append(text[sent:limit])
            nonlocal_sent_update(limit)

    for tok in ["天気を", "調べ", "ます\n", "\n", "<tool_ca", "ll>get_w"]:
        feed(tok)
    joined = "".join(chunks)
    results.append(
        (
            "tool マーカー前に content のみ送出",
            joined == "天気を調べます\n\n" and "<tool_ca" not in joined,
            f"chunks={chunks!r}",
        )
    )

    full = (
        text + "eather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>"
    )
    results.append(("ツール完了で break 判定", _tool_call_complete(full) is True, ""))

    print(f"{'ストリーミング項目':<30} {'OK':<4} 備考")
    print("-" * 70)
    all_ok = True
    for name, ok, note in results:
        all_ok = all_ok and ok
        print(f"{name:<30} {'✅' if ok else '❌':<4} {note}")
    print("-" * 70)
    return all_ok, results


def check_wiring():
    src = API.read_text()
    tree = ast.parse(src)
    results = []

    def _has_call(name):
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == name
            for n in ast.walk(tree)
        )

    def _has_attr(attr):
        return any(
            isinstance(n, ast.Attribute) and n.attr == attr for n in ast.walk(tree)
        )

    def _has_name(name):
        return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(tree))

    def _has_str(s):
        return any(
            isinstance(n, ast.Constant) and isinstance(n.value, str) and s in n.value
            for n in ast.walk(tree)
        )

    results.append(("_handle_completions 定義", _has_attr("_handle_completions"), ""))
    results.append(("_handle_responses 定義", _has_attr("_handle_responses"), ""))
    results.append(("_handle_messages 定義", _has_attr("_handle_messages"), ""))
    results.append(("reasoning_content 出力", _has_str("reasoning_content"), ""))
    results.append(("_common_params 定義", _has_attr("_common_params"), ""))
    common = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_common_params"
        ),
        None,
    )
    cp_src = ast.get_source_segment(src, common) if common else ""
    results.append(("top_p 配線", '"top_p"' in cp_src, ""))
    results.append(("top_k 配線", '"top_k"' in cp_src, ""))
    results.append(("min_p 配線", '"min_p"' in cp_src, ""))
    results.append(("tool_choice none 判定", _has_str('"none"'), ""))
    results.append(("_reasoning_enabled 呼び出し", _has_call("_reasoning_enabled"), ""))
    results.append(("_apply_tool_replay 定義", _has_attr("_apply_tool_replay"), ""))
    results.append(("ToolReplayStore 使用", _has_name("ToolReplayStore"), ""))
    results.append(("_tool_replay 参照", _has_attr("_tool_replay"), ""))

    print(f"{'配線項目':<28} {'OK':<4} 備考")
    print("-" * 60)
    all_ok = True
    for name, ok, note in results:
        all_ok = all_ok and ok
        print(f"{name:<28} {'✅' if ok else '❌':<4} {note}")
    print("-" * 60)
    return all_ok, results


def check_tool_replay():
    import tempfile

    results = []

    tmp = tempfile.mkdtemp(prefix="elfmoon_replay_test_")
    replay_path = os.path.join(tmp, "replay.json")
    store = ToolReplayStore(max_entries=100, max_bytes=10_000_000, filepath=replay_path)
    call1 = {
        "id": "call_1",
        "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'},
    }
    call2 = {
        "id": "call_2",
        "function": {"name": "get_time", "arguments": '{"tz":"UTC"}'},
    }
    raw = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call><tool_call>get_time<arg_key>tz</arg_key><arg_value>UTC</arg_value></tool_call>"
    store.remember([call1, call2], raw)
    results.append(
        ("remember + exact 一致", store.exact_block([call1, call2]) == raw, "")
    )

    rendered = store.render_tool_calls([call1, call2])
    results.append(("render 1回出力", rendered == raw, f"{rendered!r}"))

    unknown = {"id": "call_9", "function": {"name": "x", "arguments": "{}"}}
    results.append(
        ("未知混在は canonical", store.exact_block([call1, unknown]) is None, "")
    )

    block = ToolReplayStore.extract_raw_block(
        "前文\n\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>後文"
    )
    results.append(
        (
            "extract_raw_block 連続ブロック",
            block
            == "\n\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
            f"{block!r}",
        )
    )

    store.persist()
    store2 = ToolReplayStore(filepath=replay_path)
    results.append(
        (
            "永続化往復",
            store2.exact_block([call1, call2]) == raw,
            f"count={store2.count}",
        )
    )

    print(f"{'replay項目':<32} {'OK':<4} 備考")
    print("-" * 70)
    all_ok = True
    for name, ok, note in results:
        all_ok = all_ok and ok
        print(f"{name:<32} {'✅' if ok else '❌':<4} {note}")
    print("-" * 70)
    return all_ok, results


def main():
    ok_pure, _ = check_pure()
    print()
    ok_stream, _ = check_streaming()
    print()
    ok_wire, _ = check_wiring()
    print()
    ok_replay, _ = check_tool_replay()
    ok = ok_pure and ok_stream and ok_wire and ok_replay
    print(f"総合: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
