"""ツールコール抽出の E2E 回帰テスト（api_server を起動して実モデルで検証）。

api_server.py をバックグラウンドで起動し、ツールコールを要求するプロンプトを
送信して、応答の tool_calls が正しく抽出されるかを検証する。

抽出マーカー形式はモデルごとに異なる:
  - laguna 系: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
  - セクション系: <|tool_call_begin|>{json}<|tool_call_end|>
  - gemma 系: <|tool_call|>call:func{args}<tool_call|>

usage:
  python3 elfmoon/test/test_tool_call_e2e.py [--model NAME] [--port PORT] [--keep-server]
  （既定: model=laguna-s-2.1-4bit, port=自動選択）
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

SERVER_SCRIPT = ROOT / "elfmoon" / "api_server.py"

NO_ARG_PROMPT = "現在の日時を教えてください。get_current_time関数を呼び出してください。"
ARG_PROMPT = "get_mountain_info関数を name=富士山 で呼んでください。"

NO_ARG_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "現在の日時を取得する",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

ARG_TOOL = {
    "type": "function",
    "function": {
        "name": "get_mountain_info",
        "description": "山の情報を取得する",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "山の名前"}},
            "required": ["name"],
        },
    },
}


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url, timeout=300):
    """サーバー起動を /v1/models への GET が成功するまで待つ。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def chat(url, model, messages, tools):
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "tools": tools,
            "max_tokens": 512,
        }
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def check_tool_call(resp, expected_name, expected_args):
    """応答から tool_calls を取り出し、名前と引数を検証する。"""
    msg = resp["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, f"tool_calls が空。content={msg.get('content')!r}"
    names = [c["function"]["name"] for c in calls]
    if expected_name not in names:
        return False, f"関数名不一致: {names} (期待 {expected_name})"
    call = next(c for c in calls if c["function"]["name"] == expected_name)
    try:
        args = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError as e:
        return (
            False,
            f"arguments が JSON として不正: {call['function']['arguments']!r} ({e})",
        )
    if expected_args is not None and args != expected_args:
        return False, f"引数不一致: {args} (期待 {expected_args})"
    return True, f"OK name={expected_name} args={args}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laguna-s-2.1-4bit")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument(
        "--keep-server", action="store_true", help="テスト後もサーバーを残す"
    )
    ap.add_argument(
        "--server-only", action="store_true", help="サーバー起動のみ（手動確認用）"
    )
    args = ap.parse_args()

    if not SERVER_SCRIPT.exists():
        print(f"api_server.py が見つかりません: {SERVER_SCRIPT}")
        return 1

    port = args.port or _free_port()
    url = f"http://127.0.0.1:{port}"
    models_root = os.environ.get("ELFMOON_MODELS_ROOT128") or os.environ.get(
        "ELFMOON_MODELS_ROOT", ""
    )
    if not Path(models_root, args.model).exists():
        print(f"モデルが見つかりません: {models_root}/{args.model}")
        return 1

    proc = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT), str(port), "--model", args.model],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    try:
        print(f"サーバー起動待機: {url} (model={args.model})", flush=True)
        if not wait_ready(url):
            out, _ = proc.communicate(timeout=5)
            print("サーバーが起動しませんでした。出力:")
            print(out)
            return 1
        print("起動OK", flush=True)

        if args.server_only:
            print("server-only モード: Ctrl-C で終了", flush=True)
            proc.wait()
            return 0

        results = []

        # ケース1: 引数なしツール
        t0 = time.time()
        resp = chat(
            url, args.model, [{"role": "user", "content": NO_ARG_PROMPT}], [NO_ARG_TOOL]
        )
        ok, detail = check_tool_call(resp, "get_current_time", None)
        results.append(
            ("引数なしツール (get_current_time)", ok, detail, time.time() - t0)
        )
        print(
            f"  引数なし: {'✅' if ok else '❌'} {detail} ({time.time() - t0:.1f}s)",
            flush=True,
        )

        # ケース2: 引数ありツール
        t0 = time.time()
        resp = chat(
            url, args.model, [{"role": "user", "content": ARG_PROMPT}], [ARG_TOOL]
        )
        ok, detail = check_tool_call(resp, "get_mountain_info", {"name": "富士山"})
        results.append(
            ("引数ありツール (get_mountain_info)", ok, detail, time.time() - t0)
        )
        print(
            f"  引数あり: {'✅' if ok else '❌'} {detail} ({time.time() - t0:.1f}s)",
            flush=True,
        )

        print()
        print(f"{'項目':<38} {'結果':<4} 備考")
        print("-" * 78)
        all_ok = True
        for name, ok, detail, dt in results:
            all_ok = all_ok and ok
            print(f"{name:<38} {'✅' if ok else '❌':<4} {detail} ({dt:.1f}s)")
        print("-" * 78)
        print(f"総合: {'PASS' if all_ok else 'FAIL'}")
        return 0 if all_ok else 1
    finally:
        if not args.keep_server and not args.server_only:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
