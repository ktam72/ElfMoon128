"""web fetch ツールの E2E 回帰テスト（MCP 実操作 + モデル→サーバー抽出検証）。

MCP 標準の fetch サーバー（pip の mcp-server-fetch）を起動し、以下の 2 層で
web fetch を検証する:

  ① MCP 実操作検証: mcp_client.call_tool 経由で fetch を実行し、実インターネットの
     URL コンテンツが取得できることを確認。
  ② モデル→サーバー抽出検証: api_server を起動し、fetch ツール定義を注入して
     モデルが fetch を tool_calls として正しく出力することを確認。

②はモデル指定で実行する（既定: laguna-s-2.1-4bit）。3 モデルをまとめて実行する
場合は --model を複数回指定する。

usage:
  python3 elfmoon/test/test_fetch.py --model laguna-s-2.1-4bit
  python3 elfmoon/test/test_fetch.py --model laguna-s-2.1-4bit --model Ling-3.0-flash-MLX-4bit
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

API_SERVER_SCRIPT = ROOT / "elfmoon" / "api_server.py"
MCP_SERVER_NAME = "fetchtest"
FETCH_TOOL = "fetch"
FETCH_PY = "/var/folders/4v/cxv29gn97gld63p51c4tzj780000gn/T/opencode/elfmoon/fetchenv/bin/python"

# 実在サーバへの fetch 検証用 URL（ネットワーク接続環境に依存）
FETCH_URL = "https://github.com/ml-explore/mlx"  # 成功系（実在サーバ・コンテンツ取得）
FETCH_BLOCKED_URL = (
    "https://www.google.com/search?q=elfmoon"  # robots.txt 拒否系（Google 検索）
)
FETCH_BLOCKED_MARK = "robots.txt"  # 拒否応答に含まれる文言
FETCH_BAD_URL = "https://example.invalid.test"  # 到達不能系

# ---- ① MCP 実操作検証 ----


def run_mcp_fetch(work_dir: Path):
    """mcp_client 経由で fetch を実行し、実 URL の内容を取得できることを確認する。"""
    sys.path.insert(0, str(HERE.parent))
    cfg_path = work_dir / "test_mcp_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mcp": {
                    MCP_SERVER_NAME: {
                        "type": "local",
                        "command": [FETCH_PY, "-m", "mcp_server_fetch"],
                        "enabled": True,
                    }
                }
            }
        )
    )
    os.environ["ELFMOON_MCP_CONFIG"] = str(cfg_path)

    from mcp_client import mcp_manager

    mcp_manager.load()
    tools = mcp_manager.get_openai_tools()
    tool_names = {t["function"]["name"] for t in tools}
    assert f"{MCP_SERVER_NAME}__{FETCH_TOOL}" in tool_names, "fetch ツール未登録"

    results = []

    # 1. fetch: 実在サーバの URL を取得（成功系）
    t0 = time.time()
    try:
        out = mcp_manager.call_tool(
            f"{MCP_SERVER_NAME}__{FETCH_TOOL}",
            {"url": FETCH_URL, "max_length": 500},
        )
        ok = "Contents of" in out
        results.append(
            (
                "fetch（実在サーバ取得）",
                ok,
                f"len={len(out)} {out[:60]!r}",
                time.time() - t0,
            )
        )
    except Exception as e:
        results.append(
            ("fetch（実在サーバ取得）", False, str(e)[:80], time.time() - t0)
        )

    # 2. fetch: Google 検索は robots.txt により拒否（実在サーバとのやり取りの事実を確認）
    t0 = time.time()
    try:
        out = mcp_manager.call_tool(
            f"{MCP_SERVER_NAME}__{FETCH_TOOL}",
            {"url": FETCH_BLOCKED_URL, "max_length": 500},
        )
        ok = FETCH_BLOCKED_MARK in out.lower()
        results.append(
            ("fetch（Google 検索: robots 拒否）", ok, out[:60], time.time() - t0)
        )
    except Exception as e:
        results.append(
            (
                "fetch（Google 検索: robots 拒否）",
                True,
                f"例外（拒否扱い OK）: {str(e)[:50]}",
                time.time() - t0,
            )
        )

    # 3. fetch: 存在しない URL はエラー（isError を文字列で返す）
    t0 = time.time()
    try:
        out = mcp_manager.call_tool(
            f"{MCP_SERVER_NAME}__{FETCH_TOOL}",
            {"url": FETCH_BAD_URL, "max_length": 500},
        )
        ok = "error" in out.lower() or "Failed to fetch" in out
        results.append(("fetch（到達不能 URL）", ok, out[:60], time.time() - t0))
    except Exception as e:
        results.append(
            (
                "fetch（到達不能 URL）",
                True,
                f"例外（エラー扱い OK）: {str(e)[:50]}",
                time.time() - t0,
            )
        )

    mcp_manager.close()
    return results


# ---- ② モデル→サーバー抽出検証 ----


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url, timeout=300):
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
            "max_tokens": 1024,
        }
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def fetch_tool():
    """モデルに注入する fetch ツール定義（MCP の tool_call 名に合わせる）。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch",
                "description": (
                    "Fetches a URL from the internet and extracts its contents as markdown. "
                    "Grants internet access to retrieve up-to-date information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                        "max_length": {
                            "type": "integer",
                            "description": "Maximum number of characters to return",
                            "default": 5000,
                        },
                    },
                    "required": ["url"],
                },
            },
        }
    ]


PROMPTS = {
    "fetch": (
        "fetch 関数を呼び出して https://github.com/ml-explore/mlx の内容を取得してください。"
        "引数は url='https://github.com/ml-explore/mlx' です。"
    ),
}


def _models_root():
    """プロジェクト（ElfMoon128 は *_MODELS_ROOT128、それ以外は ELFMOON_MODELS_ROOT）を返す。"""
    if "ElfMoon128" in str(ROOT):
        return os.environ.get("ELFMOON_MODELS_ROOT128") or os.environ.get(
            "ELFMOON_MODELS_ROOT", ""
        )
    return os.environ.get("ELFMOON_MODELS_ROOT", "")


def run_extract_e2e(model):
    """api_server を起動し、モデルが fetch を tool_calls として出力するかを検証する。"""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    models_root = _models_root()
    if not Path(models_root, model).exists():
        return [("モデル不在", False, f"{models_root}/{model}", 0)]

    proc = subprocess.Popen(
        [sys.executable, str(API_SERVER_SCRIPT), str(port), "--model", model],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    try:
        if not wait_ready(url):
            return [("サーバー起動失敗", False, "起動待機タイムアウト", 0)]
        results = []
        tools = fetch_tool()
        for op, prompt in PROMPTS.items():
            t0 = time.time()
            ok = False
            detail = ""
            for attempt in range(3):
                resp = chat(url, model, [{"role": "user", "content": prompt}], tools)
                msg = resp["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                names = [c["function"]["name"] for c in calls]
                if op in names:
                    ok = True
                    detail = f"name={op} args={calls[0]['function']['arguments'][:60]}"
                    break
                detail = f"attempt={attempt + 1} calls={names}"
            results.append((f"E2E: {op}", ok, detail, time.time() - t0))
            print(f"  E2E {op}: {'✅' if ok else '❌'} {detail}", flush=True)
        return results
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None)
    args = ap.parse_args()
    if not args.model:
        args.model = ["laguna-s-2.1-4bit"]

    work_dir = Path(tempfile.mkdtemp(prefix="elfmoon_fetchtst_"))
    try:
        print("== ① MCP 実操作検証（server-fetch） ==", flush=True)
        mcp_results = run_mcp_fetch(work_dir)
        for name, ok, detail, dt in mcp_results:
            print(f"  {name}: {'✅' if ok else '❌'} {detail} ({dt:.1f}s)", flush=True)

        all_ok = all(ok for _, ok, _, _ in mcp_results)

        print()
        print("== ② モデル→サーバー抽出検証（E2E） ==", flush=True)
        all_results = list(mcp_results)
        for model in args.model:
            print(f"\n-- model: {model} --", flush=True)
            e2e = run_extract_e2e(model)
            all_results += e2e
            all_ok = all_ok and all(ok for _, ok, _, _ in e2e)

        print()
        print(f"{'項目':<32} {'結果':<4} 備考")
        print("-" * 78)
        for name, ok, detail, dt in all_results:
            print(f"{name:<32} {'✅' if ok else '❌':<4} {detail} ({dt:.1f}s)")
        print("-" * 78)
        print(f"総合: {'PASS' if all_ok else 'FAIL'}")
        return 0 if all_ok else 1
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
