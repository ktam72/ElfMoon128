"""ファイル操作ツールの E2E 回帰テスト（MCP 実操作 + モデル→サーバー抽出検証）。

テスト専用 MCP サーバー（elfmoon/test/test_mcp_file_server.py）を一時ディレクトリで
起動し、以下の 2 層でファイル操作を検証する:

  ① MCP 実操作検証: mcp_client.call_tool 経由で read/write/edit/delete を実行し、
     一時ディレクトリ内の実ファイルが正しく操作されることを確認。
  ② モデル→サーバー抽出検証: api_server を起動し、ツール定義を注入して
     モデルが read/write/edit/delete を tool_calls として正しく出力することを確認。

ファイル操作はすべてテスト専用の一時ディレクトリ内で行われ、テスト終了時に削除される。
既存のファイル・ディレクトリには触れない。

usage:
  python3 elfmoon/test/test_file_ops.py [--model NAME]
  （既定: model=laguna-s-2.1-4bit）
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

MCP_SERVER_SCRIPT = HERE / "test_mcp_file_server.py"
API_SERVER_SCRIPT = ROOT / "elfmoon" / "api_server.py"
MCP_SERVER_NAME = "filetest"

# ---- ① MCP 実操作検証 ----


def run_mcp_ops(fs_dir: Path):
    """テスト専用 MCP サーバーを mcp_client 経由で起動し、4 操作を実検証する。"""
    sys.path.insert(0, str(HERE.parent))
    os.environ["ELFMOON_TEST_FS_DIR"] = str(fs_dir)

    # mcp_client はモジュール import 時に設定ファイルを自動検出しない（load() 時に読み込む）ため、
    # ELFMOON_MCP_CONFIG にテスト用の設定 JSON を渡す
    cfg_path = fs_dir / "test_mcp_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mcp": {
                    MCP_SERVER_NAME: {
                        "type": "local",
                        "command": [sys.executable, str(MCP_SERVER_SCRIPT)],
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
    for tn in ("read_file", "write_file", "edit_file", "delete_file"):
        assert f"{MCP_SERVER_NAME}__{tn}" in tool_names, f"ツール未登録: {tn}"

    results = []

    def call(tn, args):
        return mcp_manager.call_tool(f"{MCP_SERVER_NAME}__{tn}", args)

    # 1. write_file: 新規作成
    t0 = time.time()
    try:
        out = call("write_file", {"path": "sample.txt", "content": "hello world"})
        ok = (fs_dir / "sample.txt").read_text() == "hello world"
        results.append(
            ("write_file（新規作成）", ok, f"content={out[:40]}", time.time() - t0)
        )
    except Exception as e:
        results.append(("write_file（新規作成）", False, str(e), time.time() - t0))

    # 2. read_file: 読み込み
    t0 = time.time()
    try:
        out = call("read_file", {"path": "sample.txt"})
        ok = out.strip() == "hello world"
        results.append(
            ("read_file（読み込み）", ok, f"content={out[:40]!r}", time.time() - t0)
        )
    except Exception as e:
        results.append(("read_file（読み込み）", False, str(e), time.time() - t0))

    # 3. edit_file: 置換編集
    t0 = time.time()
    try:
        call(
            "edit_file",
            {"path": "sample.txt", "old_string": "world", "new_string": "elfmoon"},
        )
        ok = (fs_dir / "sample.txt").read_text() == "hello elfmoon"
        results.append(
            (
                "edit_file（置換編集）",
                ok,
                f"content={(fs_dir / 'sample.txt').read_text()!r}",
                time.time() - t0,
            )
        )
    except Exception as e:
        results.append(("edit_file（置換編集）", False, str(e), time.time() - t0))

    # 4. delete_file: 削除
    t0 = time.time()
    try:
        call("delete_file", {"path": "sample.txt"})
        ok = not (fs_dir / "sample.txt").exists()
        results.append(
            ("delete_file（削除）", ok, "ファイル消滅確認", time.time() - t0)
        )
    except Exception as e:
        results.append(("delete_file（削除）", False, str(e), time.time() - t0))

    # 5. トラバーサル拒否（ルート外アクセス）。MCP の call_tool は isError を例外にせず
    #    エラーテキストを返すため、戻り値にエラー文言が含まれることを確認する。
    t0 = time.time()
    try:
        out = call("read_file", {"path": "../../../etc/hosts"})
        ok = "許可されません" in out or "存在しません" in out or "ルート外" in out
        detail = out[:60] if ok else f"拒否が機能していない: {out[:60]}"
        results.append(("トラバーサル拒否", ok, detail, time.time() - t0))
    except Exception as e:
        results.append(
            (
                "トラバーサル拒否",
                True,
                f"拒否 OK（例外）: {str(e)[:50]}",
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


def file_tools():
    """モデルに注入するファイル操作ツール定義（MCP の tool_call 名に合わせる）。"""
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "ファイルパス"}},
        "required": ["path"],
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "ファイルを読み込む",
                "parameters": schema,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "ファイルを作成・上書きする",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "ファイルの文字列を置換して編集する",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "ファイルを削除する",
                "parameters": schema,
            },
        },
    ]


PROMPTS = {
    "read_file": "read_file 関数を呼び出して sample.txt の内容を読み込んでください。引数は path='sample.txt' です。",
    "write_file": "write_file 関数を呼び出して newfile.txt を新規作成してください。引数は path='newfile.txt', content='Hello from test' です。",
    "edit_file": "edit_file 関数を呼び出して sample.txt 内の「old」を「new」に置換してください。引数は path='sample.txt', old_string='old', new_string='new' です。",
    "delete_file": "delete_file 関数を呼び出して sample.txt を削除してください。引数は path='sample.txt' です。",
}


def _models_root():
    """プロジェクト（ElfMoon128 は *_MODELS_ROOT128、それ以外は ELFMOON_MODELS_ROOT）を返す。"""
    if "ElfMoon128" in str(ROOT):
        return os.environ.get("ELFMOON_MODELS_ROOT128") or os.environ.get(
            "ELFMOON_MODELS_ROOT", ""
        )
    return os.environ.get("ELFMOON_MODELS_ROOT", "")


def run_extract_e2e(model, fs_dir: Path):
    """api_server を起動し、モデルが各ファイル操作ツールを tool_calls として出力するかを検証する。"""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    models_root = _models_root()
    if not Path(models_root, model).exists():
        return None, [("モデル不在", False, f"{models_root}/{model}", 0)]

    # テスト用ファイルを用意
    (fs_dir / "sample.txt").write_text("old content")

    proc = subprocess.Popen(
        [sys.executable, str(API_SERVER_SCRIPT), str(port), "--model", model],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    try:
        if not wait_ready(url):
            return proc, [("サーバー起動失敗", False, "起動待機タイムアウト", 0)]
        results = []
        tools = file_tools()
        for op, prompt in PROMPTS.items():
            t0 = time.time()
            # モデル出力の揺れに備えてリトライ（ツール名が一致するまで最大 3 回）
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
        return proc, results
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laguna-s-2.1-4bit")
    args = ap.parse_args()

    fs_dir = Path(tempfile.mkdtemp(prefix="elfmoon_filetest_"))
    try:
        print("== ① MCP 実操作検証（自前ファイル操作サーバー） ==", flush=True)
        mcp_results = run_mcp_ops(fs_dir)
        for name, ok, detail, dt in mcp_results:
            print(f"  {name}: {'✅' if ok else '❌'} {detail} ({dt:.1f}s)", flush=True)

        print()
        print("== ② モデル→サーバー抽出検証（E2E） ==", flush=True)
        proc, e2e_results = run_extract_e2e(args.model, fs_dir)

        all_results = mcp_results + e2e_results
        print()
        print(f"{'項目':<32} {'結果':<4} 備考")
        print("-" * 78)
        all_ok = True
        for name, ok, detail, dt in all_results:
            all_ok = all_ok and ok
            print(f"{name:<32} {'✅' if ok else '❌':<4} {detail} ({dt:.1f}s)")
        print("-" * 78)
        print(f"総合: {'PASS' if all_ok else 'FAIL'}")
        return 0 if all_ok else 1
    finally:
        shutil.rmtree(fs_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
