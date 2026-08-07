"""ElfMoon4 全モデル速度ベンチ（M5 Max 実測）

- 文言: evidence/bench_prompts.txt（ElfMoonCoder coding_eval 由来 6 タスク）
- 経路1: api_server（HTTP / OpenAI 互換）
- 経路2: chat.py（CLI を pty 駆動、同一エンジン）
- 出力: evidence/bench_all_models.md + 標準出力に表

usage:
  python3 elfmoon/bench_models.py                 # 全モデル（--list と同一）
  python3 elfmoon/bench_models.py --model NAME    # 指定モデル
  python3 elfmoon/bench_models.py --skip-chat     # chat.py 経路を省略
  python3 elfmoon/bench_models.py --model NAME --skip-server
"""

import argparse
import json
import os
import pty
import re
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROMPTS_FILE = ROOT / "evidence" / "bench_prompts.txt"
OUTPUT_FILE = ROOT / "evidence" / "bench_all_models.md"
PORT_BASE = 11440
MAX_TOKENS = 150
WARMUP_TOKENS = 30
TIMEOUT = 900


def list_models():
    out = subprocess.run(
        [sys.executable, str(HERE / "chat.py"), "--list"],
        capture_output=True,
        text=True,
    ).stdout
    models = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("利用可能"):
            continue
        name = line.split()[0]
        if name:
            models.append(name)
    return models


def load_prompts():
    prompts = []
    name = None
    buf = []
    for line in PROMPTS_FILE.read_text().splitlines():
        if line.startswith("### "):
            if name and buf:
                prompts.append((name, "\n".join(buf)))
            name = line[4:].strip()
            buf = []
        else:
            buf.append(line)
    if name and buf:
        prompts.append((name, "\n".join(buf)))
    return prompts


def wait_server(port, timeout=300):
    url = f"http://localhost:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(1)
    return False


def stop_server(port):
    """api_server を確実に終了し、モデル解放まで待つ。

    pkill は SIGTERM を送るだけで、MLX の GPU メモリ解放（数秒〜数十秒）を
    待たないため、次プロセスと重なって二重起動 OOM（watchdog panic）を
    起こす。ここでは SIGTERM → 猶予 → 残存なら SIGKILL の順で確実に殺し、
    ポートが閉じてから返る。
    """
    pattern = f"api_server.py {port}"
    subprocess.run(["pkill", "-f", pattern], capture_output=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        if not _port_open(port):
            return
        # モデル解放の完了には数秒かかる。ポートが閉じるまで待つ。
        time.sleep(1)
    # 猶予後も残存 → 強制終了
    subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        if not _port_open(port):
            return
        time.sleep(1)


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def gen_stream(port, prompt, max_tokens, model="agents-a1-4bit"):
    """SSE ストリーミングで生成し、プリフィル(TTFT)を除外した生成 tok/s を返す。

    returns: (completion_tokens, generate_tok_per_sec, prompt_tokens, ttft, dt_gen)
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 20,
            "tools": [],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        body,
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    n_content = 0
    usage = None
    buf = b""
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            # \n\n で SSE イベントを分割して処理
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                for line in event.splitlines():
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    if obj.get("usage"):
                        usage = obj["usage"]
                    else:
                        n_content += 1
                # [DONE] 到達時の break を外側で扱うためループ継続
            if b"[DONE]" in buf:
                break
    dt_total = time.perf_counter() - t0
    dt_gen = dt_total - (ttft or 0.0)
    if usage:
        comp = usage["completion_tokens"]
        prompt_tok = usage["prompt_tokens"]
    else:
        comp = n_content
        prompt_tok = 0
    gen_tps = comp / dt_gen if dt_gen > 0 else 0.0
    return comp, gen_tps, prompt_tok, ttft, dt_gen


def gen(port, prompt, max_tokens, model="agents-a1-4bit"):
    """(completion_tokens, 生成時間 dt_gen, prompt_tokens) を返す。"""
    comp, _gen_tps, pn, _ttft, dt_gen = gen_stream(port, prompt, max_tokens, model)
    return comp, dt_gen, pn


def bench_one(model, port):
    log = f"/tmp/ef4_bench_{model.replace('/', '_')}.log"
    stop_server(port)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "api_server.py"),
            str(port),
            "6144",
            "--model",
            model,
        ],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        cwd=ROOT,
    )
    if not wait_server(port):
        stop_server(port)
        return None, f"起動失敗（{log}）"

    prompts = load_prompts()
    results = []
    try:
        for pname, prompt in prompts:
            # warmup
            try:
                gen(port, prompt, WARMUP_TOKENS, model)
            except Exception as e:
                results.append((pname, None, f"warmup 失敗: {e}"))
                continue
            # 計測 2 回
            for _ in range(2):
                try:
                    n, dt, pn = gen(port, prompt, MAX_TOKENS, model)
                    results.append((pname, n / dt, f"prompt={pn}"))
                except Exception as e:
                    results.append((pname, None, f"計測失敗: {e}"))
                    break
    finally:
        stop_server(port)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    ok = [r[1] for r in results if r[1]]
    if not ok:
        return None, "; ".join(r[2] for r in results if r[1] is None)
    return sum(ok) / len(ok), "; ".join(r[2] for r in results if r[1] is None)


CHAT_TPS_RE = re.compile(rb"decode (\d+) tokens, ([\d.]+) tok/s")

# chat 経路を server 経路と揃えるためのサンプラー設定。
# server 経路（gen_stream）は temp=0.0, top_p=0.95, top_k=20 を送信するため、
# chat.py 側も同じサンプラーになるようスラッシュコマンドで設定する。
CHAT_PARAM_CMDS = [
    b"/temp 0\n",
    b"/top-p 0.95\n",
    b"/top-k 20\n",
    b"/min-p 0\n",
    b"/repeat 1\n",
]


def chat_measure(model, prompt, chat_timeout=600):
    """chat.py を pty で起動し、1 プロンプトの decode tok/s を測る。

    ElfMoon4 と同じく「1 プロセス = 1 プロンプト」。プロンプトごとに新規起動するため
    会話履歴・KV キャッシュの残留がなく、ハングしない。server 経路と測定条件を
    揃えるため、サンプラーを CHAT_PARAM_CMDS で統一し、warmup 1 回後に計測 2 回する。
    returns: (平均 tok/s, note)
    """
    master, slave = pty.openpty()
    env = dict(os.environ)
    env.pop("PYTHONUNBUFFERED", None)
    env["ELFMOON_MAX_TOKENS"] = str(MAX_TOKENS)
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "chat.py"), "--model", model, "--no-think"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=ROOT,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    results = []
    try:
        # 起動完了（あなた>）を待つ
        buf = b""
        t0 = time.time()
        while b"\xe3\x81\x82\xe3\x81\xaa\xe3\x81\x9f" not in buf:  # 'あなた'
            r, _, _ = select.select([master], [], [], 1.0)
            if r:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    return None, "pty read error (起動時)"
                if not chunk:
                    return None, "pty EOF (起動時)"
                buf += chunk
            elif time.time() - t0 > chat_timeout:
                return None, "プロンプト表示タイムアウト"

        # chat.py の read_user_input は「あなた>」印字→setcbreak(生モード) の間に
        # 数 ms の窓があり、この窓に書くと canonical モードが \n を処理して
        # 改行なしで読み続けてブロックする（0% CPU）。検出後 0.8 秒待つ。
        time.sleep(0.8)

        # サンプラーを server 経路と同一に設定
        param_done = re.compile(rb"(repeat:|top-k:|top-p:|min-p:|temperature:)")
        for cmd in CHAT_PARAM_CMDS:
            os.write(master, cmd)
            _chat_read_until(master, param_done, chat_timeout)
            time.sleep(0.8)

        # chat.py の read_user_input は複数行をペースト扱いして空行待ちでブロックするため、
        # 計測時は改行をスペースに畳んで 1 行入力として送る（速度計測に影響しない）
        flat = " ".join(prompt.split())
        # warmup（server 経路の WARMUP_TOKENS と同様に 1 回）
        time.sleep(0.5)
        os.write(master, (flat + "\n").encode())
        _chat_read_until(master, CHAT_TPS_RE, chat_timeout)
        # 計測 2 回（server 経路と同一）
        for _ in range(2):
            time.sleep(0.8)
            os.write(master, (flat + "\n").encode())
            buf = _chat_read_until(master, CHAT_TPS_RE, chat_timeout)
            m = CHAT_TPS_RE.search(buf)
            if not m:
                break
            n_tok, tps = int(m.group(1)), float(m.group(2))
            results.append(tps)
    finally:
        try:
            os.write(master, b"exit\n")
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        os.close(master)
    ok = [v for v in results if v]
    if not ok:
        return None, "tok/s 行が出力されなかった"
    return sum(ok) / len(ok), ""


def chat_session(model, prompts, chat_timeout=600):
    """互換用: プロンプトごとに chat_measure を呼ぶ（1 プロセス = 1 プロンプト）。"""
    results = []
    for pname, prompt in prompts:
        tps, note = chat_measure(model, prompt, chat_timeout)
        results.append((pname, tps, note))
    return results


def _chat_read_until(fd, pat, timeout=600):
    buf = b""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r, _, _ = select.select([fd], [], [], 1.0)
        if r:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            if isinstance(pat, bytes):
                if pat in buf:
                    return buf
            elif pat.search(buf):
                return buf
    return buf


def bench_one_chat(model):
    prompts = load_prompts()
    results = []
    for pname, prompt in prompts:
        try:
            tps, note = chat_measure(model, prompt)
        except Exception as e:
            tps, note = None, f"計測失敗: {e}"
        results.append((pname, tps, note))
    ok = [r[1] for r in results if r[1]]
    if not ok:
        return None, "; ".join(r[2] for r in results if r[1] is None)
    return sum(ok) / len(ok), "; ".join(r[2] for r in results if r[1] is None)


def _store_dir(model):
    from stream_model import resolve_model

    mp, sd = resolve_model(model)
    return Path(sd) if sd else None


class StoreRename:
    """計測中だけ store/ を退避しオンメモリ経路で測る（streaming 非互換モデル用）。"""

    def __init__(self, model):
        self.sd = _store_dir(model)
        self.renamed = False

    def __enter__(self):
        if (
            self.sd
            and self.sd.is_dir()
            and not self.sd.with_name("store.bench").exists()
        ):
            self.sd.rename(self.sd.with_name("store.bench"))
            self.renamed = True
            print(f"  store/ → store.bench（オンメモリ計測）", flush=True)
        return self

    def __exit__(self, *exc):
        if self.renamed:
            self.sd.with_name("store.bench").rename(self.sd)
            print(f"  store.bench → store/ 復元", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--skip-server", action="store_true")
    ap.add_argument("--skip-chat", action="store_true")
    ap.add_argument(
        "--store-rename",
        action="append",
        default=[],
        help="計測中 store/ を退避しオンメモリで計測するモデル名（streaming 非互換モデル用）",
    )
    args = ap.parse_args()

    models = args.model or list_models()
    print(f"対象モデル: {models}")

    rows = []
    for i, m in enumerate(models):
        port = PORT_BASE + i
        print(f"[{i + 1}/{len(models)}] {m} 計測中...", flush=True)
        ctx = StoreRename(m) if m in args.store_rename else None
        if ctx:
            ctx.__enter__()
        try:
            s_tps, s_note = (
                (None, "スキップ") if args.skip_server else bench_one(m, port)
            )
            c_tps, c_note = (None, "スキップ") if args.skip_chat else bench_one_chat(m)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
        est_c = None
        est_s = None
        rows.append((m, s_tps, c_tps, est_s, est_c, s_note, c_note))
        s_str = f"{s_tps:.1f}" if s_tps else "—"
        c_str = f"{c_tps:.1f}" if c_tps else "—"
        print(f"  → server {s_str} / chat {c_str}")
        if not s_tps:
            print(f"    server 失敗: {s_note}")
        if not c_tps:
            print(f"    chat 失敗: {c_note}")

    lines = [
        "# ElfMoon4 全モデル速度ベンチ（M5 Max 実測）",
        "",
        f"- 実施日: {time.strftime('%Y-%m-%d')} / ハードウェア: M5 Max 128GB",
        f"- 文言: evidence/bench_prompts.txt（ElfMoonCoder coding_eval 由来 6 タスク）",
        f"- 経路: api_server（HTTP）・chat.py（pty、1 プロセス = 1 プロンプト）、temp=0.0/top_p=0.95/top_k=20、max_tokens={MAX_TOKENS}、warmup 後 2 回計測平均",
        "",
        "| モデル | M5 Max server (tok/s) | M5 Max chat (tok/s) | 備考 |",
        "|---|---|---|---|",
    ]
    for m, s_tps, c_tps, _est_s, _est_c, s_note, c_note in rows:
        s_str = f"{s_tps:.1f}" if s_tps else "—"
        c_str = f"{c_tps:.1f}" if c_tps else "—"
        note = s_note if s_tps else f"server: {s_note}"
        note += f"; chat: {c_note}" if (not c_tps and s_tps) else ""
        if m in args.store_rename:
            note = ("オンメモリ計測; " + note) if note else "オンメモリ計測"
        lines.append(f"| {m} | {s_str} | {c_str} | {note} |")
    lines.append("")
    OUTPUT_FILE.write_text("\n".join(lines))
    print(f"\n保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
