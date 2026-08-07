"""ElfMoon4 各モデルのコーディング品質チェック＋速度計測。

プロンプト（elfmoon/test/coding_eval_prompts.txt）で関数名・クラス構造・
メソッド名を明確に指定し、モデル出力を**そのまま実行**して期待結果と比較する。
添削・補正は一切行わない（関数名違い・構造違いは品質問題として FAIL）。

usage:
  python3 elfmoon/test/coding_eval_models.py [--model NAME]...
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

PROMPTS_FILE = HERE / "coding_eval_prompts.txt"
OUTPUT_FILE = ROOT / "evidence" / "coding_eval_models.md"


def load_prompts():
    prompts = []
    name = None
    buf = []
    for line in PROMPTS_FILE.read_text().splitlines():
        if line.startswith("### "):
            if name and buf:
                prompts.append((name, "\n".join(buf).strip()))
            name = line[4:].strip()
            buf = []
        else:
            buf.append(line)
    if name and buf:
        prompts.append((name, "\n".join(buf).strip()))
    return prompts


def extract_code(text):
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_py(code, timeout=20):
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None


def _fizzbuzz_expected():
    exp = ""
    for i in range(1, 101):
        if i % 15 == 0:
            exp += "FizzBuzz\n"
        elif i % 3 == 0:
            exp += "Fizz\n"
        elif i % 5 == 0:
            exp += "Buzz\n"
        else:
            exp += str(i) + "\n"
    return exp


def eval_fizzbuzz(code):
    """モデル出力の fizzbuzz 関数を呼び出し、標準出力を期待 FizzBuzz と厳密比較。"""
    script = code + "\n\n" + "fizzbuzz()"
    r = run_py(script)
    if r is None:
        return False, "実行タイムアウト"
    if r.returncode != 0:
        return False, f"実行エラー exit={r.returncode} {r.stderr.strip()[:120]}"
    expected = _fizzbuzz_expected()
    ok = r.stdout == expected
    if ok:
        return True, "出力一致"
    return False, f"出力不一致(len={len(r.stdout)}/{len(expected)})"


def eval_quicksort(code):
    """quicksort 関数をそのまま実行してソート結果を検証。"""
    script = (
        code
        + "\n\n"
        + "assert quicksort([3,1,4,1,5]) == [1,1,3,4,5]\n"
        + "assert quicksort([]) == []\n"
        + "assert quicksort([9,8,7]) == [7,8,9]"
    )
    r = run_py(script)
    if r is None:
        return False, "実行タイムアウト"
    if r.returncode != 0:
        return False, f"実行エラー exit={r.returncode} {r.stderr.strip()[:150]}"
    return True, "OK"


def eval_bank_account(code):
    """BankAccount クラスをそのまま実行し、入出金と残高を検証。"""
    script = (
        code
        + "\n\n"
        + "b = BankAccount()\n"
        + "b.deposit(100)\n"
        + "b.deposit(50)\n"
        + "assert b.balance() == 150\n"
        + "b.withdraw(30)\n"
        + "assert b.balance() == 120"
    )
    r = run_py(script)
    if r is None:
        return False, "実行タイムアウト"
    if r.returncode != 0:
        return False, f"実行エラー exit={r.returncode} {r.stderr.strip()[:150]}"
    return True, "OK"


def eval_palindrome(code):
    """is_palindrome 関数をそのまま実行して検証。"""
    script = (
        code
        + "\n\n"
        + "assert is_palindrome('tacocat') == True\n"
        + "assert is_palindrome('level') == True\n"
        + "assert is_palindrome('hello') == False\n"
        + "assert is_palindrome('') == True"
    )
    r = run_py(script)
    if r is None:
        return False, "実行タイムアウト"
    if r.returncode != 0:
        return False, f"実行エラー exit={r.returncode} {r.stderr.strip()[:150]}"
    return True, "OK"


def eval_bugfix(code):
    """修正後コードをそのまま実行し、6 が出力されるか検証。"""
    r = run_py(code)
    if r is None:
        return False, "実行タイムアウト"
    if r.returncode != 0:
        return False, f"実行エラー exit={r.returncode} {r.stderr.strip()[:120]}"
    out = r.stdout.strip()
    ok = out == "6"
    return ok, (f"出力={out!r}" if not ok else "出力=6")


def eval_prime_check(code):
    """is_prime 関数をそのまま実行して検証。"""
    script = (
        code
        + "\n\n"
        + "assert is_prime(2) == True\n"
        + "assert is_prime(3) == True\n"
        + "assert is_prime(17) == True\n"
        + "assert is_prime(4) == False\n"
        + "assert is_prime(1) == False"
    )
    r = run_py(script)
    if r is None:
        return False, "実行タイムアウト"
    if r.returncode != 0:
        return False, f"実行エラー exit={r.returncode} {r.stderr.strip()[:150]}"
    return True, "OK"


EVAL = {
    "fizzbuzz": eval_fizzbuzz,
    "quicksort": eval_quicksort,
    "bank_account": eval_bank_account,
    "palindrome": eval_palindrome,
    "bugfix": eval_bugfix,
    "prime_check": eval_prime_check,
}


def eval_task(name, text):
    code = extract_code(text)
    fn = EVAL.get(name.lower())
    if fn is None:
        return False, f"未知タスク: {name}"
    return fn(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None)
    args = ap.parse_args()

    from stream_model import list_models

    models = args.model or [m for m, _, _ in list_models()]
    prompts = load_prompts()
    print(f"対象モデル: {models}")

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    rows = []
    models_root = os.environ.get("ELFMOON_MODELS_ROOT128") or os.environ.get(
        "ELFMOON_MODELS_ROOT", ""
    )
    for mi, model in enumerate(models):
        print(f"[{mi + 1}/{len(models)}] {model}", flush=True)
        model_path = os.path.join(models_root, model)
        from mlx_lm import load as _mlx_load
        from mlx_lm.utils import load_model
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        try:
            m, tok = load(model_path, lazy=True)[:2]
        except Exception:
            # カスタム tokenizer_class（laguna の TokenizersBackend 等）は transformers
            # に存在しないため、tokenizer.json から PreTrainedTokenizerFast でフォールバック
            try:
                from transformers import PreTrainedTokenizerFast
                from tokenizers import Tokenizer

                _mp = Path(model_path)
                m = load_model(_mp, lazy=True)[0]
                tk = Tokenizer.from_file(str(_mp / "tokenizer.json"))
                tok = PreTrainedTokenizerFast(tokenizer_object=tk)
                _ct = _mp / "chat_template.jinja"
                if _ct.exists():
                    tok.chat_template = _ct.read_text()
                with open(_mp / "config.json") as _cfgf:
                    _cfg = json.load(_cfgf)
                _eos = _cfg.get("eos_token_id", 1)
                _eos_ids = _eos if isinstance(_eos, list) else [_eos]
                tok.eos_token_id = _eos_ids[0]
                tok = TokenizerWrapper(tok, eos_token_ids=_eos_ids)
            except Exception as e2:
                print(f"  ロード失敗: {e2}")
                rows.append((model, [(n, None, str(e2), 0, 0) for n, _ in prompts], 0))
                continue
        sampler = make_sampler(temp=0.0, top_p=0.95, top_k=20)
        task_rows = []
        speeds = []
        for name, prompt in prompts:
            p_tok = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            t0 = time.perf_counter()
            out = generate(
                m,
                tok,
                prompt=p_tok,
                max_tokens=300,
                sampler=sampler,
                prefill_step_size=4096,
                verbose=False,
            )
            dt = time.perf_counter() - t0
            try:
                n_tok = len(tok.encode(out))
            except Exception:
                n_tok = 0
            speeds.append(n_tok / dt if dt > 0 else 0)
            ok, detail = eval_task(name, out)
            task_rows.append((name, ok, detail, n_tok, dt))
            mark = "✅" if ok else "❌"
            print(
                f"  {name:<12} {mark} {n_tok}tok {n_tok / dt if dt > 0 else 0:.0f}t/s {detail[:50]}"
            )
        avg = sum(speeds) / len(speeds) if speeds else 0
        rows.append((model, task_rows, avg))

    lines = [
        "# ElfMoon4 各モデル コーディング品質チェック（2026-08-07）",
        "",
        "- 文言: elfmoon/test/coding_eval_prompts.txt（関数名・構造を明示）",
        "- 検証: モデル出力をそのまま実行→期待結果と厳密比較（添削・補正なし、temp=0.0, max_tokens=300）",
        "",
        "| モデル | fizzbuzz | quicksort | bank_account | palindrome | bugfix | prime_check | 平均 (tok/s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, tasks, avg in rows:
        cells = []
        for name, ok, detail, n_tok, dt in tasks:
            if ok is None:
                cells.append("⚠️")
            else:
                cells.append("✅" if ok else "❌")
        speed = f"{avg:.1f}" if avg else "—"
        lines.append(f"| {model} | {' | '.join(cells)} | {speed} |")
    lines.append("")
    OUTPUT_FILE.write_text("\n".join(lines))
    print(f"\n保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
