"""ElfMoon128 ストリーミング MoE モデルのコーディング品質チェック。

chat.py と同じロード経路（load_model → wire_streaming → stream_generate）で
6 タスクを生成し、モデル出力をそのまま実行→期待結果と厳密比較する。

usage:
  python3 elfmoon/test/coding_eval_streaming.py --model NAME
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

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.utils import load_model
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache

PROMPTS_FILE = HERE / "coding_eval_prompts.txt"
OUTPUT_FILE = ROOT / "evidence" / "coding_eval_streaming.md"


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
    r = run_py(code + "\n\nfizzbuzz()")
    if r is None:
        return False, "タイムアウト"
    if r.returncode != 0:
        return False, f"exit={r.returncode} {r.stderr.strip()[:120]}"
    expected = _fizzbuzz_expected()
    ok = r.stdout == expected
    return ok, ("出力一致" if ok else f"不一致(len={len(r.stdout)})")


def eval_bugfix(code):
    r = run_py(code)
    if r is None:
        return False, "タイムアウト"
    if r.returncode != 0:
        return False, f"exit={r.returncode} {r.stderr.strip()[:120]}"
    ok = r.stdout.strip() == "6"
    return ok, (f"出力={r.stdout.strip()!r}" if not ok else "出力=6")


def eval_standard(code, asserts):
    script = code + "\n\n" + "\n".join(asserts)
    r = run_py(script)
    if r is None:
        return False, "タイムアウト"
    if r.returncode != 0:
        return False, f"exit={r.returncode} {r.stderr.strip()[:150]}"
    return True, "OK"


ASSERTS = {
    "quicksort": [
        "assert quicksort([3,1,4,1,5]) == [1,1,3,4,5]",
        "assert quicksort([]) == []",
        "assert quicksort([9,8,7]) == [7,8,9]",
    ],
    "bank_account": [
        "b = BankAccount()",
        "b.deposit(100)",
        "b.deposit(50)",
        "assert b.balance() == 150",
        "b.withdraw(30)",
        "assert b.balance() == 120",
    ],
    "palindrome": [
        "assert is_palindrome('tacocat') == True",
        "assert is_palindrome('level') == True",
        "assert is_palindrome('hello') == False",
        "assert is_palindrome('') == True",
    ],
    "prime_check": [
        "assert is_prime(2) == True",
        "assert is_prime(3) == True",
        "assert is_prime(17) == True",
        "assert is_prime(4) == False",
        "assert is_prime(1) == False",
    ],
}


def eval_task(name, text):
    code = extract_code(text)
    n = name.lower()
    if n == "fizzbuzz":
        return eval_fizzbuzz(code)
    if n == "bugfix":
        return eval_bugfix(code)
    return eval_standard(code, ASSERTS.get(n, []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    from stream_model import wire_streaming, resolve_model, MODELS_ROOT

    model_path, store_dir = resolve_model(args.model)
    prompts = load_prompts()
    print(f"モデル: {args.model}（ストリーミング MoE）")

    _mp = Path(model_path)
    model, _ = load_model(_mp, lazy=True)
    try:
        _, tok = load(model_path, lazy=True)
    except Exception:
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from mlx_lm.tokenizer_utils import TokenizerWrapper

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

    if os.path.isdir(os.path.join(model_path, "store")):
        cache, _ = wire_streaming(
            model, None, store_dir=store_dir, model_path=model_path
        )
    else:
        cache = None

    sampler = make_sampler(temp=0.0, top_p=0.95, top_k=20, min_p=0.0)
    task_rows = []
    speeds = []
    for name, prompt in prompts:
        print(f"  [{name}] 生成開始...", flush=True)
        p_tok = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        t0 = time.perf_counter()
        out_parts = []
        try:
            for resp in stream_generate(
                model,
                tok,
                prompt=p_tok,
                max_tokens=400,
                sampler=sampler,
                prefill_step_size=4096,
            ):
                out_parts.append(resp.text)
            out = "".join(out_parts)
            dt = time.perf_counter() - t0
            try:
                n_tok = len(tok.encode(out))
            except Exception:
                n_tok = 0
            speeds.append(n_tok / dt if dt > 0 else 0)
            ok, detail = eval_task(name, out)
        except Exception as e:
            dt = time.perf_counter() - t0
            n_tok = 0
            ok, detail = None, f"生成エラー: {e}"
        task_rows.append((name, ok, detail, n_tok, dt))
        mark = "✅" if ok else "❌" if ok is not None else "⚠️"
        print(
            f"  {name:<12} {mark} {n_tok}tok {n_tok / dt if dt > 0 else 0:.1f}t/s {detail[:50]}"
        )

    avg = sum(speeds) / len(speeds) if speeds else 0
    lines = [
        "# ElfMoon128 ストリーミング MoE 品質チェック（2026-08-07）",
        "",
        f"- モデル: {args.model}",
        "- 検証: モデル出力をそのまま実行→期待結果と厳密比較（添削・補正なし）",
        "",
        "| モデル | fizzbuzz | quicksort | bank_account | palindrome | bugfix | prime_check | 平均 (tok/s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    cells = []
    for name, ok, detail, n_tok, dt in task_rows:
        cells.append("✅" if ok else "❌" if ok is not None else "⚠️")
    lines.append(f"| {args.model} | {' | '.join(cells)} | {avg:.1f} |")
    lines.append("")
    OUTPUT_FILE.write_text("\n".join(lines))
    print(f"\n保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
