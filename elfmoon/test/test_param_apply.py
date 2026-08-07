"""パラメータ適用の出力変化検証（mlx_lm generate 直接呼び出し）。

同一プロンプト・同一シードでパラメータ値を変え、出力が変化するか
（または期待通り働くか）を確認する。

検査項目:
  - temperature: temp 0.0（貪欲）と temp 0.8 で出力が異なるか
  - min-p: min_p 0.0 と min_p 0.1 で出力が異なるか
  - repetition penalty: repeat 1.0 と repeat 1.3 で繰り返しが減るか
  - システムプロンプト: system 指定で出力に反映されるか

usage:
  python3 elfmoon/test/test_param_apply.py [--model NAME]
  （既定は agents-a1-4b-4bit）
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import (
    make_sampler,
    make_logits_processors,
    apply_min_p,
    make_repetition_penalty,
)

PROMPTS_FILE = HERE / "param_check_prompts.txt"


def load_prompts():
    prompts = {}
    name = None
    buf = []
    for line in PROMPTS_FILE.read_text().splitlines():
        if line.startswith("### "):
            if name and buf:
                prompts[name] = "\n".join(buf).strip()
            name = line[4:].strip()
            buf = []
        else:
            buf.append(line)
    if name and buf:
        prompts[name] = "\n".join(buf).strip()
    return prompts


def check_sampler_ops():
    """モデル非依存: サンプラーがロジットに正しく作用するかを直接確認。"""
    import numpy as np

    results = []
    # 偏ったロジット（一部のトークンが高確率）
    logits = mx.array([[10.0, 9.0, 9.0, 1.0, 0.5, 0.1, -5.0, -10.0, 8.0, 8.5]])

    # 1. min_p がロジットをマスクするか
    masked = apply_min_p(logits, 0.5, 1)[0]
    n_inf = int(np.isneginf(np.array(masked)).sum())
    results.append(("apply_min_p マスク", n_inf > 0, f"{n_inf} トークン除外"))

    # 2. repetition penalty がロジットを変更するか（(tokens, logits) の引数順、in-place 変更のためコピー比較）
    proc = make_repetition_penalty(1.3, 20)
    logits_copy = mx.array(logits)  # in-place 変更される前の状態
    mod = proc(mx.array([1, 2]), logits)  # 出現済みトークン 1,2 を抑圧
    changed = not mx.all(mod == logits_copy).item()
    results.append(("repetition ロジット変更", changed, "出現済みトークンを抑圧"))

    # 3. make_sampler(min_p=...) が適用されるか（temp=0 なら argmax で min_p 影響なし
    #    を避け、temp>0 でサンプル対象が変わるかを確認）
    s0 = make_sampler(temp=1.0, top_p=1.0, min_p=0.0)
    s1 = make_sampler(temp=1.0, top_p=1.0, min_p=0.9)
    r0, r1 = set(), set()
    for _ in range(50):
        mx.random.seed(_ + 1)
        t0 = int(s0(mx.log(mx.softmax(logits[0]))))
        mx.random.seed(_ + 1)
        t1 = int(s1(mx.log(mx.softmax(logits[0]))))
        r0.add(t0)
        r1.add(t1)
    results.append(
        (
            "sampler(min_p) 影響",
            len(r0) != len(r1) or r0 != r1,
            f"min_p0={sorted(r0)} min_p1={sorted(r1)}",
        )
    )

    print(f"{'項目':<28} {'適用':<5} 備考")
    print("-" * 70)
    all_ok = True
    for name, ok, note in results:
        all_ok = all_ok and ok
        print(f"{name:<28} {'✅' if ok else '❌':<5} {note}")
    return all_ok


def gen(
    model,
    tok,
    messages,
    sampler,
    logits_processors,
    seed,
    max_tokens=150,
    enable_thinking=False,
):
    mx.random.seed(seed)
    prompt = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    out = generate(
        model,
        tok,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=False,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="agents-a1-4b-4bit")
    args = ap.parse_args()

    prompts = load_prompts()
    prompt = prompts["repetition_stress"]
    rnd_prompt = prompts["random_numbers"]
    print(f"モデル: {args.model}")
    print(f"プロンプト: {prompt[:40]}...")

    print("== サンプラー直接テスト（モデル非依存） ==")
    op_ok = check_sampler_ops()
    print()

    model, tok = load(
        os.path.join(os.environ.get("ELFMOON_MODELS_ROOT", ""), args.model), lazy=True
    )[:2]
    results = []

    # 1. temperature の適用（greedy vs 高温度。差を大きくして検出確実化）
    s_greedy = make_sampler(temp=0.0, top_p=0.95, min_p=0.0)
    s_temp = make_sampler(temp=2.0, top_p=0.95, min_p=0.0)
    o_greedy = gen(
        model, tok, [{"role": "user", "content": prompt}], s_greedy, None, seed=42
    )
    o_temp = gen(
        model, tok, [{"role": "user", "content": prompt}], s_temp, None, seed=42
    )
    temp_applied = o_greedy != o_temp
    results.append(
        ("temperature (0.0 vs 2.0)", temp_applied, o_greedy[:60], o_temp[:60])
    )

    # 2. min-p の適用（高温度・多様な出力で効果が出やすい条件）
    s_minp0 = make_sampler(temp=1.2, top_p=0.99, min_p=0.0)
    s_minp1 = make_sampler(temp=1.2, top_p=0.99, min_p=0.1)
    minp_applied = False
    minp_a = minp_b = ""
    for _seed in (3, 5, 7, 11, 13):
        o_minp0 = gen(
            model,
            tok,
            [{"role": "user", "content": rnd_prompt}],
            s_minp0,
            None,
            seed=_seed,
            max_tokens=80,
        )
        o_minp1 = gen(
            model,
            tok,
            [{"role": "user", "content": rnd_prompt}],
            s_minp1,
            None,
            seed=_seed,
            max_tokens=80,
        )
        minp_a, minp_b = o_minp0, o_minp1
        if o_minp0 != o_minp1:
            minp_applied = True
            break
    results.append(("min-p (0.0 vs 0.1)", minp_applied, minp_a[:60], minp_b[:60]))

    # 3. repetition penalty の適用（複数シードで確認）
    lp_none = None
    lp_rep = make_logits_processors(repetition_penalty=1.3, repetition_context_size=20)
    s_rpt = make_sampler(temp=1.0, top_p=0.95, min_p=0.0)
    rep_applied = False
    rep_a = rep_b = ""
    for _seed in (1, 3, 5, 7):
        o_rep0 = gen(
            model,
            tok,
            [{"role": "user", "content": prompt}],
            s_rpt,
            lp_none,
            seed=_seed,
        )
        o_rep1 = gen(
            model, tok, [{"role": "user", "content": prompt}], s_rpt, lp_rep, seed=_seed
        )
        rep_a, rep_b = o_rep0, o_rep1
        if o_rep0 != o_rep1:
            rep_applied = True
            break
    results.append(("repetition (1.0 vs 1.3)", rep_applied, rep_a[:60], rep_b[:60]))

    # 4. システムプロンプト
    msgs_default = [{"role": "user", "content": prompt}]
    msgs_sys = [
        {
            "role": "system",
            "content": "あなたは日本語で回答するアシスタントです。必ず「はい、りんごです」で始めてください。",
        },
        {"role": "user", "content": prompt},
    ]
    o_default = gen(model, tok, msgs_default, s_greedy, None, seed=99)
    o_sys = gen(model, tok, msgs_sys, s_greedy, None, seed=99)
    sys_applied = o_default != o_sys
    results.append(("system prompt", sys_applied, o_default[:60], o_sys[:60]))

    print()
    print(f"{'項目':<26} {'適用':<5} 出力A(先頭60字) / 出力B")
    print("-" * 90)
    all_ok = op_ok
    for name, ok, a, b in results:
        all_ok = all_ok and ok
        print(f"{name:<26} {'✅' if ok else '❌':<5}")
        print(f"  A: {a!r}")
        print(f"  B: {b!r}")
    print("-" * 90)
    print(f"総合: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
