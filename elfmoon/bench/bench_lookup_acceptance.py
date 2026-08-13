"""prompt-lookup 受容率のオフライン模擬（ElfMoonCoder bench_lookup_acceptance 相当）。

実際に greedy 生成した系列に対し、lookup_gen.lookup_candidate が何トークン提案し、
何トークン一致したかを数えて受容率と理論高速化を出す。
投機は貪欲デコード等価なので、生成結果は投機の有無で変わらない。

think / no-think 両モードで測り、修正（--no-think 反映）が lookup の
効果に影響しないことを確認する。

usage:
  python3 bench/bench_lookup_acceptance.py [モデル名]
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # elfmoon/
sys.path.insert(0, str(ROOT))
os.environ["ELFMOON_KVC"] = "0"
os.environ["ELFMOON_KVC_LOG"] = "0"

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from lookup_gen import lookup_candidate, stream_generate_lookup  # noqa: E402

GEN_TOKENS = 300
BLOCK = 16


def simulate(prompt_ids, gen_ids):
    """生成済み系列に対して prompt-lookup を模擬し、統計を返す。"""
    seq = list(prompt_ids)
    i = 0
    forwards = 0
    accepted_total = 0
    proposals = 0
    while i < len(gen_ids):
        cand = lookup_candidate(seq, k=min(BLOCK, len(gen_ids) - i))
        if not cand:
            seq.append(gen_ids[i])
            i += 1
            forwards += 1
            continue
        proposals += 1
        actual = gen_ids[i : i + len(cand)]
        acc = 0
        for a, b in zip(cand, actual):
            if a != b:
                break
            acc += 1
        take = min(acc + 1, len(gen_ids) - i)
        seq.extend(gen_ids[i : i + take])
        i += take
        forwards += 1
        accepted_total += acc
    return {
        "gen": len(gen_ids),
        "forwards": forwards,
        "accepted": accepted_total,
        "proposals": proposals,
        "spec_ratio": forwards / max(len(gen_ids), 1),
    }


def report(label, st):
    acc_rate = st["accepted"] / max(st["gen"], 1)
    # 1 forward のコストを 1.0 とすると、通常は gen 回、投機は forwards 回
    base_cost = st["gen"] * 1.0
    spec_cost = st["forwards"] * 1.0
    speedup = base_cost / spec_cost if spec_cost else 1.0
    print(
        f"  {label:28} 受容率 {acc_rate:5.1%}  "
        f"forward {st['forwards']}/{st['gen']}  "
        f"提案 {st['proposals']}  → 理論 {speedup:4.2f}x"
    )
    return acc_rate, speedup


def main():
    from stream_model import MODELS_ROOT

    name = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit"
    )
    model_path = os.path.join(MODELS_ROOT, name)
    model, tok = load(model_path, lazy=True)
    sampler = make_sampler(temp=0.0, top_p=1.0, min_p=0.0)

    # タスク: コード反復（受容率が高くなる人工タスク）
    target = (ROOT / "lookup_gen.py").read_text()
    tasks = {
        "コード編集(小さい): docstring追加": [
            {
                "role": "user",
                "content": f"次のファイルを編集してください。\n\n```python\n{target[:3000]}\n```\n\n各関数の先頭に日本語 docstring を追加したファイル全体を出力してください。",
            }
        ],
        "白紙からのコード生成": [
            {
                "role": "user",
                "content": "ブロック崩しの HTML ゲームを作ってください。日本語でコメントを書いてください。",
            }
        ],
        "日本語会話": [
            {
                "role": "user",
                "content": "メモリ帯域律速について、初心者にも分かるように説明してください。",
            }
        ],
    }

    print(f"モデル: {name} / 生成 {GEN_TOKENS} トークン / block={BLOCK}\n")

    for thinking in (True, False):
        label = "think" if thinking else "no-think"
        print(f"[{label} モード]")
        for tlabel, msgs in tasks.items():
            p = tok.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=thinking,
            )
            ids = tok.encode(p)
            pc = make_prompt_cache(model)
            text = ""
            for r in stream_generate_lookup(
                model,
                tok,
                ids,
                max_tokens=GEN_TOKENS,
                sampler=sampler,
                prompt_cache=pc,
                prefill_step_size=4096,
                enable_lookup=False,
            ):
                text += r.text
            gen_ids = tok.encode(text, add_special_tokens=False)
            report(f"{tlabel} (prompt {len(ids)})", simulate(ids, gen_ids))
        print()


if __name__ == "__main__":
    main()
