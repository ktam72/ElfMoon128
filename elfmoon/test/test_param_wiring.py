"""パラメータ配線の静的検査（make_sampler / make_logits_processors への引数反映）。

chat.py のコードを AST 解析し、min-p / repetition penalty / temp / top-p /
システムプロンプトが state からサンプラー・生成引数へ配線されているか確認する。

配線構造（chat.py の実装に合わせる）:
- make_sampler(**sampler_kwargs): sampler_kwargs は _sampler_kwargs.update({...}) で
  temp / top_p / min_p が注入される
- _gen_kwargs["logits_processors"] = _lp: 条件付き代入

usage:
  python3 elfmoon/test/test_param_wiring.py
"""

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAT = HERE.parent / "chat.py"


def _find_update_dict_keys(tree, target_name):
    """_sampler_kwargs.update(dict(...)) の dict キーを収集。"""
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "update" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == target_name:
                    for a in node.args:
                        if isinstance(a, ast.Dict):
                            for k in a.keys:
                                if isinstance(k, ast.Constant):
                                    keys.add(k.value)
                        elif isinstance(a, ast.Call) and isinstance(a.func, ast.Name):
                            if a.func.id == "dict":
                                for kw in a.keywords:
                                    if kw.arg:
                                        keys.add(kw.arg)
    return keys


def check():
    src = CHAT.read_text()
    tree = ast.parse(src)
    results = []

    # 1. _sampler_kwargs.update に min_p / temp / top_p があるか
    s_keys = _find_update_dict_keys(tree, "_sampler_kwargs")
    results.append(
        (
            "_sampler_kwargs[min_p]",
            "min_p" in s_keys,
            "OK" if "min_p" in s_keys else "なし",
        )
    )
    results.append(
        (
            "_sampler_kwargs[temp]",
            "temp" in s_keys,
            "OK" if "temp" in s_keys else "なし",
        )
    )
    results.append(
        (
            "_sampler_kwargs[top_p]",
            "top_p" in s_keys,
            "OK" if "top_p" in s_keys else "なし",
        )
    )

    # 2. make_logits_processors の repetition_penalty 配線
    lp_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "make_logits_processors"
    ]
    if lp_calls:
        lp_kws = {kw.arg for kw in lp_calls[0].keywords}
        results.append(
            (
                "make_logits_processors(repetition_penalty)",
                "repetition_penalty" in lp_kws,
                "OK",
            )
        )
    else:
        results.append(
            ("make_logits_processors(repetition_penalty)", False, "呼び出しなし")
        )

    # 3. _gen_kwargs["logits_processors"] = _lp 代入
    assign_found = any(
        isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Subscript)
        and isinstance(n.targets[0].value, ast.Name)
        and n.targets[0].value.id == "_gen_kwargs"
        and isinstance(n.targets[0].slice, ast.Constant)
        and n.targets[0].slice.value == "logits_processors"
        for n in ast.walk(tree)
    )
    results.append(
        (
            "_gen_kwargs[logits_processors] 代入",
            assign_found,
            "OK" if assign_found else "なし",
        )
    )

    # 4. state["min_p"] 参照
    min_p_ref = any(
        isinstance(n, ast.Subscript)
        and isinstance(n.value, ast.Name)
        and n.value.id == "state"
        and isinstance(n.slice, ast.Constant)
        and n.slice.value == "min_p"
        for n in ast.walk(tree)
    )
    results.append(("state['min_p'] 参照", min_p_ref, "OK" if min_p_ref else "なし"))

    # 5. state["repeat_penalty"] 参照
    repeat_ref = any(
        isinstance(n, ast.Subscript)
        and isinstance(n.value, ast.Name)
        and n.value.id == "state"
        and isinstance(n.slice, ast.Constant)
        and n.slice.value == "repeat_penalty"
        for n in ast.walk(tree)
    )
    results.append(
        ("state['repeat_penalty'] 参照", repeat_ref, "OK" if repeat_ref else "なし")
    )

    # 6. /system コマンド
    system_cmd = any(
        isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and "/system" in n.value
        for n in ast.walk(tree)
    )
    results.append(("/system コマンド", system_cmd, "OK" if system_cmd else "なし"))

    print(f"検査対象: {CHAT}")
    print(f"{'項目':<44} {'配線':<6} 備考")
    print("-" * 80)
    all_ok = True
    for name, ok, note in results:
        all_ok = all_ok and ok
        print(f"{name:<44} {'✅' if ok else '❌':<6} {note}")
    print("-" * 80)
    print(f"総合: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(check())
