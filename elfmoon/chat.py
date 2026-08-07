"""ElfMoon 対話型チャットCLI。

モデルを一度だけロード＆ストリーミング化し、以降は対話ループで
何度でもプロンプトを投げられる。会話履歴を保持する。

使い方:
    cd elfmoon
    python3 chat.py                       # 常駐容量はメモリ予算から自動 (既定モデル, 既定 no-think)
    python3 chat.py --model 80b           # ELFMOON_MODELS_ROOT/80b を使用
    python3 chat.py --model 80b 1200      # モデル指定 + 省メモリ
    python3 chat.py --think               # 思考プロセスを表示（既定は no-think）
    python3 chat.py --no-think            # 思考プロセスを非表示（後方互換）
    python3 chat.py --fast                # 高速モード（top_k=6、実測~1.4-1.6x、品質トレードオフ）
    python3 chat.py --list                # 利用可能なモデル一覧

環境変数 ELFMOON_TOP_K=N でも指定可（--fast より優先、ストリーミング MoE のみ有効）。
環境変数 ELFMOON_MIN_P / ELFMOON_REPEAT で min-p / repetition penalty の既定を上書き可
（既定 0.05 / 1.15。対話中は /min-p /repeat で変更）。
"""

import logging
import fcntl
import os
import re
import select
import sys
import termios
import threading
import time
import tty
import unicodedata

import mlx.core as mx

# プロジェクトルートをパスに追加
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# 一部モデルのカスタムtokenizer実装が動作に無関係なWARNINGログを出すため抑制する
# （例: Kimi-Linearの tokenization_kimi.py が encode() 呼び出しごとに警告ログを出す）。
logging.disable(logging.WARNING)

from mlx_lm import load, stream_generate
from mlx_lm.utils import load_model
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache
from stream_model import MODELS_ROOT, list_models, resolve_model, wire_streaming
from pathlib import Path

SYSTEM = "You are an expert coding assistant. Write clean, correct, concise code."
MAX_TOKENS = 16384
MAX_HISTORY = 8
TEMP = 0.4

GREEN = "\033[1;32m"
MAGENTA = "\033[1;35m"
CYAN = "\033[1;36m"
DIM = "\033[2m"
RESET = "\033[0m"

# min-p / repetition penalty の既定値（ElfMoonMetal 相当）。環境変数で上書き可。
MIN_P = float(os.environ.get("ELFMOON_MIN_P", "0.05"))
REPEAT_PENALTY = float(os.environ.get("ELFMOON_REPEAT", "1.15"))
REPEAT_CONTEXT = 20


def use_color():
    if os.environ.get("NO_COLOR") is not None:
        return False
    term = os.environ.get("TERM", "")
    return any(k in term for k in ("color", "xterm", "screen", "tmux", "ansi"))


HELP = """コマンド一覧:
  /exit               終了（Ctrl+C / Ctrl+D でも終了）
  /help               この一覧
  /clear              会話履歴をクリア
  /n <数>             生成上限を変更
  /temp <値>          temperature を変更
  /top-p <値>         top-p を変更
  /top-k <数>         top-k を変更（0 で無効）
  /min-p <値>         min-p を変更（0〜1、0 で無効）
  /repeat <値>        repetition penalty を変更（1 以下で無効）
  /think              思考モードを有効化
  /nothink            思考モードを無効化（既定）
  /system <テキスト>  システムプロンプトを変更
"""
# プレフィルのチャンク幅。stream_generate 既定(512)は融合gather経路の閾値未満で
# 高速化されないため大きくする。api_server.py と同じ環境変数で連動。
PREFILL_STEP = int(os.environ.get("ELFMOON_PREFILL_STEP", "4096"))
# KV キャッシュ永続化（api_server と同じ kv_manager を利用）。ELFMOON_KVC=0 で無効。
KVC = os.environ.get("ELFMOON_KVC", "1") != "0"
# 対話 CLI では KVC の情報ログがプロンプト表示に割り込むため既定で抑制（エラーは出る）
os.environ.setdefault("ELFMOON_KVC_LOG", "0")


def _read_utf8_char(fd):
    b0 = os.read(fd, 1)
    if not b0:
        return None
    c0 = b0[0]
    if c0 < 0x80:
        return chr(c0)
    if c0 < 0xC0:
        return "\ufffd"
    need = 2 if c0 < 0xE0 else 3 if c0 < 0xF0 else 4
    raw = b0
    for _ in range(need - 1):
        more = os.read(fd, 1)
        if not more:
            break
        raw += more
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "\ufffd"


def _read_esc_seq(fd):
    """Read escape sequence after ESC (consumed). Returns bytes after ESC.
    CSI: ESC [ <params> <final> → returns b'[<params><final>'
    SS3: ESC O <final>       → returns b'O<final>'
    Two-char: ESC <final>    → returns b'<final>'
    Each byte read has 50ms timeout to avoid consuming user keystrokes."""
    r, _, _ = select.select([fd], [], [], 0.05)
    if not r:
        return None
    first = os.read(fd, 1)
    if not first:
        return None
    if first == b"[":  # CSI: read until final byte (0x40-0x7E)
        seq = first
        for _ in range(15):
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                break
            cb = os.read(fd, 1)
            if not cb:
                break
            seq += cb
            if cb[0] in range(0x40, 0x7F):
                break
        return seq
    if first == b"O":  # SS3: read one more byte
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            cb = os.read(fd, 1)
            if cb:
                return first + cb
        return first
    if first[0] in range(0x40, 0x7F):  # Two-char sequence
        return first
    return first


def _read_until_paste_end(fd):
    """Read and echo all characters until \x1b[201~ (bracketed paste end).
    Returns the pasted content."""
    chars = []
    while True:
        ch = _read_utf8_char(fd)
        if ch is None:
            raise EOFError
        if ch == "\x1b":
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                seq = _read_esc_seq(fd)
                if seq == b"[201~":
                    return "".join(chars)
            chars.append(ch)
            continue
        if ch == "\x7f":
            if chars:
                chars.pop()
            continue
        chars.append(ch)
        sys.stdout.write(ch)
        sys.stdout.flush()


def _char_width(ch):
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _clear_chars(fd, chars):
    sys.stdout.write("\b \b" * sum(_char_width(c) for c in chars))
    sys.stdout.flush()


def _read_line_raw(fd, history=None, hist_idx=None):
    """Read one line in raw mode. Returns the line (without newline).

    history: 過去の入力リスト（↑↓キーで参照）。hist_idx: 現在の位置（list で共有）。
    """
    chars = []
    while True:
        ch = _read_utf8_char(fd)
        if ch is None:
            raise EOFError
        if ch in "\n\r":
            print()
            return "".join(chars)
        if ch == "\x7f":
            if chars:
                removed = chars.pop()
                sys.stdout.write("\b \b" * _char_width(removed))
                sys.stdout.flush()
            continue
        if ch == "\x1b":
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                seq = _read_esc_seq(fd)
                if seq == b"[200~":
                    paste_content = _read_until_paste_end(fd)
                    return paste_content
                if history and seq in (b"[A", b"[B"):
                    # 現在行を退避してから履歴へ移動
                    cur = "".join(chars)
                    if seq == b"[A":  # 上: 過去へ
                        n = hist_idx[0] + 1
                        if n < len(history):
                            hist_idx[0] = n
                        else:
                            continue
                    else:  # 下: 未来へ
                        n = hist_idx[0] - 1
                        if n >= 0:
                            hist_idx[0] = n
                        else:
                            hist_idx[0] = -1
                            _clear_chars(fd, chars)
                            chars = []
                            continue
                    hist = history[-(hist_idx[0] + 1)]
                    _clear_chars(fd, chars)
                    chars = list(hist)
                    sys.stdout.write(hist)
                    sys.stdout.flush()
                    continue
                # Other CSI/escape sequences — discard
            elif chars:
                sys.stdout.write("\b \b" * sum(_char_width(c) for c in chars))
                sys.stdout.flush()
                chars = []
            continue
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x04":
            raise EOFError
        # Tab is handled explicitly (isprintable=False). All other non-control
        # characters — including full-width space (U+3000, isprintable=False in
        # Python for Unicode Separator category) — are accepted here.
        if ch == "\t" or ch.isprintable() or ch.isspace():
            chars.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()


def read_user_input(prompt, history=None, hist_idx=None):
    """Read user input in raw mode. Detects paste instantly, ESC to clear.

    history/hist_idx: ↑↓キー履歴ナビゲーション用（main ループで管理）。
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(prompt, end="", flush=True)
    try:
        tty.setcbreak(fd)
        a = termios.tcgetattr(fd)
        a[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG)
        termios.tcsetattr(fd, termios.TCSANOW, a)

        first = _read_line_raw(fd, history, hist_idx)

        # Drain with progressive timeout: 100ms initial, then 200ms per chunk
        extra_data = b""
        r, _, _ = select.select([fd], [], [], 0.1)
        while r:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            extra_data += chunk
            r, _, _ = select.select([fd], [], [], 0.2)

        # Second pass: if we got data, wait 400ms for any trailing chunks
        if extra_data:
            r, _, _ = select.select([fd], [], [], 0.4)
            while r:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                extra_data += chunk
                r, _, _ = select.select([fd], [], [], 0.2)

        if extra_data:
            extra = extra_data.decode("utf-8", errors="replace").split("\n")
            if extra[-1] == "":
                extra = extra[:-1]
            lines = [first] + extra if first else extra
            for l in extra:
                print(f"  {l}")
            print("\033[2m（空行で確定）\033[0m")
            while True:
                # Drain background paste data before each confirmation read
                r, _, _ = select.select([fd], [], [], 0)
                while r:
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        break
                    extra_data += chunk
                    r, _, _ = select.select([fd], [], [], 0)
                if extra_data:
                    more_parts = (
                        extra_data.decode("utf-8", errors="replace")
                        .rstrip("\n")
                        .split("\n")
                    )
                    extra_data = b""
                    for mp in more_parts:
                        lines.append(mp)
                        print(f"  {mp}")
                more = _read_line_raw(fd, history, hist_idx)
                if not more:
                    break
                lines.append(more)
            return "\n".join(lines).strip()

        if not first:
            return None
        return first.strip()

    except (KeyboardInterrupt, EOFError):
        raise
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSANOW, old)
        except:
            pass


class EscCancelMonitor:
    """生成中に stdin を監視し、ESC キーで生成を中断する（tty のみ有効）。

    SIGINT ではなく共有フラグ(cancelled)を立てる方式。MLX カーネル実行中は
    SIGINT 配送が遅延するため、ジェネレータ側でフラグをチェックして中断する。
    生成中は read_user_input が tty を canonical に復元しているため、監視中は
    非カノニカル(cbreak)に切り替えて 1 バイト読みを可能にする。
    """

    def __init__(self):
        self.cancelled = False
        self._stop = False
        self._thread = None
        self._fd = sys.stdin.fileno()
        self._saved_attr = None

    def start(self):
        if not sys.stdin.isatty():
            return
        self._stop = False
        self.cancelled = False
        self._saved_attr = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        fd = self._fd
        while not self._stop:
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                try:
                    ch = os.read(fd, 1)
                except OSError:
                    return
                if ch == b"\x1b":
                    # 以降の入力（ESC シーケンスの残り）を読み捨て
                    try:
                        for _ in range(3):
                            r2, _, _ = select.select([fd], [], [], 0.05)
                            if not r2:
                                break
                            os.read(fd, 1)
                    except OSError:
                        pass
                    self.cancelled = True
                    self._stop = True
                    return
                elif ch in (b"\x03", b"\x04"):
                    # Ctrl+C / Ctrl+D は後段の通常処理に任せる
                    pass

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._saved_attr is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._saved_attr)
            except Exception:
                pass
            self._saved_attr = None


def _strip_think(text_iter, no_think, in_think=False, c="", r=""):
    """Strip <think>/</think> blocks from stream if no_think is set.

    以下3形式に対応:
    1. <think>...</think> 形式（Qwen標準、templateが開きタグを出力）
    2. 開きタグ無しで推論内容→</think> 形式（一部fine-tuneモデル）
    3. in_think=True: Thinking専用モデル（templateがプロンプト側に <think> を
       置くため出力にタグが無い）。最初から think 内として </think> まで捨てる。

    no_think=False のとき、思考ブロックはシアン(c)で表示し、回答は標準色(r)で表示する。
    """
    if not no_think:
        # 思考をシアン、回答を標準色で表示（render_stream 相当）
        buf = ""
        in_think = in_think  # プロンプト側に <think> がある場合は最初から思考中
        if in_think:
            yield c
        for piece in text_iter:
            buf += piece
            while True:
                if not in_think and "<think>" in buf:
                    idx = buf.index("<think>")
                    pre, buf = buf[:idx], buf[idx + len("<think>") :]
                    if pre:
                        yield pre
                    yield c
                    in_think = True
                    continue
                if in_think and "</think>" in buf:
                    idx = buf.index("</think>")
                    think_part, buf = buf[:idx], buf[idx + len("</think>") :]
                    if think_part:
                        yield think_part
                    yield r
                    in_think = False
                    continue
                think_prefixes = ("<", "<t", "<th", "<thi", "<thin", "<think")
                end_prefixes = ("</", "</t", "</th", "</thi", "</thin", "</think")
                marker_prefix = think_prefixes if not in_think else end_prefixes
                if any(buf.endswith(p) for p in marker_prefix):
                    break
                if buf:
                    yield buf
                    buf = ""
                break
        if buf:
            yield buf
        if in_think:
            yield r
        return

    if in_think:
        buf = ""
        for piece in text_iter:
            buf += piece
            if "</think>" in buf:
                after = buf.split("</think>", 1)[1]
                if after.strip():
                    yield after.lstrip("\n")
                yield from text_iter
                return
            # </think> がタグ途中で分割されて届く場合に備え末尾だけ保持
            buf = buf[-16:]
        return

    buf = ""
    for piece in text_iter:
        buf += piece
        # 開きタグがあれば以降を discard
        if "<think" in buf:
            buf = buf[buf.find("<think") + len("<think>") :]
            while "</think>" not in buf:
                buf = next(text_iter, "")
                if not buf:
                    return
            after = buf.split("</think>", 1)[1]
            if after:
                yield after
            yield from text_iter
            return
        if "</think>" in buf:
            after = buf.split("</think>", 1)[1]
            if after:
                yield after
            yield from text_iter
            return
        yield buf
        buf = ""
    if buf:
        yield buf


def _think_kwargs(model_type, no_think):
    """思考モード抑制のテンプレート引数を返す（引数名がモデルで異なる）。

    Kimi K3 は独自トークナイザ(encoding_k3)で引数名が `thinking`。汎用の
    `enable_thinking` は **kwargs に落ちて黙って無視されるため、--no-think が
    効かない。thinking=False で think チャネル自体がプロンプトから消える。
    """
    if model_type == "kimi_k3":
        return {"thinking": not no_think}
    return {"enable_thinking": not no_think}


# Kimi K3 のチャネル制御トークン。表示上のノイズなので出力から取り除く。
_K3_CTRL = re.compile(
    r"<\|close\|>(?:think|response|message)?(?:<\|sep\|>)?"
    r"|<\|open\|>(?:think|response|message)?(?:<\|sep\|>)?"
    r"|<\|sep\|>|<\|end_of_msg\|>"
)


def _warn_if_too_large_without_store(model_path, store_dir):
    """store 未検出のままオンメモリ動作に入ると危険な規模なら警告する。

    store を別ドライブに置いた構成でパスを取り違えると、ストリーミングが黙って
    無効化され巨大モデルを丸ごとロードして OOM kill される（原因が分かりにくい）。
    """
    try:
        from resident_cache import budget_bytes_from_env

        total = sum(
            os.path.getsize(os.path.join(model_path, f))
            for f in os.listdir(model_path)
            if f.endswith(".safetensors")
        )
        budget = budget_bytes_from_env()
    except Exception:
        return
    if total <= budget:
        return
    print(
        f"\n⚠️  store/ が見つからないためストリーミングを使いません: {store_dir}\n"
        f"   モデル {total / 1024**3:.0f}GB > 利用可能 {budget / 1024**3:.0f}GB のため、"
        f"このままでは強制終了(OOM)する可能性が高いです。\n"
        f"   ELFMOON_STORE_DIR / ELFMOON_STORE_ROOT128 の指定、または "
        f"integrate.py split_all による store 生成を確認してください。\n",
        flush=True,
    )


def main():
    argv = sys.argv[1:]

    if "--list" in argv:
        models = list_models()
        print(f"利用可能なモデル（ELFMOON_MODELS_ROOT={MODELS_ROOT}）:")
        for name, has_store, is_native in models:
            if is_native:
                print(f"  {name}  ✅ オンメモリ動作")
            elif has_store:
                print(f"  {name}")
            else:
                print(f"  {name}  ⚠️ store/ 未生成（integrate.py split_all が必要）")
        if not models:
            print("  (見つかりません)")
        return

    # 既定は no-think（思考非表示）。--think で思考モード有効化（--no-think は後方互換）。
    no_think = "--think" not in argv
    perf = "--perf" in argv
    fast = "--fast" in argv
    no_color = "--no-color" in argv
    system_arg = None
    if "--system" in argv:
        _si = argv.index("--system")
        system_arg = argv[_si + 1]
        argv = argv[:_si] + argv[_si + 2 :]
    if fast and not os.environ.get("ELFMOON_TOP_K"):
        # \u5b9f\u6e2c ~1.4-1.6x\uff08\u54c1\u8cea\u30c8\u30ec\u30fc\u30c9\u30aa\u30d5\u3042\u308a\u30fbopt-in\uff09\u3002\u660e\u793a env \u304c\u3042\u308c\u3070\u305d\u3061\u3089\u3092\u512a\u5148\u3002
        os.environ["ELFMOON_TOP_K"] = "6"
    model_name = None
    if "--model" in argv:
        idx = argv.index("--model")
        model_name = argv[idx + 1].strip().replace("\u3000", "")
        argv = argv[:idx] + argv[idx + 2 :]
    cap_strs = [
        a
        for a in argv
        if a not in ("--think", "--no-think", "--perf", "--fast", "--no-color")
    ]
    cap = int(cap_strs[0]) if cap_strs else None  # None=メモリ予算から自動導出

    isatty = sys.stdin.isatty() and sys.stdout.isatty()
    use_c = isatty and use_color() and not no_color
    g, m, c, r, d = (
        (GREEN, MAGENTA, CYAN, RESET, DIM) if use_c else ("", "", "", "", "")
    )

    model_path, store_dir = resolve_model(model_name)
    if KVC:
        # KV キャッシュをモデル別に分離（モデル間の同一プロンプト衝突＝形状不一致を防ぐ）
        try:
            from kv_manager import kv_manager

            kv_manager.set_namespace(os.path.basename(model_path))
        except Exception:
            pass
    import json

    with open(os.path.join(model_path, "config.json")) as f:
        _cfg = json.load(f)
    _model_type = _cfg.get("model_type", "")

    _sampler_kwargs = {}
    _model_name = os.path.basename(model_path).lower()
    if _model_type == "gemma4":
        TEMP = 1.0
        _sampler_kwargs = dict(temp=TEMP, top_p=0.95, top_k=64)
    elif "ornith" in _model_name:
        # Ornith 推奨: agentic coding temp=1.0, top_p=1.0
        TEMP = 1.0
        _sampler_kwargs = dict(temp=TEMP, top_p=1.0, top_k=64)
    elif "glm" in _model_name:
        # GLM 推奨: temp=1.0, top_p=0.95, min_p=0.01（repeat_penalty=1.0は未対応）
        TEMP = 1.0
        _sampler_kwargs = dict(temp=TEMP, top_p=0.95, min_p=0.01)
    elif "agents-a1" in _model_name:
        # Agents-A1 推奨: temp=0.85, top_p=0.95, top_k=20（presence_penalty は未対応）
        TEMP = 0.85
        _sampler_kwargs = dict(temp=TEMP, top_p=0.95, top_k=20)
    else:
        TEMP = 0.4

    mode = "性能" if perf else "省メモリ"
    print(f"モデル: {model_path}（type={_model_type}）")
    print(f"モデルをロード中...（{mode}モード, capacity={cap or 'auto'}）")
    t0 = time.perf_counter()

    if _model_type == "laguna" or "laguna" in os.path.basename(model_path).lower():
        from laguna_opt import Model as OptimizedLagunaModel, ModelArgs

        _mp_laguna = Path(model_path)

        def _get_laguna_classes(config):
            return OptimizedLagunaModel, ModelArgs

        model, _ = load_model(
            _mp_laguna, lazy=True, get_model_classes=_get_laguna_classes
        )
        mx.clear_cache()
        cache = None
        try:
            _, tok = load(model_path, lazy=True)
        except Exception:
            from transformers import PreTrainedTokenizerFast
            from tokenizers import Tokenizer

            tk = Tokenizer.from_file(str(_mp_laguna / "tokenizer.json"))
            tok = PreTrainedTokenizerFast(tokenizer_object=tk)
            ct_path = _mp_laguna / "chat_template.jinja"
            if ct_path.exists():
                tok.chat_template = ct_path.read_text()
            with open(_mp_laguna / "config.json") as __cfgf:
                __cfg = json.load(__cfgf)
            __eos_raw = __cfg.get("eos_token_id", 1)
            __eos_ids = __eos_raw if isinstance(__eos_raw, list) else [__eos_raw]
            tok.eos_token_id = __eos_ids[0]
            from mlx_lm.tokenizer_utils import TokenizerWrapper

            tok = TokenizerWrapper(tok, eos_token_ids=__eos_ids)
    else:
        _mp = Path(model_path)
        model, _ = load_model(_mp, lazy=True)
        # トークナイザのロード（カスタムtokenizer_class対応）
        # tokenizer.json を持たずカスタムコード経由のみのモデル（例: Kimi K3 の
        # encoding_k3.py/tiktoken）は mlx_lm.load() 内の AutoTokenizer 呼び出しが
        # trust_remote_code なしで対話プロンプトを出して止まるため、先に弾いて
        # trust_remote_code=True 経路へ直行する。
        if not (_mp / "tokenizer.json").exists():
            from transformers import AutoTokenizer
            from mlx_lm.tokenizer_utils import TokenizerWrapper

            _tok_hf = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            __eos_ids = getattr(_tok_hf, "eos_token_id", None)
            __eos_ids = __eos_ids if isinstance(__eos_ids, list) else [__eos_ids]
            tok = TokenizerWrapper(_tok_hf, eos_token_ids=__eos_ids)
        else:
            try:
                _, tok = load(model_path, lazy=True)
            except Exception:
                from transformers import PreTrainedTokenizerFast
                from tokenizers import Tokenizer

                tk = Tokenizer.from_file(str(_mp / "tokenizer.json"))
                tok = PreTrainedTokenizerFast(tokenizer_object=tk)
                ct_path = _mp / "chat_template.jinja"
                if ct_path.exists():
                    tok.chat_template = ct_path.read_text()
                # config.json から EOS トークンを動的設定
                import json

                with open(_mp / "config.json") as __cfgf:
                    __cfg = json.load(__cfgf)
                __eos_raw = __cfg.get("eos_token_id", 1)
                __eos_ids = __eos_raw if isinstance(__eos_raw, list) else [__eos_raw]
                tok.eos_token_id = __eos_ids[0]
                from mlx_lm.tokenizer_utils import TokenizerWrapper

                tok = TokenizerWrapper(tok, eos_token_ids=__eos_ids)
        # store の在処は resolve_model が決める（内蔵SSD等、モデルディレクトリ外も可）。
        # ここで model_path/store を直に見ると store 別置き構成でストリーミングが
        # 無効化され、巨大モデルを丸ごとロードして OOM kill される。
        if _model_type == "gemma4" or not os.path.isdir(store_dir):
            cache = None
            _warn_if_too_large_without_store(model_path, store_dir)
        else:
            cache, _ = wire_streaming(
                model, cap, perf=perf, store_dir=store_dir, model_path=model_path
            )

    # mx.compile: 全denseモデルを高速化（streaming MoE は store/ があるので除外）
    if not os.path.isdir(store_dir):
        try:
            mx.compile(model.__call__)
        except Exception:
            pass

    print(
        f"準備完了（{time.perf_counter() - t0:.0f}秒）。会話をどうぞ。'exit' か Ctrl-D で終了。\n"
    )

    messages = [{"role": "system", "content": system_arg or SYSTEM}]
    state = {
        "max_tokens": MAX_TOKENS,
        "temp": TEMP,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": MIN_P,
        "repeat_penalty": REPEAT_PENALTY,
        "thinking": not no_think,
    }
    input_history = []
    hist_idx = [-1]
    print(
        f"{d}思考モード: {'think' if state['thinking'] else 'nothink'} ／ "
        f"生成上限: {state['max_tokens']} トークン（/n <数> で変更） ／ "
        f"min-p {state['min_p']:.2f}（/min-p） ／ repeat {state['repeat_penalty']:.2f}（/repeat）{r}"
    )
    while True:
        try:
            user = read_user_input(f"\n{g}あなた>{r} ", input_history, hist_idx)
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break
        if user is None:
            continue
        if user.lower() in ("exit", "quit"):
            print("終了します。")
            break
        if not user:
            continue

        if user.startswith("/"):
            cmd, *rest = user.split(maxsplit=1)
            arg = rest[0].strip() if rest else ""
            if cmd == "/exit":
                print("終了します。")
                break
            elif cmd == "/help":
                print(HELP)
                continue
            elif cmd == "/clear":
                messages = [messages[0]]
                print("会話履歴をクリアしました")
                continue
            elif cmd == "/n":
                try:
                    state["max_tokens"] = max(1, int(arg))
                    print(f"生成上限: {state['max_tokens']} トークン")
                except ValueError:
                    print("使用法: /n <数>")
                continue
            elif cmd == "/temp":
                try:
                    state["temp"] = float(arg)
                    print(f"temperature: {state['temp']}")
                except ValueError:
                    print("使用法: /temp <値>")
                continue
            elif cmd == "/top-p":
                try:
                    state["top_p"] = float(arg)
                    print(f"top-p: {state['top_p']}")
                except ValueError:
                    print("使用法: /top-p <値>")
                continue
            elif cmd == "/top-k":
                try:
                    state["top_k"] = max(0, int(arg))
                    print(f"top-k: {state['top_k']}")
                except ValueError:
                    print("使用法: /top-k <数>（0 で無効）")
                continue
            elif cmd == "/min-p":
                try:
                    v = float(arg)
                    if 0 <= v <= 1:
                        state["min_p"] = v
                        print(f"min-p: {state['min_p']}")
                    else:
                        print("min-p は 0〜1 の範囲で指定してください")
                except ValueError:
                    print("使用法: /min-p <値>（0〜1、0 で無効）")
                continue
            elif cmd == "/repeat":
                try:
                    v = float(arg)
                    state["repeat_penalty"] = v
                    print(f"repeat: {state['repeat_penalty']}（1 以下で無効）")
                except ValueError:
                    print("使用法: /repeat <値>（1.1〜1.15 が目安）")
                continue
            elif cmd == "/think":
                state["thinking"] = True
                no_think = False
                print("思考モード: think")
                continue
            elif cmd == "/nothink":
                state["thinking"] = False
                no_think = True
                print("思考モード: nothink")
                continue
            elif cmd == "/system":
                if arg:
                    messages = [m for m in messages if m["role"] != "system"]
                    messages.insert(0, {"role": "system", "content": arg})
                    print("システムプロンプトを更新しました")
                else:
                    print("使用法: /system <テキスト>")
                continue
            else:
                print(f"不明なコマンド: {cmd}（/help で一覧）")
                continue

        input_history.append(user)
        hist_idx[0] = -1
        messages.append({"role": "user", "content": user})
        if len(messages) > 1 + MAX_HISTORY * 2:
            messages = [messages[0]] + messages[-MAX_HISTORY * 2 :]

        prompt = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **_think_kwargs(_model_type, no_think),
        )

        print(f"{m}ElfMoon>{r} ", end="", flush=True)
        resp, t, answer_t = "", time.perf_counter(), 0.0
        n_raw = [0]  # 全生成トークン数（think 含む）
        think_s = 0.0
        _prefill_n = 0
        _prefill_t0 = None
        _prefill_t1_ref = [None]
        _first_raw_t = [0.0]
        _kvc_save_ids, _kvc_snap = None, None

        try:
            # 動的プリフィル調整: 最終チャンクが小さくなりすぎないよう PREFILL_STEP を調整
            _prompt_ids_for_tune = tok.encode(prompt)
            _prompt_len = len(_prompt_ids_for_tune)
            from stream_model import optimal_prefill_step

            PREFILL_STEP = optimal_prefill_step(_prompt_len)
            _sampler_kwargs.update(
                dict(
                    temp=state["temp"],
                    top_p=state["top_p"],
                    min_p=state["min_p"],
                )
            )
            if state.get("top_k"):
                _sampler_kwargs["top_k"] = state["top_k"]
            _sampler = make_sampler(**_sampler_kwargs)
            # repetition penalty（投機デコードは ElfMoon128 に無いため常時適用可）
            _lp = None
            if state["repeat_penalty"] > 1:
                _lp = make_logits_processors(
                    repetition_penalty=state["repeat_penalty"],
                    repetition_context_size=REPEAT_CONTEXT,
                )
            _gen_kwargs = dict(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                max_tokens=state["max_tokens"],
                sampler=_sampler,
                prefill_step_size=PREFILL_STEP,
            )
            if _lp:
                _gen_kwargs["logits_processors"] = _lp
            _kvc_save_ids, _kvc_snap = None, None
            if KVC:
                # 会話履歴部分の KV を再利用し、毎ターンの全履歴再プレフィルを回避する。
                # 失敗時は従来経路にフォールバック（会話は止めない）。
                try:
                    from kv_manager import kv_manager

                    prompt_ids = tok.encode(prompt)
                    nogen = tok.apply_chat_template(
                        messages,
                        add_generation_prompt=False,
                        tokenize=False,
                        **_think_kwargs(_model_type, no_think),
                    )
                    nogen_ids = tok.encode(nogen)
                    boundary = 0
                    for bi in range(min(len(nogen_ids), len(prompt_ids))):
                        if prompt_ids[bi] != nogen_ids[bi]:
                            break
                        boundary = bi + 1
                    cached_cache, cached_len = kv_manager.lookup(prompt_ids, model)
                    if cached_cache is not None and cached_len < len(prompt_ids):
                        prompt_cache = cached_cache
                    else:
                        prompt_cache = make_prompt_cache(model)
                        cached_len = 0
                    _prefill_n = len(prompt_ids) - cached_len
                    _prefill_t0 = time.perf_counter()
                    if cached_len < boundary:
                        remaining = prompt_ids[cached_len:boundary]
                        for ci in range(0, len(remaining), PREFILL_STEP):
                            model(
                                mx.array([remaining[ci : ci + PREFILL_STEP]]),
                                cache=prompt_cache,
                            )
                        _kvc_snap = kv_manager.snapshot(prompt_cache)
                        _kvc_save_ids = prompt_ids[:boundary]
                    if cached_len:
                        print(
                            f"{d}（KVC: {cached_len}tok 再利用）{r} ",
                            end="",
                            flush=True,
                        )
                    _gen_kwargs.update(
                        prompt=prompt_ids[boundary:], prompt_cache=prompt_cache
                    )
                except Exception:
                    pass  # 従来経路（全プレフィル）で続行
            if _prefill_n == 0:
                # 従来経路: stream_generate 内で全プレフィル
                prompt_ids = tok.encode(prompt) if isinstance(prompt, str) else prompt
                _prefill_n = len(prompt_ids) if isinstance(prompt_ids, list) else 0
                _prefill_t0 = time.perf_counter()
            generator = stream_generate(**_gen_kwargs)

            # Thinking専用モデル: template がプロンプト側に <think> を置き
            # 出力に開きタグが現れないため、最初から think 内として扱う
            _gp = _gen_kwargs["prompt"]
            if isinstance(_gp, str):
                _in_think = _gp.rstrip().endswith("<think>")
            else:
                _tail = tok.decode(list(_gp[-8:])) if len(_gp) else ""
                _in_think = _tail.rstrip().endswith("<think>")

            _first_raw_t = [0.0]
            esc_mon = EscCancelMonitor()

            def _texts():
                for out in generator:
                    if _first_raw_t[0] == 0.0:
                        _first_raw_t[0] = time.perf_counter()
                        if _prefill_t1_ref[0] is None:
                            _prefill_t1_ref[0] = _first_raw_t[0]
                    if esc_mon.cancelled:
                        return
                    n_raw[0] += 1
                    yield out.text

            if _in_think:
                print(f"{d}（思考中…）{r}", end="", flush=True)
            esc_mon.start()
            try:
                for piece in _strip_think(
                    _texts(), no_think, in_think=_in_think, c=c, r=r
                ):
                    if _model_type == "kimi_k3":
                        # チャネル制御トークンは表示上のノイズなので落とす
                        piece = _K3_CTRL.sub("", piece)
                        if not piece:
                            continue
                    if answer_t == 0.0:
                        answer_t = time.perf_counter()
                        if _in_think:
                            # 思考中表示を消して回答を書き始める
                            print(f"\r\033[K{m}ElfMoon>{r} ", end="", flush=True)
                        piece = piece.lstrip("\n")  # 回答冒頭の空行を除去
                        if not piece:
                            answer_t = 0.0  # 空白のみなら次piece を冒頭扱い
                            continue
                    print(piece, end="", flush=True)
                    # 会話履歴用テキストには色コードを含めない
                    if piece and piece not in (c, r):
                        resp += piece.replace(c, "").replace(r, "")
            finally:
                esc_mon.stop()
            if esc_mon.cancelled:
                print(f"\n{d}（ESC で生成を中断しました）{r}")
            if n_raw and n_raw[0] >= state["max_tokens"]:
                print(
                    f"\n{r}{d}⚠ 生成上限 {state['max_tokens']} トークンに達して中断しました。"
                    f"（/n <数> で上げて再質問）{r}"
                )
            # 隠した思考の時間（応答 t/s の分母には乗せない）
            think_s = (
                (answer_t - _first_raw_t[0])
                if (_in_think and answer_t and _first_raw_t[0])
                else 0.0
            )
            if _kvc_save_ids is not None:
                try:
                    from kv_manager import kv_manager

                    kv_manager.save(_kvc_save_ids, _kvc_snap)
                except Exception:
                    pass
        except Exception as e:
            print(f"\n\033[1;31m[エラー] 生成が中断されました: {e}\033[0m")

        elapsed = (time.perf_counter() - _first_raw_t[0]) if _first_raw_t[0] else 0.0
        # 学術的定義（vLLM/LMDeploy）準拠の計測:
        #   prefill = 入力トークン処理（model 実行開始〜最初の出力トークン確定）
        #   TTFT    = リクエスト開始（t）〜最初のトークン表示（_first_raw_t[0]）
        #   decode  = 最初のトークン以降の逐次生成（全トークン数 n_raw / その時間）
        #   （従来は answer_t 以降の回答のみで測っていたが、思考型モデルで server 経路と
        #     区間がずれるため、全トークン基準に統一）
        pf_info = ""
        _ft = _first_raw_t[0] if _first_raw_t else 0.0
        if (
            _prefill_t0 is not None
            and _prefill_t1_ref[0] is not None
            and _prefill_t1_ref[0] > _prefill_t0
        ):
            pf_time = _prefill_t1_ref[0] - _prefill_t0
            if pf_time > 0:
                pf_speed = _prefill_n / pf_time
                ttft_ms = (_ft - t) * 1000 if _ft else 0.0
                pf_info = (
                    f"prefill {_prefill_n}tok {pf_speed:.0f}tok/s ／ "
                    f"TTFT {ttft_ms:.0f}ms ／ "
                )
        hit = f", 命中率{cache.hit_rate * 100:.0f}%" if cache else ""
        think_info = f"思考 {think_s:.1f}s ／ " if think_s > 0 else ""
        _n_raw = n_raw[0] if n_raw else 0
        _tps = (_n_raw / elapsed) if elapsed > 0 else 0.0
        print(
            f"\n\033[2m（{think_info}{pf_info}decode {_n_raw} tokens, {_tps:.1f} tok/s{hit}）\033[0m"
        )
        messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()
