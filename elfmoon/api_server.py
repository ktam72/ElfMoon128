"""ElfMoon OpenAI 互換 API サーバ（generation-thread 方式）。

POST /v1/chat/completions   (stream/non-stream, OpenAI 互換)
POST /v1/completions        (OpenAI テキスト補完)
POST /v1/responses          (OpenAI Responses API)
POST /v1/messages           (Anthropic 互換)
GET  /v1/models

これにより Claude Code / VS Code Continue / Cursor / Zed / Open Interpreter 等の
OpenAI 互換 API をサポートする全ツールから ElfMoon を使える。

使い方:
    python3 api_server.py [port] [resident_capacity] [--model NAME] [--no-think]
    python3 api_server.py --list                      # 利用可能なモデル一覧

    デフォルト: port=11434, capacity=auto(メモリ予算から導出), バインド先=127.0.0.1, model=ELFMOON_MODEL(既定qwen3.6-35b-mlx)
    （LAN に公開する場合のみ ELFMOON_HOST=0.0.0.0 を指定。認証は無いので注意）
    モデル置き場は ELFMOON_MODELS_ROOT で指定（既定 ../models）。各モデルは
    <ELFMOON_MODELS_ROOT>/<name>/ に元重み一式 + integrate.py が作る store/ を持つ。

    curl http://localhost:11434/v1/chat/completions \\
      -d '{"model":"qwen3.6-35b","messages":[{"role":"user","content":"SwiftでFizzBuzzを書いて"}],"stream":true}'

Claude Code から使う場合 (~/.clauderc.json):
    {
      "models": [{
        "name": "elfmoon",
        "provider": "openai",
        "model": "qwen3.6-35b",
        "apiKey": "sk-not-needed",
        "baseUrl": "http://localhost:11434/v1"
      }]
    }
"""

import json

import logging
import os
from pathlib import Path
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from queue import Queue
from socketserver import ThreadingMixIn
from threading import Thread, Event as ThreadEvent
from urllib.parse import urlparse

logging.disable(logging.WARNING)

import mlx.core as mx
from kv_manager import kv_manager
from mcp_client import mcp_manager, MCPError
from mlx_lm import load as _mlx_load
from mlx_lm.utils import load_model
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import (
    make_frequency_penalty,
    make_presence_penalty,
    make_sampler,
)
from stream_model import MODELS_ROOT, list_models, resolve_model, wire_streaming
from tool_replay import ToolReplayStore, tool_replay_path
from lookup_gen import LookupOut, stream_generate_lookup

HOST = os.environ.get("ELFMOON_HOST", "127.0.0.1")
DEFAULT_PORT = 11434
DEFAULT_CAPACITY = None  # None=メモリ予算から自動導出
MODEL_ID = "elfmoon"
# 出力トークン上限。思考型モデル（Nemotron 等）はリーズニング込みで長い応答になるため
# 余裕を持たせる。クライアントが max_tokens 未指定時の既定値でもある。
MAX_TOKENS = 65536
TEMP = 0.6
# プレフィルのチャンク幅。gather_qmm 経路では融合テンソル読込(~18GB/チャンク巡回)が
# チャンク数に比例する固定費のため、大きいほど長プロンプトで有利。8192 は活性化で
# ピーク ~21.7GB に達し 24GB 機では危険なため既定 4096（ピーク ~5GB）。
PREFILL_STEP = int(os.environ.get("ELFMOON_PREFILL_STEP", "4096"))
NO_THINK = "--no-think" in sys.argv


class ThinkStripper:
    """<think> ブロックをストリームから除去する（リクエスト毎に生成すること）。"""

    _PEEK = len("<think>")

    def __init__(self):
        self._buf = ""
        self._skip = True
        self._peeking = True

    def feed(self, piece):
        if not self._skip:
            return piece
        self._buf += piece
        if self._peeking:
            if len(self._buf) < self._PEEK and "<think>".startswith(self._buf):
                return None
            self._peeking = False
            if not self._buf.lstrip().startswith("<think"):
                self._skip = False
                out, self._buf = self._buf, ""
                return out if out else None
        idx = self._buf.find("</think>")
        if idx >= 0:
            self._skip = False
            after = self._buf[idx + 8 :]
            self._buf = ""
            return after if after else None
        return None

    @property
    def pending(self):
        return self._buf if self._skip else ""


class ReasoningSplitter:
    """<think>...</think> を reasoning / content に分けてストリームする。

    ThinkStripper が思考を**除去**するのに対し、本クラスは思考を reasoning として
    分離し、<think> が閉じた後を content として返す。in_think=True（プロンプト末尾
    が <think>）なら最初から reasoning とみなす。

    feed(piece) は (reasoning_piece, content_piece) を返す。いずれも None の場合あり。
    ストリーム末尾で pending に未確定分が残る。
    """

    _THINK_OPEN = "<think>"
    _THINK_CLOSE = "</think>"

    def __init__(self, in_think=False):
        self._buf = ""
        self._skip = True
        self._peeking = not in_think
        self._started = in_think

    def feed(self, piece):
        if not self._skip:
            return None, piece
        self._buf += piece
        if self._peeking:
            # 開きタグを待つ（部分一致の間は保持）
            stripped = self._buf.lstrip()
            if not stripped or self._THINK_OPEN.startswith(stripped):
                return None, None
            self._peeking = False
            if stripped.startswith("<think"):
                # 開きタグを除去して reasoning モードへ
                self._buf = stripped[len(self._THINK_OPEN) :]
            else:
                self._skip = False
                out, self._buf = self._buf, ""
                return None, (out if out else None)
        # reasoning 中: </think> を探す
        idx = self._buf.find(self._THINK_CLOSE)
        if idx >= 0:
            self._skip = False
            reasoning = self._buf[:idx]
            after = self._buf[idx + len(self._THINK_CLOSE) :]
            self._buf = ""
            return (reasoning if reasoning else None), (after if after else None)
        # 確定済み reasoning を送出（末尾 8 文字は </think> の可能性があるため保留）
        keep = max(0, len(self._buf) - len(self._THINK_CLOSE))
        if keep > 0:
            out = self._buf[:keep]
            self._buf = self._buf[keep:]
            return out, None
        return None, None

    @property
    def pending(self):
        return self._buf if self._skip else ""


# ---- generation engine（専用スレッドでモデルを動かす） ----


TOOL_CALL_START = "<|tool_call|>"
TOOL_CALL_END = "<tool_call|>"

_TC_START = re.escape(TOOL_CALL_START)
_TC_END = re.escape(TOOL_CALL_END)

# 開始マーカーは実モデルで <|tool_call> と <|tool_call|> の両形が観測されるため、
# 末尾 | を任意にした正規表現で許容する（片側欠けによる抽出漏れを防ぐ）。
_TC_START_RE = re.compile(r"<\|tool_call\|?>")

# セクション形式の tool call マーカー（例: <|tool_call_begin|>{...}<|tool_call_end|>）。
# セクション開始/終了マーカーは実モデルで <| 始まりと |< 始まりの両形が観測されるため、
# 正規表現で許容する。
TOOL_CALL_BEGIN_MARKER = "<|tool_call_begin|>"
TOOL_CALL_END_MARKER = "<|tool_call_end|>"
_SECTION_MARKER_RE = re.compile(r"[<|]{1,2}tool_calls_section_(?:begin|end)\|>")

# Laguna 等の tool call 形式（chat_template 準拠）:
# <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
TOOL_CALL_LEGACY_START = "<tool_call>"
TOOL_CALL_LEGACY_END = "</tool_call>"
_ARG_KEY_START = "<arg_key>"
_ARG_KEY_END = "</arg_key>"
_ARG_VALUE_START = "<arg_value>"
_ARG_VALUE_END = "</arg_value>"


def _match_brace(text: str, pos: int) -> int:
    """text[pos] が '{' の場合、対応する '}' の位置+1 を返す。"""
    assert text[pos] == "{"
    depth = 1
    i = pos + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return i


def _parse_gemma4_args(text: str) -> dict:
    """Gemma4 の call:func{key:val,key2:"val2"} 形式の引数を dict に変換する。"""
    import ast

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        pass
    # 簡易パース: {key: "val", key2: 123} 形式
    # str.strip("{}") はネストした {} を壊すため使わない
    result = {}
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    stripped = s.strip()
    if not stripped:
        return result
    buf = stripped
    while buf:
        buf = buf.lstrip().lstrip(",").lstrip()
        if not buf:
            break
        # キー（引用符なし識別子 or 引用符あり文字列）
        if buf[0] in ('"', "'"):
            end = buf.find(buf[0], 1)
            key = buf[1:end]
            buf = buf[end + 1 :]
        else:
            m = re.match(r"(\w+)", buf)
            if not m:
                break
            key = m.group(1)
            buf = buf[m.end() :]
        buf = buf.lstrip()
        if buf and buf[0] == ":":
            buf = buf[1:]
        buf = buf.lstrip()
        # 値
        val, buf = _parse_gemma4_value(buf)
        result[key] = val
    return result


def _parse_gemma4_value(buf: str) -> tuple:
    """Gemma4 形式の値を1つパースして (value, rest) を返す。"""
    buf = buf.lstrip()
    if not buf:
        return None, ""
    if buf[0] in ('"', "'"):
        end = buf.find(buf[0], 1)
        val = buf[1:end]
        rest = buf[end + 1 :]
        # 引用符内のエスケープ
        val = val.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
        return val, rest
    if buf[0].isdigit() or buf[0] == "-":
        m = re.match(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", buf)
        if m:
            raw = m.group(1)
            rest = buf[m.end() :]
            if "." in raw or "e" in raw.lower():
                return float(raw), rest
            return int(raw), rest
    if buf.startswith("true"):
        return True, buf[4:]
    if buf.startswith("false"):
        return False, buf[5:]
    if buf.startswith("none"):
        return None, buf[4:]
    if buf.startswith("null"):
        return None, buf[4:]
    if buf[0] == "{":
        depth = 1
        i = 1
        while i < len(buf) and depth > 0:
            if buf[i] == "{":
                depth += 1
            elif buf[i] == "}":
                depth -= 1
            i += 1
        inner = buf[1 : i - 1]
        rest = buf[i:]
        return _parse_gemma4_args(f"{{{inner}}}"), rest
    if buf[0] == "[":
        depth = 1
        i = 1
        items = []
        while i < len(buf) and depth > 0:
            if buf[i] == "[":
                depth += 1
            elif buf[i] == "]":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            elif buf[i] == "," and depth == 1:
                i += 1
                continue
            elif buf[i] == ",":
                i += 1
                continue
            i += 1
        rest = buf[i:]
        return items, rest
    # 引用符なしの識別子（リテラル文字列）
    m = re.match(r"([^,}\]]+)", buf)
    if m:
        val = m.group(1).strip().rstrip(",").rstrip("}").rstrip("]")
        rest = buf[m.end() :]
        return val, rest
    return None, buf[1:]


# Gemma4 tokenizer の特殊トークン置換マップ
_TOKEN_ARTIFACTS = {
    re.escape('<|"|>'): '"',
    re.escape("<|'|>"): "'",
    re.escape("<|\n|>"): "\n",
    re.escape("<|\r|>"): "\r",
    re.escape("<|\t|>"): "\t",
}


def _clean_token_artifacts(text: str) -> str:
    for pattern, replacement in _TOKEN_ARTIFACTS.items():
        text = re.sub(pattern, replacement, text)
    return text


def _strip_channels(text: str) -> str:
    """gemma のチャンネル形式 <|channel>NAME...<channel|>CONTENT を除去する。

    モデルの chat_template 自身の除去ロジックに準拠:
      text を <channel|> で分割し、<|channel> を含む part は <|channel> より前だけ残す。
    これで思考チャンネル（<|channel>thought...）とマーカーが消え、最終回答のみ残る。
    チャンネルマーカーを含まないテキスト（Qwen 等）は素通し。
    """
    if "<|channel>" not in text and "<channel|>" not in text:
        return text
    result = []
    for part in text.split("<channel|>"):
        if "<|channel>" in part:
            result.append(part.split("<|channel>")[0])
        else:
            result.append(part)
    return "".join(result)


def _parse_tool_json_body(body: str) -> dict | None:
    """tool_call ボディの OpenAI JSON 形式 {"name":..., "arguments":...} を dict に変換する。"""
    json_m = re.match(r"\s*(\{)", body)
    if not json_m:
        return None
    close = _match_brace(body, json_m.start(1))
    if close <= 0:
        return None
    raw = body[json_m.start(1) : close]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    name = data.get("name") or data.get("function", {}).get("name", "")
    args = data.get("arguments") or data.get("function", {}).get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _parse_legacy_tool_body(body: str) -> dict | None:
    """Laguna 形式のボディ <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call> を dict に変換する。"""
    # 関数名は <arg_key> の前までのテキスト（<tool_call> 直後。引数がない場合は全体）
    key_pos = body.find(_ARG_KEY_START)
    name_part = body if key_pos == -1 else body[:key_pos]
    m = re.match(r"\s*([^\s<]+)", name_part)
    if not m:
        return None
    name = m.group(1)
    rest = body[m.end() :] if key_pos == -1 else body[key_pos:]
    args = {}
    while True:
        ks = rest.find(_ARG_KEY_START)
        if ks == -1:
            break
        ke = rest.find(_ARG_KEY_END, ks)
        vs = rest.find(_ARG_VALUE_START, ke)
        if ke == -1 or vs == -1:
            break
        ve = rest.find(_ARG_VALUE_END, vs)
        if ve == -1:
            break
        key = rest[ks + len(_ARG_KEY_START) : ke]
        raw_val = rest[vs + len(_ARG_VALUE_START) : ve]
        args[key] = _coerce_legacy_value(raw_val)
        rest = rest[ve + len(_ARG_VALUE_END) :]
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _coerce_legacy_value(raw: str):
    """Laguna の arg_value（文字列 or JSON 化された値）を Python 値に変換する。"""
    s = raw.strip()
    if s.startswith('"') and s.endswith('"'):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return raw
    if s == "true":
        return True
    if s == "false":
        return False
    if s == "null":
        return None
    m = re.fullmatch(r"-?\d+", s)
    if m:
        return int(s)
    m = re.fullmatch(r"-?\d+\.\d+", s)
    if m:
        return float(s)
    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return raw
    return raw


def _extract_legacy_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Laguna 形式（<tool_call>name<arg_key>...</arg_key>...</tool_call>）を抽出する。

    Nemotron 形式（<tool_call>\n<function=name>\n<parameter=k>\nv\n</parameter>...</function>
    </tool_call>）もここで扱う（`_parse_function_parameter_body` 参照）。
    """
    if TOOL_CALL_LEGACY_START not in text:
        return text, []
    calls = []
    cleaned_parts = []
    i = 0
    while True:
        begin = text.find(TOOL_CALL_LEGACY_START, i)
        if begin == -1:
            cleaned_parts.append(text[i:])
            break
        cleaned_parts.append(text[i:begin])
        end = text.find(TOOL_CALL_LEGACY_END, begin)
        if end == -1:
            cleaned_parts.append(text[begin:])
            break
        body = text[begin + len(TOOL_CALL_LEGACY_START) : end]
        # qwen3 などは <tool_call> 内に JSON オブジェクト {"name":...,"arguments":...} を入れる。
        # Laguna 形式（<arg_key> タグ）よりも先に JSON 形式を試す。
        parsed = _parse_tool_json_body(body)
        if parsed is None:
            parsed = _parse_legacy_tool_body(body)
        if parsed is None:
            # Nemotron 形式: <function=name>...<parameter=k>...</parameter>...</function>
            parsed = _parse_function_parameter_body(body)
        if parsed is not None:
            calls.append(parsed)
            cleaned_parts.append("")
        else:
            cleaned_parts.append(text[begin : end + len(TOOL_CALL_LEGACY_END)])
        i = end + len(TOOL_CALL_LEGACY_END)
    cleaned = "".join(cleaned_parts).strip()
    return cleaned, calls


def _parse_function_parameter_body(body: str) -> dict | None:
    """Nemotron 形式のボディ <function=name>\n<parameter=k>\nv\n</parameter>...</function> を dict に変換する。

    例:
      <function=discover_projs>
      <parameter=workspaceRoot>
      .
      </parameter>
      </function>
    """
    m = re.search(r"<function=([^>\n]+)>", body)
    if not m:
        return None
    name = m.group(1).strip()
    args = {}
    # <parameter=k>...</parameter> ブロックを全て抽出
    for pm in re.finditer(r"<parameter=([^>\n]+)>(.*?)</parameter>", body, re.DOTALL):
        key = pm.group(1).strip()
        raw = pm.group(2)
        # 前後の空行を除去（テンプレート描画では \n が含まれる）
        val = raw.strip("\n")
        args[key] = _coerce_legacy_value(val)
    if not args and "<parameter=" not in body:
        # 引数なし（関数名のみ）
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _extract_section_tool_calls(text: str) -> tuple[str, list[dict]]:
    """セクション形式の tool call（<|tool_call_begin|>...</tool_call|>）を抽出する。

    例:
      |<tool_calls_section_begin|>\n<|tool_call_begin|>{"name":...,"arguments":...}<|tool_call_end|>
    gemma 系の <|tool_call|> マーカーは含まないため、_TC_START_RE では拾えない。
    """
    if TOOL_CALL_BEGIN_MARKER not in text:
        return text, []
    calls = []
    cleaned_parts = []
    i = 0
    while True:
        begin = text.find(TOOL_CALL_BEGIN_MARKER, i)
        if begin == -1:
            cleaned_parts.append(text[i:])
            break
        cleaned_parts.append(text[i:begin])
        content_start = begin + len(TOOL_CALL_BEGIN_MARKER)
        end = text.find(TOOL_CALL_END_MARKER, content_start)
        if end == -1:
            cleaned_parts.append(text[begin:])
            break
        body = text[content_start:end].strip()
        parsed = _parse_tool_json_body(body)
        if parsed is not None:
            calls.append(parsed)
            cleaned_parts.append("")
        else:
            cleaned_parts.append(text[begin : end + len(TOOL_CALL_END_MARKER)])
        i = end + len(TOOL_CALL_END_MARKER)
    cleaned = "".join(cleaned_parts).strip()
    # セクションマーカーも除去する
    cleaned = _SECTION_MARKER_RE.sub("", cleaned).strip()
    return cleaned, calls


def _extract_tool_calls(text: str) -> tuple[str, list[dict]]:
    """テキストから tool_call ブロックを抽出し、(マーカー除去済みテキスト, tool_call リスト) を返す。"""
    # セクション形式（<|tool_call_begin|>）を優先して処理
    if TOOL_CALL_BEGIN_MARKER in text:
        clean, calls = _extract_section_tool_calls(text)
        # マーカーはあるが解析できなかった場合は通常パスへ委譲せず、
        # マーカーを除去したテキストを返す（ループ継続での失敗を避ける）
        return clean, calls

    # Laguna 形式（<tool_call>name<arg_key>...</arg_key>...</tool_call>）
    if TOOL_CALL_LEGACY_START in text:
        clean, calls = _extract_legacy_tool_calls(text)
        return clean, calls

    calls = []
    cleaned_parts = []
    i = 0

    while i < len(text):
        # 次の開始マーカー（<|tool_call> / <|tool_call|>）を探す
        m = _TC_START_RE.search(text, i)
        if m is None:
            cleaned_parts.append(text[i:])
            break
        start = m.start()

        # 開始マーカー以前のテキストを保存
        cleaned_parts.append(text[i:start])
        content_start = m.end()

        # <tool_call|> 終了マーカーを探す
        end = text.find(TOOL_CALL_END, content_start)
        if end == -1:
            cleaned_parts.append(text[i:])
            break

        body = text[content_start:end].strip()
        call_end = end + len(TOOL_CALL_END)

        parsed = None

        # Gemma4 形式: call:func_name{args}
        gemma4_m = re.match(r"call:(\w+)\s*(\{)", body)
        if gemma4_m:
            name = gemma4_m.group(1)
            brace_pos = content_start + gemma4_m.start(2)
            close = _match_brace(text, brace_pos)
            if close > 0:
                raw_text = text[brace_pos:close]
                args = _parse_gemma4_args(raw_text)
                parsed = {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
                # call_end は END マーカーの後（line 293）を維持する。close（}位置）で
                # 上書きすると <tool_call|> が cleaned テキストに残ってしまう。

        # OpenAI JSON 形式: {"name":..., "arguments":...}
        if parsed is None:
            json_m = re.match(r"\s*(\{)", body)
            if json_m:
                brace_pos = content_start + json_m.start(1)
                close = _match_brace(text, brace_pos)
                if close > 0:
                    raw = text[brace_pos:close]
                    try:
                        data = json.loads(raw)
                        name = data.get("name") or data.get("function", {}).get(
                            "name", ""
                        )
                        args = data.get("arguments") or data.get("function", {}).get(
                            "arguments", {}
                        )
                        if isinstance(args, str):
                            args = json.loads(args)
                        parsed = {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        # call_end は END マーカーの後（line 293）を維持
                    except (json.JSONDecodeError, TypeError):
                        pass

        if parsed is not None:
            calls.append(parsed)
            i = call_end
        else:
            cleaned_parts.append(text[start:call_end])
            i = call_end

    cleaned = "".join(cleaned_parts).strip()
    return cleaned, calls


def _normalize_tool_args(messages: list) -> list:
    """受信メッセージの tool_calls[].function.arguments（OpenAI 標準の JSON 文字列）を
    dict に変換する。Laguna 等の chat_template は arguments に .items() を回すため、
    文字列のままだとテンプレート描画が UndefinedError になる。"""
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        msg = dict(msg)
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            norm = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    norm.append(tc)
                    continue
                tc = dict(tc)
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn = dict(fn)
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            fn["arguments"] = {}
                    tc["function"] = fn
                norm.append(tc)
            msg["tool_calls"] = norm
        out.append(msg)
    return out


# ---- reasoning 分割（<think>...</think>） ----


def _split_reasoning(text: str) -> tuple[str, str]:
    """出力テキストを (reasoning, content) に分割する。

    <think> が先頭にある場合: その中のテキストを reasoning に、
    残りを content にする。閉じタグが無い場合は先頭 <think> 以降を
    reasoning、content は空（未終了 reasoning）。
    """
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end != -1:
            return text[7:end], text[end + 8 :]
        return text[7:], ""
    end = text.find("</think>")
    if end != -1:
        return text[:end], text[end + 8 :]
    return "", text


def _contains_think(text: str) -> bool:
    return "<think>" in text or "</think>" in text


# ---- tool マーカー検出（ストリーミング用） ----

_TOOL_START_MARKERS = ["<tool_call>", "<|tool_call|>", "<|tool_call>"]
_TOOL_END_MARKERS = ["</tool_call>", "<tool_call|>", "<tool_call|>"]


def _pending_marker_len(text: str) -> int:
    """text 末尾が tool 開始マーカーの途中かもしれない場合、その文字数を返す。

    ストリーミング中に `<tool_` まで来た時点で content として送出してしまうと、
    直後に完成したマーカーを取り消せない。そのため「マーカーの真プレフィックスに
    一致する末尾」だけを送出保留する。固定長で後ろ倒しすると、マーカーが現れない
    通常の回答でも末尾が欠落するため、必要最小限に絞る。
    """
    best = 0
    for m in _TOOL_START_MARKERS:
        for k in range(min(len(m) - 1, len(text)), best, -1):
            if text.endswith(m[:k]):
                best = k
                break
    return best


def _first_tool_marker(text: str) -> int | None:
    """text 内の最初の tool 開始マーカーの位置（文字インデックス）を返す。無ければ None。"""
    pos = None
    for m in _TOOL_START_MARKERS:
        i = text.find(m)
        if i != -1 and (pos is None or i < pos):
            pos = i
    return pos


def _tool_call_complete(text: str) -> bool:
    """tool_call の閉じマーカーが揃ったかを判定する（Swift toolCallComplete 相当）。"""
    if TOOL_CALL_LEGACY_START in text and TOOL_CALL_LEGACY_END in text:
        return True
    if TOOL_CALL_START in text and TOOL_CALL_END in text:
        return True
    return False


# ---- content ブロック正規化（OpenAI/Anthropic 形式） ----


def _normalize_message(msg: dict) -> dict:
    """messages[].content が文字列またはブロック配列の場合を正規化する。

    - OpenAI: content が文字列の場合はそのまま
    - Anthropic 系: content が配列（text / thinking / tool_use / tool_result）
    - 戻り値: content(str) / reasoning(str) / tool_calls(list) / tool_call_id(str?)

    role:tool（function）は content が文字列 or tool_result ブロック。
    """
    role = msg.get("role", "user")
    content = msg.get("content")
    reasoning = msg.get("reasoning") or ""
    tool_calls = msg.get("tool_calls") or []
    tool_call_id = msg.get("tool_call_id")

    if isinstance(content, str):
        return {
            "role": role,
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
        }

    if not isinstance(content, list):
        return {
            "role": role,
            "content": "",
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
        }

    text_parts = []
    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        btype = block.get("type", "text")
        if btype in ("text", "input_text", "output_text"):
            text_parts.append(block.get("text") or "")
        elif btype in ("thinking", "reasoning"):
            reasoning += (
                block.get("thinking") or block.get("summary") or block.get("text") or ""
            )
        elif btype == "tool_use":
            name = block.get("name") or ""
            args = block.get("input") or {}
            cid = block.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            tool_calls.append(
                {
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        elif btype == "tool_result":
            tool_call_id = block.get("tool_use_id") or tool_call_id
            result = block.get("content")
            if isinstance(result, list):
                result = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in result
                )
            text_parts.append(result or "")
        else:
            text_parts.append(block.get("text") or "")

    return {
        "role": role,
        "content": "".join(text_parts),
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
    }


def _normalize_messages(messages: list) -> list:
    """messages 全体を正規化し、tool_calls の arguments を dict 化する。"""
    out = []
    for msg in messages:
        nm = _normalize_message(msg)
        if nm["tool_calls"]:
            nm = _normalize_tool_args([nm])[0]
        out.append(nm)
    return out


def _reasoning_enabled(thinking, think, reasoning_effort) -> bool:
    """thinking / think / reasoning_effort から reasoning を有効にするか判定する。

    - reasoning_effort == "none"|"minimal" → False
    - thinking が bool → その値
    - thinking が {type: "disabled"} → False
    - think が bool → その値
    - 既定 → True
    """
    if reasoning_effort:
        if reasoning_effort.lower() in ("none", "minimal"):
            return False
    if isinstance(thinking, bool):
        return thinking
    if isinstance(thinking, dict):
        if thinking.get("type") == "disabled":
            return False
        return True
    if think is not None:
        return bool(think)
    return True


class GenerationEngine:
    """モデルを専用スレッドで保持し、リクエストを直列化して generation する。"""

    def __init__(self, model_path: str, store_dir: str, cap: int | None, perf: bool):
        self._queue = Queue()
        self._ready = ThreadEvent()
        self._thread = Thread(target=self._run, daemon=True)
        self._model_path = model_path
        self._store_dir = store_dir
        self._cap = cap
        self._perf = perf
        self._model = None
        self._tokenizer = None
        self._moe_cache = None
        self._model_type = ""
        self._model_name = ""
        self._tool_replay = ToolReplayStore(filepath=tool_replay_path(""))
        # --no-think 起動時: クライアントが thinking 未指定なら思考無効（既定値）
        self.default_no_think = False

        self._thread.start()
        self._ready.wait()

    def generate(
        self,
        messages: list,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMP,
        no_think: bool = False,
        tools: list | None = None,
        stop=None,
        seed=None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        reasoning_effort: str | None = None,
    ):
        cancel = ThreadEvent()
        q: Queue = Queue()
        self._queue.put(
            (
                "messages",
                q,
                cancel,
                messages,
                max_tokens,
                temperature,
                no_think,
                tools,
                stop,
                seed,
                frequency_penalty,
                presence_penalty,
                top_p,
                top_k,
                min_p,
                reasoning_effort,
            )
        )
        try:
            while True:
                msg = q.get()
                if msg is None:
                    break
                if isinstance(msg, Exception):
                    raise msg
                if isinstance(msg, int):
                    yield msg
                    continue
                yield msg
                if cancel.is_set():
                    break
        except GeneratorExit:
            cancel.set()
            raise

    def generate_prompt(
        self,
        prompt: str,
        prompt_nogen: str,
        max_tokens: int,
        temperature: float,
        no_think: bool,
        stop=None,
        seed=None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ):
        cancel = ThreadEvent()
        q: Queue = Queue()
        self._queue.put(
            (
                "prompt",
                q,
                cancel,
                prompt,
                prompt_nogen,
                max_tokens,
                temperature,
                no_think,
                stop,
                seed,
                frequency_penalty,
                presence_penalty,
                top_p,
                top_k,
                min_p,
            )
        )
        try:
            while True:
                msg = q.get()
                if msg is None:
                    break
                if isinstance(msg, Exception):
                    raise msg
                if isinstance(msg, int):
                    yield msg
                    continue
                yield msg
                if cancel.is_set():
                    break
        except GeneratorExit:
            cancel.set()
            raise

    def _sampler_kwargs(
        self,
        temperature: float,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> dict:
        kwargs = {"temp": temperature}
        if self._model_type == "gemma4":
            kwargs["top_p"] = 0.95
            kwargs["top_k"] = 64
        elif "ornith" in self._model_name:
            kwargs["top_p"] = 1.0
            kwargs["top_k"] = 64
        elif "glm" in self._model_name:
            kwargs["top_p"] = 0.95
            kwargs["min_p"] = 0.01
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if min_p is not None:
            kwargs["min_p"] = min_p
        return kwargs

    def _eos_ids(self):
        """EOS トークン ID を集合として返す。

        一部のトークナイザーは eos_token_ids が int（単一）を返すため、
        `token in eos_ids` を安全に使えるよう常に set に正規化する。
        """
        raw = getattr(self._tokenizer, "eos_token_ids", None)
        if raw is None:
            raw = [self._tokenizer.eos_token_id]
        elif not isinstance(raw, (list, tuple, set)):
            raw = [raw]
        return set(raw)

    # ---- 以下、generation スレッド ---- #

    def _run(self):
        mx.eval(mx.array(0))
        mx.new_thread_local_stream(mx.default_device())
        self._load_model()
        mcp_manager.load()
        self._ready.set()
        err_count = 0
        while True:
            item = self._queue.get()
            req_type = item[0]
            if req_type == "messages":
                (
                    _dummy,
                    q,
                    cancel,
                    messages,
                    max_tokens,
                    temperature,
                    no_think,
                    tools,
                    stop,
                    seed,
                    frequency_penalty,
                    presence_penalty,
                    top_p,
                    top_k,
                    min_p,
                    reasoning_effort,
                ) = item
                gen = self._generate_impl(
                    messages,
                    max_tokens,
                    temperature,
                    no_think,
                    tools,
                    stop,
                    seed,
                    frequency_penalty,
                    presence_penalty,
                    top_p,
                    top_k,
                    min_p,
                    reasoning_effort,
                )
            elif req_type == "prompt":
                (
                    _dummy,
                    q,
                    cancel,
                    prompt,
                    prompt_nogen,
                    max_tokens,
                    temperature,
                    no_think,
                    stop,
                    seed,
                    frequency_penalty,
                    presence_penalty,
                    top_p,
                    top_k,
                    min_p,
                ) = item
                gen = self._generate_legacy(
                    prompt,
                    prompt_nogen,
                    max_tokens,
                    temperature,
                    no_think,
                    stop,
                    seed,
                    frequency_penalty,
                    presence_penalty,
                    top_p,
                    top_k,
                    min_p,
                )
            else:
                continue
            try:
                for msg in gen:
                    if cancel.is_set():
                        gen.close()
                        break
                    q.put(msg)
                err_count = 0
            except Exception as e:
                err_count += 1
                import traceback

                traceback.print_exc()
                q.put(Exception(str(e)))
            finally:
                q.put(None)

    def _load_model(self):
        mp = Path(self._model_path)
        with open(mp / "config.json") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")

        self._model_type = model_type
        self._model_name = mp.name.lower()
        self._tool_replay = ToolReplayStore(filepath=tool_replay_path(self._model_name))

        if model_type == "laguna" or "laguna" in mp.name.lower():
            from laguna_opt import Model as OptimizedLagunaModel, ModelArgs

            def _get_laguna_classes(config):
                return OptimizedLagunaModel, ModelArgs

            self._model, _ = load_model(
                mp, lazy=True, get_model_classes=_get_laguna_classes
            )
            mx.clear_cache()
            self._moe_cache = None
            # Tokenizer
            try:
                _tok_cfg = {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "add_prefix_space": False,
                }
                _, self._tokenizer = _mlx_load(
                    str(mp),
                    tokenizer_config=_tok_cfg,
                    lazy=True,
                )
            except Exception:
                from transformers import PreTrainedTokenizerFast
                from tokenizers import Tokenizer
                from mlx_lm.tokenizer_utils import TokenizerWrapper

                tk = Tokenizer.from_file(str(mp / "tokenizer.json"))
                self._tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk)
                ct_path = mp / "chat_template.jinja"
                if ct_path.exists():
                    self._tokenizer.chat_template = ct_path.read_text()
                with open(mp / "config.json") as f:
                    _eos_cfg = json.load(f)
                eos_ids = _eos_cfg.get("eos_token_id", [])
                if isinstance(eos_ids, list) and eos_ids:
                    self._tokenizer.eos_token_id = eos_ids[0]
                self._tokenizer = TokenizerWrapper(
                    self._tokenizer, eos_token_ids=eos_ids
                )
        else:
            from mlx_lm.utils import load_model as _lm_load

            self._model, _ = _lm_load(mp, lazy=True)
            try:
                _tok_cfg = {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "add_prefix_space": False,
                }
                _, self._tokenizer = _mlx_load(
                    str(mp),
                    tokenizer_config=_tok_cfg,
                    lazy=True,
                )
            except Exception:
                from transformers import PreTrainedTokenizerFast
                from tokenizers import Tokenizer
                from mlx_lm.tokenizer_utils import TokenizerWrapper

                tk = Tokenizer.from_file(str(mp / "tokenizer.json"))
                self._tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk)
                ct_path = mp / "chat_template.jinja"
                if ct_path.exists():
                    self._tokenizer.chat_template = ct_path.read_text()
                with open(mp / "config.json") as f:
                    _eos_cfg = json.load(f)
                eos_ids = _eos_cfg.get("eos_token_id", [])
                if isinstance(eos_ids, list) and eos_ids:
                    self._tokenizer.eos_token_id = eos_ids[0]
                self._tokenizer = TokenizerWrapper(
                    self._tokenizer, eos_token_ids=eos_ids
                )
            if model_type != "gemma4" and os.path.isdir(os.path.join(str(mp), "store")):
                self._moe_cache, _ = wire_streaming(
                    self._model,
                    self._cap,
                    perf=self._perf,
                    store_dir=self._store_dir,
                    model_path=str(mp),
                )
            else:
                pass  # mx.compile は現在の環境で遅くなるためスキップ

    def _generate_legacy(
        self,
        prompt,
        prompt_nogen,
        max_tokens,
        temperature,
        no_think,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
    ):
        """従来の高速パス: KV Cache 永続化 + 境界スナップショット対応。"""
        tokenizer = self._tokenizer
        model = self._model

        if seed is not None:
            mx.random.seed(seed)
        if stop is None:
            stop_list = []
        elif isinstance(stop, str):
            stop_list = [stop]
        else:
            stop_list = [s for s in stop if s]
        procs = []
        if frequency_penalty:
            procs.append(make_frequency_penalty(frequency_penalty, 64))
        if presence_penalty:
            procs.append(make_presence_penalty(presence_penalty, 64))

        prompt_ids = tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)
        # 動的プリフィル調整: 最終チャンクが小さくなりすぎないよう PREFILL_STEP を調整
        from stream_model import optimal_prefill_step

        PREFILL_STEP = optimal_prefill_step(prompt_tokens)
        yield prompt_tokens

        print(
            f"[ENGINE] prompt={prompt_tokens}tok max_tokens={max_tokens} temp={temperature}",
            file=sys.stderr,
            flush=True,
        )

        nogen_ids = tokenizer.encode(prompt_nogen)
        boundary = 0
        for i in range(min(len(nogen_ids), len(prompt_ids))):
            if prompt_ids[i] != nogen_ids[i]:
                break
            boundary = i + 1

        sampler = make_sampler(**self._sampler_kwargs(temperature, top_p, top_k, min_p))
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        eos_ids = self._eos_ids()
        stripper = ThinkStripper() if no_think else None
        # thinking 有効時は reasoning と content を分離してストリームする。
        # Nemotron（nemotron_h）も chat_template が生成プロンプト末尾に <think>\n
        # を付けるため、in_think=True で開始して </think> 以降を content とする。
        splitter = (
            ReasoningSplitter(in_think=prompt.rstrip().endswith("<think>"))
            if not no_think
            else None
        )

        cached_cache, cached_len = kv_manager.lookup(prompt_ids, model)

        if cached_cache is not None and cached_len < len(prompt_ids):
            prompt_cache = cached_cache
            print(
                f"[ENGINE] KVC hit offset={cached_len} new={len(prompt_ids) - cached_len}",
                file=sys.stderr,
                flush=True,
            )
        else:
            prompt_cache = make_prompt_cache(model)
            print(
                f"[ENGINE] KVC fresh (prompt={prompt_tokens})",
                file=sys.stderr,
                flush=True,
            )
            cached_len = 0

        save_key_ids = None
        snap = None
        prefill_t = time.time()
        if cached_len < boundary:
            remaining = prompt_ids[cached_len:boundary]
            step = PREFILL_STEP
            # チャンク境界ごとに退避点を記録（履歴の中盤書き換え時に共通 prefix
            # まで巻き戻して再利用するため）。
            kv_manager.set_live_cache(prompt_cache)
            kv_manager._session_tokens = list(prompt_ids[:cached_len])
            _acc = cached_len
            for i in range(0, len(remaining), step):
                chunk = remaining[i : i + step]
                model(mx.array([chunk]), cache=prompt_cache)
                _acc += len(chunk)
                kv_manager.add_snapshot(prompt_cache, _acc)
            snap = kv_manager.snapshot(prompt_cache)
            save_key_ids = prompt_ids[:boundary]
            print(
                f"[ENGINE] KVC history prefilled: {boundary - cached_len}tok in {time.time() - prefill_t:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        kv_manager.set_live_cache(prompt_cache)
        kv_manager._session_tokens = list(prompt_ids[:boundary])

        remaining_ids = prompt_ids[boundary:]
        if not remaining_ids:
            remaining_ids = [tokenizer.eos_token_id]

        generate_t = time.time()
        _ttft = None  # 最初のトークンが送出されるまでの時間（リクエスト受付から）
        # prompt-lookup 投機デコード: 貪欲（temp=0）時のみ出力が通常経路と等価。
        # 受容率が低いタスクでは lookup_gen 内部で自動フォールバックする。
        if temperature == 0.0 and not stop_list and not no_think and not procs:
            generator = stream_generate_lookup(
                model=model,
                tokenizer=tokenizer,
                prompt=remaining_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=prompt_cache,
                prefill_step_size=PREFILL_STEP,
                enable_lookup=True,
                kv_manager=kv_manager,
            )
        else:
            generator = generate_step(
                mx.array(remaining_ids),
                model,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=procs or None,
                prompt_cache=prompt_cache,
                prefill_step_size=PREFILL_STEP,
            )
        n = 0
        _sent = 0
        try:
            for _tok in generator:
                # 最初の送出: TTFT（リクエスト受付から最初のトークン送出まで）
                if _ttft is None:
                    _ttft = time.time() - generate_t
                # prompt-lookup 経路は LookupOut（.text に差分テキストを持つ）を
                # yield する。detokenizer は lookup_gen 内部で管理済みのため、
                # ここで add_token すると二重管理でテキストが壊れる。
                # 通常経路（generate_step）は (token, logprobs) タプルを yield し、
                # detokenizer 管理はこちらで行う。
                if isinstance(_tok, LookupOut):
                    piece = _tok.text
                    if not piece:
                        continue
                    n += 1
                    if stripper is not None:
                        piece = stripper.feed(piece)
                        if piece is None:
                            continue
                        yield (piece, n)
                    elif splitter is not None:
                        r, c = splitter.feed(piece)
                        if r:
                            yield {"reasoning": r}
                        if c:
                            yield (c, n)
                    else:
                        yield (piece, n)
                    continue
                token, _logprob = _tok
                if token in eos_ids:
                    break
                detokenizer.add_token(token)
                if stop_list:
                    full = detokenizer.text
                    cut = len(full)
                    for s in stop_list:
                        i = full.find(s)
                        if i != -1 and i < cut:
                            cut = i
                    if cut < len(full):
                        if cut > _sent:
                            piece = full[_sent:cut]
                            n += 1
                            if stripper is not None:
                                piece = stripper.feed(piece)
                                if piece is not None:
                                    yield (piece, n)
                            elif splitter is not None:
                                r, c = splitter.feed(piece)
                                if r:
                                    yield {"reasoning": r}
                                if c:
                                    yield (c, n)
                            else:
                                yield (piece, n)
                        break
                piece = detokenizer.last_segment
                if not piece:
                    continue
                _sent = len(detokenizer.text)
                n += 1
                if stripper is not None:
                    piece = stripper.feed(piece)
                    if piece is None:
                        continue
                    yield (piece, n)
                elif splitter is not None:
                    r, c = splitter.feed(piece)
                    if r:
                        yield {"reasoning": r}
                    if c:
                        yield (c, n)
                else:
                    yield (piece, n)
            if stripper is not None and stripper.pending:
                yield (stripper.pending, n)
            if splitter is not None and splitter.pending:
                yield {"reasoning": splitter.pending}
        except Exception as e:
            print(f"[ENGINE] error at token {n}: {e}", file=sys.stderr, flush=True)
            raise
        finally:
            _elapsed = time.time() - generate_t
            _ttft_ms = f"{_ttft * 1000:.0f}ms" if _ttft is not None else "n/a"
            print(
                f"[ENGINE] done: {n} tokens in {_elapsed:.1f}s"
                f" ({n / max(_elapsed, 1e-9):.1f} t/s)"
                f" TTFT {_ttft_ms}",
                file=sys.stderr,
                flush=True,
            )
            if save_key_ids is not None:
                kv_manager.save(save_key_ids, snap)

    def _apply_tool_replay(self, messages: list) -> list:
        """assistant tool_calls が全てい tool replay ストアに一致する場合、
        生ブロックに置換して返す（KV キャッシュの prefix 一致を壊さない）。

        - 一致時: assistant メッセージの content に生ブロックを連結し tool_calls を除去
        - 不一致: そのまま
        """
        if not self._tool_replay or not messages:
            return messages
        out = []
        for m in messages:
            calls = m.get("tool_calls")
            if m.get("role") == "assistant" and calls:
                block = self._tool_replay.exact_block(calls)
                if block is not None:
                    content = m.get("content") or ""
                    reasoning = m.get("reasoning") or ""
                    if reasoning:
                        content = f"<think>{reasoning}</think>{content}"
                    nm = dict(m)
                    nm["content"] = content + block
                    nm.pop("tool_calls", None)
                    out.append(nm)
                    continue
            out.append(m)
        return out

    def _generate_impl(
        self,
        messages,
        max_tokens,
        temperature,
        no_think,
        tools,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        reasoning_effort=None,
    ):
        """messages 経路（tools / completions / responses / messages）の生成。

        従来は全トークンを `output_ids` に溜めて最後に一括デコードしていたが、
        detokenizer の差分を逐次 yield するストリーミング方式に変更した。
        reasoning は `{"reasoning": ...}`、content は `(piece, n)`、
        tool_calls は `{"tool_calls": [...]}` として順次返す。
        tool 開始マーカー以降の content は送出しない（tool 領域をバッファ）。
        """
        tokenizer = self._tokenizer
        model = self._model
        eos_ids = self._eos_ids()

        if seed is not None:
            mx.random.seed(seed)
        if stop is None:
            stop_list = []
        elif isinstance(stop, str):
            stop_list = [stop]
        else:
            stop_list = [s for s in stop if s]
        procs = []
        if frequency_penalty:
            procs.append(make_frequency_penalty(frequency_penalty, 64))
        if presence_penalty:
            procs.append(make_presence_penalty(presence_penalty, 64))

        messages = _normalize_tool_args(list(messages))
        messages = self._apply_tool_replay(messages)

        if tools:
            messages = list(messages)
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Preserve file names exactly as given by the user. "
                        "Do NOT translate, romanize, or localize file names. "
                        "For example, if the user asks about 'マーメイド.mmd', "
                        "use 'マーメイド.mmd' literally, not 'mermaid.mmd'."
                    ),
                },
            )

        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=not no_think,
        )
        prompt_ids = tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)

        yield prompt_tokens
        # 動的プリフィル調整（1回目のみ）: 最終チャンクが小さくなりすぎないよう調整
        from stream_model import optimal_prefill_step

        PREFILL_STEP = optimal_prefill_step(prompt_tokens)
        print(
            f"[ENGINE] prompt={prompt_tokens}tok max_tokens={max_tokens} temp={temperature} tools={bool(tools)}",
            file=sys.stderr,
            flush=True,
        )

        prefill_t = time.time()
        prompt_cache = make_prompt_cache(model)
        for i in range(0, len(prompt_ids), PREFILL_STEP):
            model(mx.array([prompt_ids[i : i + PREFILL_STEP]]), cache=prompt_cache)
        # MLX は遅延評価のため、eval しないと prefill 時間が実質 0 になる
        try:
            mx.eval([c.state for c in prompt_cache])
        except Exception:
            pass
        prefill_elapsed = time.time() - prefill_t

        sampler = make_sampler(**self._sampler_kwargs(temperature, top_p, top_k, min_p))
        remaining = prompt_ids[-1:] if prompt_ids else [tokenizer.eos_token_id]

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        _sent = 0  # detokenizer.text の送出済み文字位置
        n = 0  # 生成トークン数（piece 数ではない）
        tool_started = False  # tool マーカー開始以降は content を出さない

        stripper = ThinkStripper() if no_think else None
        splitter = (
            ReasoningSplitter(in_think=prompt.rstrip().endswith("<think>"))
            if not no_think
            else None
        )

        def _emit_piece(piece, count):
            """piece を reasoning/content に分割して yield する。"""
            if stripper is not None:
                out = stripper.feed(piece)
                if out:
                    yield (out, count)
            elif splitter is not None:
                r, c = splitter.feed(piece)
                if r:
                    yield {"reasoning": r}
                if c:
                    yield (c, count)
            else:
                yield (piece, count)

        def _tool_scan_from(text: str) -> int | None:
            """tool マーカーの探索を開始してよい位置。think ブロック中なら None。

            思考中のモデルは tool_call の書式そのものについて言及する（「<tool_call>
            の中に <function=...> を入れる」等）。これを本物の tool_call と取り違え
            ると content の配信が止まり、偽の tool_call が組み立てられる。最終的な
            `_extract_tool_calls` は think 除去後のテキストを見るので保護済みだが、
            ストリーム中のラッチにも同じ保護が要る。
            """
            if splitter is None:
                return 0
            i = text.find(ReasoningSplitter._THINK_CLOSE)
            if i == -1:
                return None
            return i + len(ReasoningSplitter._THINK_CLOSE)

        def _tool_marker_at(text: str) -> int | None:
            """think ブロックより後ろにある最初の tool 開始マーカー位置。"""
            base = _tool_scan_from(text)
            if base is None:
                return None
            rel = _first_tool_marker(text[base:])
            return None if rel is None else base + rel

        def _tool_done(text: str) -> bool:
            """think ブロックより後ろで tool_call が閉じたか。"""
            base = _tool_scan_from(text)
            return base is not None and _tool_call_complete(text[base:])

        def _stop_cut(text: str) -> int | None:
            """stop_list のいずれかが text に現れる最初の位置。無ければ None。"""
            if not stop_list:
                return None
            cut = None
            for s in stop_list:
                if not s:
                    continue
                i = text.find(s)
                if i != -1 and (cut is None or i < cut):
                    cut = i
            return cut

        def _flush_content(final: bool = False):
            """detokenizer.text の未送出部分を送出する。

            tools 有効時は tool 開始マーカー以降を送出しない（保留 + 領域検出）。
            final=True（生成終了後の最終 flush）では保留を解除する。これ以上
            トークンは来ないため、部分マーカーとの取り違えは起こり得ない。
            """
            nonlocal _sent
            text = detokenizer.text
            if len(text) <= _sent:
                return
            limit = len(text)
            stop_at = _stop_cut(text)
            if stop_at is not None:
                limit = min(limit, stop_at)
            if tools and not tool_started:
                marker = _tool_marker_at(text)
                if marker is not None:
                    limit = min(limit, marker)
                elif not final:
                    # マーカーの途中かもしれない末尾だけを保留する
                    limit = min(limit, len(text) - _pending_marker_len(text))
            if limit <= _sent:
                return
            piece = _clean_token_artifacts(text[_sent:limit])
            _sent = limit
            if not piece:
                return
            for y in _emit_piece(piece, n):
                yield y

        generate_t = time.time()
        _ttft = None  # 最初のトークンが送出されるまでの時間（リクエスト受付から）
        generator = generate_step(
            mx.array(remaining),
            model,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=procs or None,
            prompt_cache=prompt_cache,
        )
        stop_hit = False
        try:
            for token, _logprob in generator:
                # 最初の送出: TTFT
                if _ttft is None:
                    _ttft = time.time() - generate_t
                if token in eos_ids:
                    # tool マーカー開始後、tool_call が未完のまま EOS が来ることが
                    # ある（Nemotron が回答と tool_call を分けて生成する等）。この
                    # 場合は tool_call を諦め、生成済み content を最後まで配信する。
                    if (
                        tools
                        and tool_started
                        and not _tool_done(detokenizer.text)
                    ):
                        tool_started = False
                        for y in _flush_content(final=True):
                            yield y
                    break
                detokenizer.add_token(token)
                n += 1
                text = detokenizer.text

                # tool 完了 or 停止文字列の検出
                if tools and not tool_started and _tool_marker_at(text) is not None:
                    # マーカー以降を送出しないため、まず現時点まで flush する。
                    # tool_started を立てる前に呼ぶこと（立てた後だと
                    # _flush_content 内のマーカー位置カットが効かず、
                    # <tool_call> 自体が content として漏れる）。
                    for y in _flush_content():
                        yield y
                    tool_started = True
                    if _tool_done(text):
                        break
                    continue
                if tools and tool_started:
                    if _tool_done(text):
                        break
                    # EOS で未完了のまま終わる場合の救済は生成ループ先頭で行う
                    continue
                if _stop_cut(text) is not None:
                    stop_hit = True
                    for y in _flush_content(final=True):
                        yield y
                    break
                for y in _flush_content():
                    yield y
        except Exception as e:
            print(f"[ENGINE] error at token {n}: {e}", file=sys.stderr, flush=True)
            raise

        elapsed = time.time() - generate_t
        # TTFT はリクエスト受付から最初のトークンまで = prefill + 最初の decode
        if _ttft is None:
            ttft_ms = "n/a"
        else:
            ttft_ms = (
                f"{(prefill_elapsed + _ttft) * 1000:.0f}ms"
                f"(pf {prefill_elapsed * 1000:.0f}+dec {_ttft * 1000:.0f})"
            )
        pf_info = ""
        if prefill_elapsed > 0 and prompt_tokens > 0:
            pf_info = (
                f" prefill {prompt_tokens}tok/{prefill_elapsed:.2f}s"
                f"={prompt_tokens / prefill_elapsed:.0f}t/s"
            )
        print(
            f"[ENGINE] stream {n} tokens in {elapsed:.1f}s"
            f" ({n / max(elapsed, 1e-9):.1f} t/s)"
            f"{pf_info} TTFT {ttft_ms}",
            file=sys.stderr,
            flush=True,
        )

        # 残りの未送出分（stop で打ち切った場合は送出済み）。
        # tool_call が未完のまま生成上限に達した場合も、tool_call を諦めて
        # 生成済み content を配信する（content が丸ごと落ちるのを防ぐ）。
        if tools and tool_started and not _tool_done(detokenizer.text):
            tool_started = False
        if not stop_hit and not (tools and tool_started):
            for y in _flush_content(final=True):
                yield y
        if stripper is not None and stripper.pending:
            yield (stripper.pending, n)
        if splitter is not None and splitter.pending:
            yield {"reasoning": splitter.pending}

        output_text = detokenizer.text
        content_text = output_text
        if not no_think:
            _, content_text = _split_reasoning(output_text)

        if tools:
            clean_text, tool_calls = _extract_tool_calls(content_text)
            # think ブロック内でモデルが tool_call の書式について言及するのは正常。
            # 警告は本文（think 除去後）にマーカーが残った場合のみ出す。
            if not tool_calls and _first_tool_marker(content_text) is not None:
                print(
                    f"[ENGINE] ⚠️ tool_call らしき出力を抽出できず（マーカー要確認）: "
                    f"{output_text[:240]!r}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            clean_text = content_text
            tool_calls = []

        if tool_calls:
            # ① クライアント側ツール実行方式: サーバーでは実行せず tool_calls を返して終了。
            # ② 永続 Tool replay: 生成済みの生ブロックを call ID ごとに保存する。
            raw_block = ToolReplayStore.extract_raw_block(output_text)
            if raw_block:
                self._tool_replay.remember(tool_calls, raw_block)
                self._tool_replay.persist()
            # content はストリーム送出済みのため dict では渡さない
            yield {"tool_calls": tool_calls}
        elif not stop_hit and n >= max_tokens:
            # 生成上限で打ち切られた。finish_reason=stop のままだとクライアントは
            # 回答が完結したと解釈してしまうため、length を通知する。
            yield {"finish_reason": "length"}
        return


engine: GenerationEngine = None


def _get_engine():
    global engine
    return engine


class APIHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            return self._handle_models()
        self._send_error(404, f"Not found: {path}", code="not_found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/v1/chat/completions":
            return self._handle_chat_completions()
        if path == "/v1/completions":
            return self._handle_completions()
        if path == "/v1/responses":
            return self._handle_responses()
        if path == "/v1/messages":
            return self._handle_messages()
        self._send_error(404, f"Not found: {path}", code="not_found")

    # ---- handlers ----

    def _handle_models(self):
        data = {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "elfmoon",
                }
            ],
        }
        self._send_json(200, data)

    def _read_body(self) -> dict:
        """リクエストボディを JSON として読む。不正ならエラーレスポンスを返す（None を返す）。"""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json(400, {"error": "invalid_request", "message": str(e)})
            return None
        if not isinstance(body, dict):
            self._send_json(
                400,
                {"error": "invalid_request", "message": "body must be a JSON object"},
            )
            return None
        return body

    def _check_model(self, req_id):
        if req_id is not None and req_id != "?" and req_id != MODEL_ID:
            self._send_error(
                400,
                (
                    f"model='{req_id}' はロードされていません。"
                    f"現在ロード中: {MODEL_ID}。"
                    f" クライアント設定で model を {MODEL_ID} に修正してください。"
                ),
                code="model_not_loaded",
            )
            return False
        return True

    def _default_temp(self) -> float:
        eng = _get_engine()
        if eng._model_type == "gemma4":
            return 1.0
        if "ornith" in eng._model_name:
            return 1.0
        if "glm" in eng._model_name:
            return 1.0
        return TEMP

    def _common_params(self, body: dict):
        """chat / responses / messages 共通の生成パラメータを抽出する。"""
        max_tokens = min(body.get("max_tokens", MAX_TOKENS), MAX_TOKENS)
        max_completion_tokens = body.get("max_completion_tokens")
        if max_completion_tokens is not None:
            max_tokens = min(max_completion_tokens, MAX_TOKENS)
        # --no-think 起動時は thinking 未指定のとき思考無効を既定にする。
        # クライアントが明示指定した場合はそれを優先する。
        if body.get("thinking") is None and body.get("think") is None:
            thinking = not _get_engine().default_no_think
        else:
            thinking = _reasoning_enabled(
                body.get("thinking"),
                body.get("think"),
                body.get("reasoning_effort"),
            )
        return {
            "max_tokens": max_tokens,
            "temperature": body.get("temperature", self._default_temp()),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
            "min_p": body.get("min_p"),
            "thinking": thinking,
            "no_think": not thinking,
            "stop": body.get("stop"),
            "seed": body.get("seed"),
            "frequency_penalty": body.get("frequency_penalty", 0.0),
            "presence_penalty": body.get("presence_penalty", 0.0),
            "include_usage": bool(
                (body.get("stream_options") or {}).get("include_usage")
            ),
            "reasoning_effort": body.get("reasoning_effort"),
        }

    def _tools_for(self, body: dict, messages: list) -> tuple[list | None, bool]:
        """tools / tool_choice / MCP 注入を解決する。

        戻り値: (tools, tool_choice_none)
        - tool_choice == "none" → tools を無効化
        - tools 未指定 → MCP ツールがあれば注入（ELFMOON_MCP_AUTO=0 で無効化）
        - ELFMOON_TOOLS=0 → ツールを全面的に無効化（切り分け用）
        """
        if os.environ.get("ELFMOON_TOOLS") == "0":
            return None, False
        tools = body.get("tools")
        tool_choice = body.get("tool_choice", "auto")
        if isinstance(tool_choice, str) and tool_choice.lower() == "none":
            return None, True
        if tools is None and os.environ.get("ELFMOON_MCP_AUTO", "1") != "0":
            mcp_tools = mcp_manager.get_openai_tools()
            if mcp_tools:
                tools = mcp_tools
                tool_choice = "auto"
                print(
                    f"[API] クライアントからツール未指定 → MCP {len(mcp_tools)} ツールを注入",
                    file=sys.stderr,
                    flush=True,
                )
        return tools, False

    def _handle_chat_completions(self):
        body = self._read_body()
        if body is None:
            return
        if not self._check_model(body.get("model")):
            return

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        print(
            f"[API] chat req model={body.get('model', '?')} stream={stream} msgs={len(messages)} t0={time.time():.3f}",
            file=sys.stderr,
            flush=True,
        )
        if not messages:
            return self._send_error(400, "messages is required", code="invalid_request")
        if not isinstance(messages, list):
            return self._send_error(
                400, "messages must be an array", code="invalid_request"
            )

        n = body.get("n", 1)
        if n != 1:
            return self._send_error(
                400, "n は 1 のみサポートされています", param="n", code="unsupported"
            )
        # OpenAI スキーマに存在するが未対応のパラメータ（tools/tool_choice は対応済み）
        for field in (
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "parallel_tool_calls",
            "response_format",
        ):
            if body.get(field):
                return self._send_error(
                    400, f"{field} は未サポートです", param=field, code="unsupported"
                )

        p = self._common_params(body)
        messages = _normalize_messages(messages)
        tools, _none = self._tools_for(body, messages)

        # Nemotron + ツール有効時: 回答文を先に完結させてから tool_call するよう
        # システムプロンプトで指示する（回答の途切れ対策）。
        if (
            tools
            and _get_engine()._model_type == "nemotron_h"
            and os.environ.get("ELFMOON_NEMOTRON_GUIDANCE") != "0"
        ):
            _guidance = (
                "\n\n【出力規約】ツールを使う場合も、まずユーザーへの回答文を"
                "最後まで書き切ってから tool_call を1回だけ出力してください。"
                "回答文を途中で打ち切って tool_call に切り替えないでください。"
            )
            if messages and messages[0].get("role") == "system":
                messages[0] = dict(messages[0])
                messages[0]["content"] = (messages[0].get("content") or "") + _guidance
            else:
                messages.insert(0, {"role": "system", "content": _guidance.strip()})

        # ツールなし → 従来通り API ハンドラ側で prompt レンダリング（高速パス）
        # ツールあり → エンジンに messages + tools を渡してループ処理
        if not tools:
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=p["thinking"],
                )
                prompt_nogen = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                    enable_thinking=p["thinking"],
                )
            except Exception as e:
                return self._send_error(
                    400, f"chat_template error: {e}", code="chat_template_error"
                )

            if stream:
                self._handle_stream_legacy(
                    prompt,
                    prompt_nogen,
                    p["max_tokens"],
                    p["temperature"],
                    p["no_think"],
                    p["stop"],
                    p["seed"],
                    p["frequency_penalty"],
                    p["presence_penalty"],
                    p["top_p"],
                    p["top_k"],
                    p["min_p"],
                    p["include_usage"],
                )
            else:
                self._handle_nonstream_legacy(
                    prompt,
                    prompt_nogen,
                    p["max_tokens"],
                    p["temperature"],
                    p["no_think"],
                    p["stop"],
                    p["seed"],
                    p["frequency_penalty"],
                    p["presence_penalty"],
                    p["top_p"],
                    p["top_k"],
                    p["min_p"],
                )
        else:
            if stream:
                self._handle_stream_tools(
                    messages,
                    p["max_tokens"],
                    p["temperature"],
                    p["no_think"],
                    tools,
                    p["stop"],
                    p["seed"],
                    p["frequency_penalty"],
                    p["presence_penalty"],
                    p["top_p"],
                    p["top_k"],
                    p["min_p"],
                    p["include_usage"],
                    p["reasoning_effort"],
                )
            else:
                self._handle_nonstream_tools(
                    messages,
                    p["max_tokens"],
                    p["temperature"],
                    p["no_think"],
                    tools,
                    p["stop"],
                    p["seed"],
                    p["frequency_penalty"],
                    p["presence_penalty"],
                    p["top_p"],
                    p["top_k"],
                    p["min_p"],
                    p["reasoning_effort"],
                )

    @property
    def _tokenizer(self):
        return _get_engine()._tokenizer

    # ---- 従来の高速パス（ツールなし） ---- #

    def _handle_stream_legacy(
        self,
        prompt,
        prompt_nogen,
        max_tokens,
        temperature,
        no_think,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        include_usage=False,
    ):
        t0 = time.time()
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        total = 0
        prompt_tokens = 0
        error = False

        gen = _get_engine().generate_prompt(
            prompt,
            prompt_nogen,
            max_tokens,
            temperature,
            no_think,
            stop,
            seed,
            frequency_penalty,
            presence_penalty,
            top_p,
            top_k,
            min_p,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict) and "reasoning" in msg:
                    rchunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": msg["reasoning"]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self._sse(json.dumps(rchunk, ensure_ascii=False))
                    continue
                piece, n = msg
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                    ],
                }
                self._sse(json.dumps(chunk, ensure_ascii=False))
                total = n
        except Exception as e:
            error = True
            print(
                f"[API] stream error at token {total}: {e}", file=sys.stderr, flush=True
            )
            err_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
            try:
                self._sse(json.dumps(err_chunk, ensure_ascii=False))
                self._sse("[DONE]")
            except OSError:
                pass
        else:
            done_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self._sse(json.dumps(done_chunk, ensure_ascii=False))
            if include_usage:
                usage_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": total,
                        "total_tokens": prompt_tokens + total,
                    },
                }
                self._sse(json.dumps(usage_chunk, ensure_ascii=False))
            self._sse("[DONE]")

    def _handle_nonstream_legacy(
        self,
        prompt,
        prompt_nogen,
        max_tokens,
        temperature,
        no_think,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
    ):
        t0 = time.time()
        pieces = []
        reasoning_parts = []
        total = 0
        prompt_tokens = 0
        gen = _get_engine().generate_prompt(
            prompt,
            prompt_nogen,
            max_tokens,
            temperature,
            no_think,
            stop,
            seed,
            frequency_penalty,
            presence_penalty,
            top_p,
            top_k,
            min_p,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict) and "reasoning" in msg:
                    reasoning_parts.append(msg["reasoning"])
                    continue
                piece, n = msg
                pieces.append(piece)
                total = n
        except Exception as e:
            print(f"[API] generate error: {e}", file=sys.stderr, flush=True)
            return self._send_error(
                500, str(e), type="server_error", code="generation_error"
            )

        text = "".join(pieces)
        print(
            f"[API] generate done in {time.time() - t0:.3f}s",
            file=sys.stderr,
            flush=True,
        )
        message = {"role": "assistant", "content": text}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        resp = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total,
                "total_tokens": prompt_tokens + total,
            },
        }
        self._send_json(200, resp)

    # ---- ツール対応パス ---- #

    def _handle_stream_tools(
        self,
        messages,
        max_tokens,
        temperature,
        no_think,
        tools,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        include_usage=False,
        reasoning_effort=None,
    ):
        t0 = time.time()
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())
        total = 0
        prompt_tokens = 0
        error = False

        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=no_think,
            tools=tools,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            reasoning_effort=reasoning_effort,
        )
        finish_reason = "stop"
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict) and "reasoning" in msg:
                    rchunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": msg["reasoning"]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self._sse(json.dumps(rchunk, ensure_ascii=False))
                    continue
                if isinstance(msg, dict) and "tool_calls" in msg:
                    # ① tool_calls を OpenAI ストリーミング形式の delta で返す（サーバー実行しない）
                    tc_delta = [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for i, tc in enumerate(msg["tool_calls"])
                    ]
                    self._sse(
                        json.dumps(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_ID,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"tool_calls": tc_delta},
                                        "finish_reason": None,
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        )
                    )
                    finish_reason = "tool_calls"
                    continue
                if isinstance(msg, dict) and "finish_reason" in msg:
                    finish_reason = msg["finish_reason"]
                    continue
                piece, n = msg
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                    ],
                }
                self._sse(json.dumps(chunk, ensure_ascii=False))
                total = n
        except Exception as e:
            error = True
            print(
                f"[API] stream error at token {total}: {e}",
                file=sys.stderr,
                flush=True,
            )
            err_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            }
            try:
                self._sse(json.dumps(err_chunk, ensure_ascii=False))
            except OSError:
                pass

        dt = time.time() - t0
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        if include_usage:
            final["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total,
                "total_tokens": prompt_tokens + total,
            }
        try:
            self._sse(json.dumps(final, ensure_ascii=False))
            self._sse("[DONE]")
        except OSError:
            pass
        print(
            f"[API] stream tools done: {total} tokens in {dt:.1f}s ({total / dt:.1f} t/s) error={error}",
            file=sys.stderr,
            flush=True,
        )

    def _handle_nonstream_tools(
        self,
        messages,
        max_tokens,
        temperature,
        no_think,
        tools,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        reasoning_effort=None,
    ):
        t0 = time.time()
        pieces = []
        reasoning_parts = []
        total = 0
        prompt_tokens = 0
        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=no_think,
            tools=tools,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            reasoning_effort=reasoning_effort,
        )
        tool_calls = None
        length_capped = False
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict) and "reasoning" in msg:
                    reasoning_parts.append(msg["reasoning"])
                    continue
                if isinstance(msg, dict) and "tool_calls" in msg:
                    tool_calls = msg["tool_calls"]
                    continue
                if isinstance(msg, dict) and "finish_reason" in msg:
                    length_capped = msg["finish_reason"] == "length"
                    continue
                piece, n = msg
                pieces.append(piece)
                total = n
        except Exception as e:
            print(f"[API] generate error: {e}", file=sys.stderr, flush=True)
            return self._send_error(
                500, str(e), type="server_error", code="generation_error"
            )

        print(
            f"[API] generate tools done in {time.time() - t0:.3f}s"
            f" tool_calls={len(tool_calls) if tool_calls else 0}",
            file=sys.stderr,
            flush=True,
        )
        if tool_calls:
            message = {
                "role": "assistant",
                "content": "".join(pieces) or None,
                "tool_calls": tool_calls,
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": "".join(pieces)}
            finish_reason = "length" if length_capped else "stop"
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        resp = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total,
                "total_tokens": prompt_tokens + total,
            },
        }
        self._send_json(200, resp)

    # ---- /v1/completions（OpenAI テキスト補完） ----

    def _handle_completions(self):
        body = self._read_body()
        if body is None:
            return
        if not self._check_model(body.get("model")):
            return

        prompt = body.get("prompt")
        stream = body.get("stream", False)
        print(
            f"[API] completions req model={body.get('model', '?')} stream={stream} t0={time.time():.3f}",
            file=sys.stderr,
            flush=True,
        )
        if prompt is None:
            return self._send_error(400, "prompt is required", code="invalid_request")

        p = self._common_params(body)
        messages = [{"role": "user", "content": prompt}]
        if stream:
            self._handle_stream_completions(
                messages,
                p["max_tokens"],
                p["temperature"],
                p["no_think"],
                p["stop"],
                p["seed"],
                p["frequency_penalty"],
                p["presence_penalty"],
                p["top_p"],
                p["top_k"],
                p["min_p"],
                p["include_usage"],
            )
        else:
            self._handle_nonstream_completions(
                messages,
                p["max_tokens"],
                p["temperature"],
                p["no_think"],
                p["stop"],
                p["seed"],
                p["frequency_penalty"],
                p["presence_penalty"],
                p["top_p"],
                p["top_k"],
                p["min_p"],
            )

    def _handle_stream_completions(
        self,
        messages,
        max_tokens,
        temperature,
        no_think,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        include_usage=False,
    ):
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        completion_id = f"cmpl-{int(time.time())}"
        created = int(time.time())
        total = 0
        prompt_tokens = 0
        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=no_think,
            tools=None,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict):
                    continue  # reasoning / tool_calls は completions では出さない
                piece, n = msg
                chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                }
                self._sse(json.dumps(chunk, ensure_ascii=False))
                total = n
        except Exception as e:
            print(f"[API] completions stream error: {e}", file=sys.stderr, flush=True)
            return
        done = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }
        if include_usage:
            done["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total,
                "total_tokens": prompt_tokens + total,
            }
        self._sse(json.dumps(done, ensure_ascii=False))
        self._sse("[DONE]")

    def _handle_nonstream_completions(
        self,
        messages,
        max_tokens,
        temperature,
        no_think,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
    ):
        pieces = []
        total = 0
        prompt_tokens = 0
        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=no_think,
            tools=None,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict):
                    continue
                piece, n = msg
                pieces.append(piece)
                total = n
        except Exception as e:
            print(f"[API] completions generate error: {e}", file=sys.stderr, flush=True)
            return self._send_error(
                500, str(e), type="server_error", code="generation_error"
            )
        resp = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{"index": 0, "text": "".join(pieces), "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total,
                "total_tokens": prompt_tokens + total,
            },
        }
        self._send_json(200, resp)

    # ---- /v1/responses（OpenAI Responses API） ----

    def _responses_messages(self, body: dict) -> list:
        """input（文字列 or アイテム配列）を messages に変換する。"""
        messages = []
        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            messages.append({"role": "system", "content": instructions})
        elif isinstance(instructions, list):
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in instructions
            )
            if text.strip():
                messages.append({"role": "system", "content": text})

        inp = body.get("input")
        if isinstance(inp, str):
            messages.append({"role": "user", "content": inp})
            return messages
        if not isinstance(inp, list):
            return messages
        for item in inp:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            itype = item.get("type", "message")
            if itype == "message":
                role = item.get("role", "user")
                content = item.get("content")
                messages.append(
                    {"role": role, "content": content if content is not None else ""}
                )
            elif itype == "input_text":
                messages.append({"role": "user", "content": item.get("text", "")})
            elif itype == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.get("call_id")
                                or f"call_{uuid.uuid4().hex[:12]}",
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": json.dumps(
                                        item.get("arguments") or {},
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        ],
                    }
                )
            elif itype == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "content": item.get("output") or "",
                        "tool_call_id": item.get("call_id"),
                    }
                )
            else:
                text = item.get("text")
                if text:
                    messages.append({"role": "user", "content": text})
        return messages

    def _handle_responses(self):
        body = self._read_body()
        if body is None:
            return
        if not self._check_model(body.get("model")):
            return

        stream = body.get("stream", False)
        print(
            f"[API] responses req model={body.get('model', '?')} stream={stream} t0={time.time():.3f}",
            file=sys.stderr,
            flush=True,
        )
        messages = self._responses_messages(body)
        if not messages:
            return self._send_error(400, "input is required", code="invalid_request")

        p = self._common_params(body)
        tools, _none = self._tools_for(body, messages)
        if stream:
            self._handle_stream_responses(
                messages,
                p["max_tokens"],
                p["temperature"],
                p["no_think"],
                tools,
                p["stop"],
                p["seed"],
                p["frequency_penalty"],
                p["presence_penalty"],
                p["top_p"],
                p["top_k"],
                p["min_p"],
                p["reasoning_effort"],
            )
        else:
            self._handle_nonstream_responses(
                messages,
                p["max_tokens"],
                p["temperature"],
                p["no_think"],
                tools,
                p["stop"],
                p["seed"],
                p["frequency_penalty"],
                p["presence_penalty"],
                p["top_p"],
                p["top_k"],
                p["min_p"],
                p["reasoning_effort"],
            )

    def _responses_output(
        self,
        reasoning: str,
        content: str,
        tool_calls: list | None,
        completion_id: str,
    ) -> list:
        out = []
        if reasoning:
            out.append(
                {
                    "id": f"rs_{completion_id}",
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": reasoning}],
                }
            )
        if content:
            out.append(
                {
                    "id": f"msg_{completion_id}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": content, "annotations": []}
                    ],
                }
            )
        for call in tool_calls or []:
            out.append(
                {
                    "id": f"fc_{call.get('id', '')}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call.get("id", ""),
                    "name": call.get("function", {}).get("name", ""),
                    "arguments": call.get("function", {}).get("arguments", "{}"),
                }
            )
        return out

    def _handle_nonstream_responses(
        self,
        messages,
        max_tokens,
        temperature,
        no_think,
        tools,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        reasoning_effort=None,
    ):
        pieces = []
        reasoning_parts = []
        total = 0
        prompt_tokens = 0
        tool_calls = None
        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=no_think,
            tools=tools,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            reasoning_effort=reasoning_effort,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict) and "reasoning" in msg:
                    reasoning_parts.append(msg["reasoning"])
                    continue
                if isinstance(msg, dict) and "tool_calls" in msg:
                    tool_calls = msg["tool_calls"]
                    continue
                piece, n = msg
                pieces.append(piece)
                total = n
        except Exception as e:
            print(f"[API] responses generate error: {e}", file=sys.stderr, flush=True)
            return self._send_error(
                500, str(e), type="server_error", code="generation_error"
            )
        cid = f"resp_{uuid.uuid4().hex[:20]}"
        resp = {
            "id": cid,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": MODEL_ID,
            "output": self._responses_output(
                "".join(reasoning_parts), "".join(pieces), tool_calls, cid
            ),
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": total,
                "total_tokens": prompt_tokens + total,
            },
        }
        self._send_json(200, resp)

    def _handle_stream_responses(
        self,
        messages,
        max_tokens,
        temperature,
        no_think,
        tools,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        reasoning_effort=None,
    ):
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        cid = f"resp_{uuid.uuid4().hex[:20]}"
        self._sse(
            json.dumps(
                {
                    "type": "response.created",
                    "response": {
                        "id": cid,
                        "object": "response",
                        "status": "in_progress",
                        "model": MODEL_ID,
                    },
                },
                ensure_ascii=False,
            )
        )
        reasoning_buf = ""
        content_buf = ""
        total = 0
        prompt_tokens = 0
        tool_calls = None
        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=no_think,
            tools=tools,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            reasoning_effort=reasoning_effort,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict) and "reasoning" in msg:
                    reasoning_buf += msg["reasoning"]
                    continue
                if isinstance(msg, dict) and "tool_calls" in msg:
                    tool_calls = msg["tool_calls"]
                    continue
                piece, n = msg
                content_buf += piece
                total = n
                self._sse(
                    json.dumps(
                        {
                            "type": "response.output_text.delta",
                            "delta": piece,
                            "item_id": f"msg_{cid}",
                        },
                        ensure_ascii=False,
                    )
                )
        except Exception as e:
            print(f"[API] responses stream error: {e}", file=sys.stderr, flush=True)
            return
        out = self._responses_output(reasoning_buf, content_buf, tool_calls, cid)
        self._sse(
            json.dumps(
                {
                    "type": "response.completed",
                    "response": {
                        "id": cid,
                        "object": "response",
                        "status": "completed",
                        "model": MODEL_ID,
                        "output": out,
                        "usage": {
                            "input_tokens": prompt_tokens,
                            "output_tokens": total,
                            "total_tokens": prompt_tokens + total,
                        },
                    },
                },
                ensure_ascii=False,
            )
        )
        self._sse("[DONE]")

    # ---- /v1/messages（Anthropic 互換） ----

    def _anthropic_messages(self, body: dict) -> list:
        messages = []
        system = body.get("system")
        if isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in system
            )
            if text.strip():
                messages.append({"role": "system", "content": text})
        for m in body.get("messages", []):
            if not isinstance(m, dict):
                continue
            messages.append(_normalize_message(m))
        return messages

    def _anthropic_tools(self, tools: list) -> list | None:
        """Anthropic 形式のツール定義（name/description/input_schema）を OpenAI 形式に変換する。"""
        if not tools:
            return None
        out = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema") or {},
                    },
                }
            )
        return out

    def _handle_messages(self):
        body = self._read_body()
        if body is None:
            return
        if not self._check_model(body.get("model")):
            return

        stream = body.get("stream", False)
        print(
            f"[API] messages req model={body.get('model', '?')} stream={stream} t0={time.time():.3f}",
            file=sys.stderr,
            flush=True,
        )
        messages = self._anthropic_messages(body)
        if not messages:
            return self._send_error(400, "messages is required", code="invalid_request")

        p = self._common_params(body)
        p["stop"] = body.get("stop_sequences")
        tools = self._anthropic_tools(body.get("tools"))
        if stream:
            self._handle_stream_messages(
                messages,
                p["max_tokens"],
                p["temperature"],
                p["no_think"],
                tools,
                p["stop"],
                p["seed"],
                p["frequency_penalty"],
                p["presence_penalty"],
                p["top_p"],
                p["top_k"],
                p["min_p"],
            )
        else:
            self._handle_nonstream_messages(
                messages,
                p["max_tokens"],
                p["temperature"],
                p["no_think"],
                tools,
                p["stop"],
                p["seed"],
                p["frequency_penalty"],
                p["presence_penalty"],
                p["top_p"],
                p["top_k"],
                p["min_p"],
            )

    def _anthropic_content(
        self, reasoning: str, content: str, tool_calls: list | None
    ) -> list:
        blocks = []
        if reasoning:
            blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
        if content:
            blocks.append({"type": "text", "text": content})
        for call in tool_calls or []:
            fn = call.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = fn.get("arguments") or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                }
            )
        return blocks

    def _handle_nonstream_messages(
        self,
        messages,
        max_tokens,
        temperature,
        no_think,
        tools,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
    ):
        pieces = []
        reasoning_parts = []
        total = 0
        prompt_tokens = 0
        tool_calls = None
        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=no_think,
            tools=tools,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict) and "reasoning" in msg:
                    reasoning_parts.append(msg["reasoning"])
                    continue
                if isinstance(msg, dict) and "tool_calls" in msg:
                    tool_calls = msg["tool_calls"]
                    continue
                piece, n = msg
                pieces.append(piece)
                total = n
        except Exception as e:
            print(f"[API] messages generate error: {e}", file=sys.stderr, flush=True)
            return self._send_error(
                500, str(e), type="server_error", code="generation_error"
            )
        content = self._anthropic_content(
            "".join(reasoning_parts), "".join(pieces), tool_calls
        )
        resp = {
            "id": f"msg_{uuid.uuid4().hex[:20]}",
            "type": "message",
            "role": "assistant",
            "model": MODEL_ID,
            "content": content,
            "stop_reason": "tool_use" if tool_calls else "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": total,
            },
        }
        self._send_json(200, resp)

    def _handle_stream_messages(
        self,
        messages,
        max_tokens,
        temperature,
        no_think,
        tools,
        stop=None,
        seed=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
    ):
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        mid = f"msg_{uuid.uuid4().hex[:20]}"
        self._sse(
            json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": mid,
                        "type": "message",
                        "role": "assistant",
                        "model": MODEL_ID,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
                ensure_ascii=False,
            )
        )
        reasoning_buf = ""
        content_buf = ""
        total = 0
        prompt_tokens = 0
        tool_calls = None
        block_index = 0
        text_started = False
        gen = _get_engine().generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            no_think=no_think,
            tools=tools,
            stop=stop,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        try:
            for msg in gen:
                if isinstance(msg, int):
                    prompt_tokens = msg
                    continue
                if isinstance(msg, dict) and "reasoning" in msg:
                    reasoning_buf += msg["reasoning"]
                    continue
                if isinstance(msg, dict) and "tool_calls" in msg:
                    tool_calls = msg["tool_calls"]
                    continue
                piece, n = msg
                if not text_started:
                    text_started = True
                    self._sse(
                        json.dumps(
                            {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                            ensure_ascii=False,
                        )
                    )
                content_buf += piece
                total = n
                self._sse(
                    json.dumps(
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": piece},
                        },
                        ensure_ascii=False,
                    )
                )
        except Exception as e:
            print(f"[API] messages stream error: {e}", file=sys.stderr, flush=True)
            return
        if text_started:
            self._sse(
                json.dumps(
                    {"type": "content_block_stop", "index": block_index},
                    ensure_ascii=False,
                )
            )
            block_index += 1
        # thinking / tool_use ブロックを後続で追加
        if reasoning_buf:
            self._sse(
                json.dumps(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "thinking",
                            "thinking": reasoning_buf,
                            "signature": "",
                        },
                    },
                    ensure_ascii=False,
                )
            )
            self._sse(
                json.dumps(
                    {"type": "content_block_stop", "index": block_index},
                    ensure_ascii=False,
                )
            )
            block_index += 1
        for call in tool_calls or []:
            fn = call.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = fn.get("arguments") or {}
            self._sse(
                json.dumps(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        },
                    },
                    ensure_ascii=False,
                )
            )
            self._sse(
                json.dumps(
                    {"type": "content_block_stop", "index": block_index},
                    ensure_ascii=False,
                )
            )
            block_index += 1
        self._sse(
            json.dumps(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "tool_use" if tool_calls else "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": total,
                    },
                },
                ensure_ascii=False,
            )
        )
        self._sse(json.dumps({"type": "message_stop"}, ensure_ascii=False))

    # ---- helpers ----

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(
        self,
        status,
        message,
        type="invalid_request_error",
        param=None,
        code="invalid_request_error",
    ):
        return self._send_json(
            status,
            {"error": {"message": message, "type": type, "param": param, "code": code}},
        )

    def _sse(self, data):
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def log_message(self, fmt, *args):
        print(f"[API] {fmt % args}", file=sys.stderr, flush=True)


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

    perf = "--perf" in argv or os.environ.get("ELFMOON_PERF") == "1"
    no_think = "--no-think" in argv
    model_name = None
    if "--model" in argv:
        idx = argv.index("--model")
        model_name = argv[idx + 1]
        argv = argv[:idx] + argv[idx + 2 :]
    args = [a for a in argv if a not in ("--no-think", "--perf")]
    port = int(args[0]) if len(args) > 0 else DEFAULT_PORT
    cap = int(args[1]) if len(args) > 1 else DEFAULT_CAPACITY

    model_path, store_dir = resolve_model(model_name)
    # KV キャッシュをモデル別に分離（モデル間の同一プロンプト衝突＝形状不一致を防ぐ）
    kv_manager.set_namespace(os.path.basename(model_path))

    global MODEL_ID, engine
    MODEL_ID = model_name or os.path.basename(model_path)

    mode = "性能" if perf else "省メモリ"
    print(f"モデル: {model_path}", flush=True)
    print(f"モデルをロード中...（{mode}モード, capacity={cap or 'auto'}）", flush=True)
    t0 = time.perf_counter()

    engine = GenerationEngine(model_path, store_dir, cap, perf)
    engine.default_no_think = no_think

    print(f"準備完了（{time.perf_counter() - t0:.0f}秒）", flush=True)
    print("", flush=True)
    print(f"  ElfMoon API サーバ起動: http://{HOST}:{port}", flush=True)
    if HOST == "127.0.0.1":
        print(
            "  （LAN公開する場合: ELFMOON_HOST=0.0.0.0 で起動。認証なし注意）",
            flush=True,
        )
    print("  POST /v1/chat/completions  (OpenAI 互換, stream/non-stream)", flush=True)
    print("  POST /v1/completions       (OpenAI テキスト補完)", flush=True)
    print("  POST /v1/responses         (OpenAI Responses API)", flush=True)
    print("  POST /v1/messages          (Anthropic 互換)", flush=True)
    print("  GET  /v1/models", flush=True)
    print("", flush=True)
    print("  Claude Code 設定例 (~/.clauderc.json または claude.json):", flush=True)
    print('    {"models":[{"name":"elfmoon","provider":"openai",', flush=True)
    print(f'      "model":"{MODEL_ID}","apiKey":"sk-not-needed",', flush=True)
    print(f'      "baseUrl":"http://localhost:{port}/v1"}}]}}', flush=True)
    print("", flush=True)
    print("  VS Code Continue 設定例 (~/.continue/config.json):", flush=True)
    print('    {"models":[{"title":"ElfMoon","provider":"openai",', flush=True)
    print(
        f'      "model":"{MODEL_ID}","apiBase":"http://localhost:{port}/v1"}}]}}',
        flush=True,
    )
    print("  Ctrl-C で終了", flush=True)

    server = ThreadingHTTPServer((HOST, port), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nシャットダウン中...")
        server.shutdown()


if __name__ == "__main__":
    main()
