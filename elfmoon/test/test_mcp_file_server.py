"""ファイル操作テスト専用の MCP サーバー（ELFMOON テスト用）。

@modelcontextprotocol/server-filesystem には delete ツールが無いため、
ファイル操作（read/write/edit/delete）の 4 操作をテストするための
軽量 MCP サーバーを提供する。

起動: 環境変数 ELFMOON_TEST_FS_DIR で管理ルートを指定し、mcp_client.py 経由で
stdio 起動する（elfmoon/test/test_file_ops.py から使用）。

ツール:
  - read_file:  ファイル内容の読み込み
  - write_file: ファイルの新規作成・上書き
  - edit_file:  文字列置換による編集（old_string を new_string に置換）
  - delete_file: ファイルの削除

usage:
  ELFMOON_TEST_FS_DIR=/path python3 elfmoon/test/test_mcp_file_server.py
"""

import asyncio
import os
from pathlib import Path

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

ROOT = Path(os.environ.get("ELFMOON_TEST_FS_DIR", ".")).resolve()

server = Server("elfmoon-test-file-server")


def _resolve(rel_path: str) -> Path:
    """テストルート直下のパスに解決する（ディレクトリトラバーサル防止）。"""
    p = (ROOT / rel_path).resolve()
    if not (p == ROOT or ROOT in p.parents):
        raise ValueError(f"テストルート外のパスは許可されません: {rel_path}")
    return p


@server.list_tools()
async def list_tools():
    from mcp.types import Tool

    return [
        Tool(
            name="read_file",
            description="テストルート内のファイル内容を読み込む",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "テストルートからの相対パス",
                    }
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="write_file",
            description="テストルート内にファイルを新規作成する（既存は上書き）",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "テストルートからの相対パス",
                    },
                    "content": {"type": "string", "description": "書き込む内容"},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="edit_file",
            description="テストルート内のファイルの文字列を置換して編集する",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "テストルートからの相対パス",
                    },
                    "old_string": {"type": "string", "description": "置換対象の文字列"},
                    "new_string": {"type": "string", "description": "置換後の文字列"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        ),
        Tool(
            name="delete_file",
            description="テストルート内のファイルを削除する",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "テストルートからの相対パス",
                    }
                },
                "required": ["path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    from mcp.types import TextContent

    if name == "read_file":
        p = _resolve(arguments["path"])
        if not p.is_file():
            raise ValueError(f"ファイルが存在しません: {arguments['path']}")
        content = p.read_text()
        return [TextContent(type="text", text=content)]
    if name == "write_file":
        p = _resolve(arguments["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(arguments["content"])
        return [TextContent(type="text", text=f"書き込み完了: {arguments['path']}")]
    if name == "edit_file":
        p = _resolve(arguments["path"])
        if not p.is_file():
            raise ValueError(f"ファイルが存在しません: {arguments['path']}")
        old = arguments["old_string"]
        new = arguments["new_string"]
        text = p.read_text()
        if old not in text:
            raise ValueError(f"置換対象の文字列がファイルに存在しません: {old!r}")
        p.write_text(text.replace(old, new, 1))
        return [TextContent(type="text", text=f"編集完了: {arguments['path']}")]
    if name == "delete_file":
        p = _resolve(arguments["path"])
        if not p.is_file():
            raise ValueError(f"ファイルが存在しません: {arguments['path']}")
        p.unlink()
        return [TextContent(type="text", text=f"削除完了: {arguments['path']}")]
    raise ValueError(f"未知ツール: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="elfmoon-test-file-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
