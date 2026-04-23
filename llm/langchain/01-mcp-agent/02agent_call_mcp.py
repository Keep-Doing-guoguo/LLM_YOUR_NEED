#!/usr/bin/env python
# coding=utf-8

"""
agent_call_mcp.py
一个最小可跑通的：LLM + MCP(天气工具) 示例
"""

import json
import asyncio
from typing import Any, Dict, List

from openai import OpenAI

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

# ====== LLM 配置（DashScope 兼容模式） ======
client = OpenAI(
    api_key="sk-79fcaf8f7fe24839b4abcbdd9c9e8980",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-plus"


def mcp_tools_to_openai_tools(mcp_tools: List[Any]) -> List[Dict[str, Any]]:
    """把 MCP 的工具描述转换成 OpenAI tools 格式"""
    tools: List[Dict[str, Any]] = []
    for t in mcp_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            }
        })
    return tools

import os
async def run_agent(user_prompt: str) -> str:
    # 找到 01mcp_weather_server.py 的绝对路径
    server_script = os.path.join(
        os.path.dirname(__file__),  # 当前这个 py 文件所在目录
        "01mcp_weather_server.py"
    )

    # 关键：command 是程序，args 是参数列表
    server_params = StdioServerParameters(
        command="python",  # 或 "python3"，看你本机命令
        args=[server_script],
    )
    # 2) 通过 stdio_client 建立到 MCP Server 的连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 3) 获取 MCP 提供的 tools 列表
            mcp_tool_list = (await session.list_tools()).tools
            print("MCP tools:", [t.name for t in mcp_tool_list])
            openai_tools = mcp_tools_to_openai_tools(mcp_tool_list)

            # 4) 第一次调用 LLM，让它决定要不要用工具
            messages: List[Dict[str, Any]] = [
                {
                    "role": "system",
                    "content": "你是一个会调用工具（天气）的智能助手，如果问题涉及天气，请使用工具。"
                },
                {"role": "user", "content": user_prompt},
            ]

            resp1 = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            msg1 = resp1.choices[0].message
            tool_calls = getattr(msg1, "tool_calls", None)

            # 5) 如果模型决定要调用 MCP 工具
            if tool_calls:
                messages.append(msg1.model_dump())

                for call in tool_calls:
                    fn = call.function.name
                    args = json.loads(call.function.arguments or "{}")
                    print(f"[Agent] 调用 MCP 工具: {fn}({args})")

                    # 调 MCP tool
                    tool_result = await session.call_tool(fn, args)

                    # 把工具执行结果回填给模型
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            [c.model_dump() for c in tool_result.content],
                            ensure_ascii=False
                        ),
                    })

                # 6) 第二次调用 LLM，让它基于工具结果给最终回答
                resp2 = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )
                return resp2.choices[0].message.content or ""

            # 7) 模型不想用工具，就直接返回第一次回答
            return msg1.content or ""


if __name__ == "__main__":
    ans = asyncio.run(run_agent("帮我查一下济南今天的天气，然后给一句出门建议。"))
    print("\n=== 最终回答 ===")
    print(ans)