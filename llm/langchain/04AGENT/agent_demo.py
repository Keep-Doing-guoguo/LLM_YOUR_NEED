#!/usr/bin/env python
# coding=utf-8

"""
LangChain Agent runnable demo.

Run:
    export DASHSCOPE_API_KEY="your_api_key"
    python agent_demo.py

Optional:
    export AGENT_MODEL="qwen-plus"
    export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
"""

from __future__ import annotations

import ast
import operator
import os
import sqlite3
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI


DB_PATH = Path(__file__).with_name("agent_demo.sqlite3")


def build_llm() -> ChatOpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "请先设置 DASHSCOPE_API_KEY；如果使用 OpenAI 兼容服务，也可以设置 OPENAI_API_KEY。"
        )

    return ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "qwen-plus"),
        temperature=0,
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )


def init_demo_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            user_name TEXT NOT NULL,
            product TEXT NOT NULL,
            status TEXT NOT NULL,
            amount REAL NOT NULL
        )
        """
    )
    cur.execute("DELETE FROM orders")
    cur.executemany(
        """
        INSERT INTO orders (id, user_name, product, status, amount)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("A1001", "Alice", "机械键盘", "已发货", 399.0),
            ("A1002", "Bob", "显示器", "待付款", 1299.0),
            ("A1003", "Carol", "无线鼠标", "已签收", 159.0),
        ],
    )
    conn.commit()
    conn.close()


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_math(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _safe_eval_math(node.left)
        right = _safe_eval_math(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        operand = _safe_eval_math(node.operand)
        return _ALLOWED_UNARY_OPS[type(node.op)](operand)
    raise ValueError("只支持数字和 + - * / // % ** 括号等基础数学表达式")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入必须是纯数学表达式，例如 3 * (5 + 7)。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_math(tree)
        return str(result)
    except Exception as exc:
        return f"计算失败：{exc}"


@tool
def get_weather(city: str) -> str:
    """查询城市天气。输入城市名称，返回天气摘要。"""
    weather_map = {
        "北京": "晴，最高 24 摄氏度，空气质量良。",
        "上海": "多云，最高 26 摄氏度，东南风 3 级。",
        "深圳": "阵雨，最高 29 摄氏度，湿度较高。",
        "杭州": "小雨，最高 25 摄氏度。",
    }
    return weather_map.get(city, f"{city} 暂无实时天气数据，建议接入真实天气 API。")


@tool
def query_orders(sql: str) -> str:
    """查询订单 SQLite 表。只允许 SELECT 语句，例如 SELECT * FROM orders LIMIT 3。"""
    normalized = sql.strip().rstrip(";")
    if not normalized.lower().startswith("select"):
        return "拒绝执行：该工具只允许 SELECT 查询。"

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(normalized)
        rows = cur.fetchall()
        columns = [item[0] for item in cur.description or []]
        conn.close()
    except Exception as exc:
        return f"SQL 查询失败：{exc}"

    records: list[dict[str, Any]] = [dict(zip(columns, row)) for row in rows]
    return str(records)


def build_agent():
    init_demo_db()
    llm = build_llm()
    return create_agent(
        model=llm,
        tools=[calculator, get_weather, query_orders],
        system_prompt=(
            "你是一个工具型助手。需要计算、查天气或查订单时，优先调用合适的工具；"
            "回答要简洁，并说明你使用了哪个工具。"
        ),
    )


def ask(agent, question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    return messages[-1].content


def main() -> None:
    agent = build_agent()
    questions = [
        "3 * (5 + 7) 等于多少？",
        "上海今天的天气怎么样？",
        "查询订单表里已发货的订单，只返回 id、user_name、product、status。",
    ]

    for question in questions:
        print(f"\n用户：{question}")
        print(f"Agent：{ask(agent, question)}")


if __name__ == "__main__":
    main()
