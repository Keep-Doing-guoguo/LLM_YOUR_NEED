#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/10/11 18:04
@source from: 
"""
"""
多工具 Agent 示例（SQLite 版本）：
- Calculator
- SQL 查询（直写 SQL）
- Milvus 语义检索（外部 embedding + Milvus 搜索）
"""

import os
import json
import sqlite3

from langchain_community.chat_models import ChatOpenAI

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from tools.tools_select import  tools

LLM = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    api_key="你的key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 标准 ReAct 提示词（从 Hub 拉取官方模板）
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=LLM,
    tools=tools,
    prompt=prompt,
)## === AgentType.ZERO_SHOT_REACT_DESCRIPTION 等价于 ===

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# ========== 6) 测试 ==========
def demo():
    print("\n=== 计算器 ===")
    print(agent_executor.invoke({"input": "3 * (5 + 7) 等于多少？"})["output"])

    print("\n=== SQLite ===")


    print(agent_executor.invoke({"input": "请用SQL查询：SELECT * FROM message LIMIT 3;"})["output"])

    print("\n=== Milvus 检索 ===")
    print(agent_executor.invoke({"input": "帮我在短信库里找到与“您的申通快递已到站，请凭手机号取件”相似的记录，给我前 5 条。"})["output"])

if __name__ == "__main__":
    demo()