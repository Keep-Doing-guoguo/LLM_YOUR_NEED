#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/5 11:43
@source from: 
"""
# ✅ Function Call VS Agent 使用对比

# ---
# 🎯 场景目标：用户问 "2 的 10 次方是多少？"，系统通过 LLM 判断需要调用计算函数返回结果

# ✅ 1. 使用 Function Call（OpenAI functions）
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
#from langchain.agents.agent_toolkits import create_openai_functions_agent
from langchain.agents import AgentExecutor

# 函数封装为 Tool
@tool
def power(base: float, exponent: float) -> float:
    """计算 base 的 exponent 次方"""
    return base ** exponent

llm = ChatOpenAI(
        openai_api_key="",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        temperature=0.7,
)

# agent = create_openai_functions_agent(
#     llm=llm,
#     tools=[power],
# )
# executor = AgentExecutor(agent=agent, tools=[power], verbose=True)
#
# result = executor.invoke({"input": "2 的 10 次方是多少？"})
# print("[Function Call] 🤖", result["output"])


# ✅ 2. 使用 Agent + Tool（Zero-Shot Agent）
from langchain.agents import Tool, initialize_agent, AgentType

# 普通 Tool
def power_tool(input: str) -> str:
    try:
        base, exp = map(float, input.split("^"))
        return str(base ** exp)
    except:
        return "格式错误，示例：2^10"

tools = [
    Tool(name="Power Calculator", func=power_tool, description="计算幂，格式 2^10 表示 2 的 10 次方")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result2 = agent.run("请帮我计算 2^10")
print("[Agent] 🤖", result2)
