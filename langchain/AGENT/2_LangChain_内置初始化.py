#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/6 19:54
@source from: 
"""
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool
def calculator_tool(input_str: str) -> str:
    try:
        return str(eval(input_str))
    except Exception as e:
        return f"计算错误: {e}"

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="用于数学表达式计算，如 2 + 2、3 * (5 + 6) 等",
        return_direct=True,  # 关键：工具输出直接作为最终答案返回
    )
]
tool_names = [tool.name for tool in tools]

llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    api_key="你的key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

tools = [Tool(name="Calculator", func=calculator_tool, description="数学表达式计算")]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,         # 防御
    handle_parsing_errors=True,
)

print(agent_executor.run("3 * (5 + 7) 是多少？"))



###由于langchain的更新可以更改为下面的代码####


# from langchain.agents import create_react_agent, AgentExecutor
# from langchain import hub
# from langchain_community.chat_models import ChatOpenAI
# from langchain.tools import Tool
#
# def calculator_tool(expr: str) -> str:
#     try:
#         return str(eval(expr))
#     except Exception as e:
#         return f"计算错误: {e}"
#
# llm = ChatOpenAI(
#     model="qwen-plus",
#     temperature=0,
#     api_key="YOUR_KEY",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
#
# tools = [
#     Tool(
#         name="Calculator",
#         func=calculator_tool,
#         description="用于数学表达式计算，如 2 + 2、3 * (5 + 6) 等",
#         # return_direct=True,  # 若你希望工具输出直接作为最终答案，可打开
#     )
# ]
#
# # 官方 ReAct 模板
# prompt = hub.pull("hwchase17/react")
#
# agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
# executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=4)
#
# print(executor.invoke({"input": "3 * (5 + 7) 等于多少？"})["output"])