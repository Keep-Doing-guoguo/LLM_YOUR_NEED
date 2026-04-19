#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/6 16:57
@source from: 
"""
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re

# ✅ 1. 创建工具函数
def calculator_tool(input: str) -> str:
    try:
        return str(eval(input))
    except:
        return "无法计算"

# ✅ 2. 注册 Tool
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="用于进行数学表达式计算，比如：3 * (2 + 4)"
    )
]

# ✅ 3. Prompt 模板（Agent 使用 Tool 的提示）
prefix = """你是一个智能助手，有如下工具：

{tools}

你应该按照如下格式来回答：

问题: 用户的问题
思考: 你应该做什么
操作: 工具名称
操作输入: 工具的输入
观察: 工具的输出结果
...（这个过程可以重复）
最终答案: 你要回复用户的答案

"""

suffix = """现在请开始：

问题: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)

# ✅ 4. 解析 LLM 输出
class SimpleOutputParser(AgentOutputParser):
    def parse(self, text: str):
        if "最终答案:" in text:
            return AgentFinish(
                return_values={"output": text.split("最终答案:")[-1].strip()},
                log=text
            )
        match = re.search(r"操作: (.*?)\n操作输入: (.*)", text, re.DOTALL)
        if match:
            return AgentAction(
                tool=match.group(1).strip(),
                tool_input=match.group(2).strip(),
                log=text
            )
        raise ValueError("无法解析输出: ", text)

# ✅ 5. 创建 LLM 和 AgentExecutor
llm = ChatOpenAI(
    temperature=0,
    openai_api_key="sk-",  # ✅替换为你的 key
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name="qwen-plus"
)

agent = ZeroShotAgent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    output_parser=SimpleOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ✅ 6. 测试调用
res = agent_executor.invoke({"input": "3 * (5 + 7)"})
print("✅ 最终答案：", res["output"])