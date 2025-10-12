#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/5 11:49
@source from: 
"""
"""

✅ 一、核心组件：
	•	AgentExecutor：执行 Agent 的核心逻辑器
	•	initialize_agent：快速构建 Agent（可选）
	•	LLMSingleActionAgent / ZeroShotAgent：灵活可配置的推理 Agent
	•	Tool：你定义的可被调用的函数或模块
	•	Memory：如 ConversationBufferMemory，Agent 会记住历史
	•	PromptTemplate：自定义 Agent 的提示词
	•	AgentOutputParser：处理 LLM 输出，提取 Action 或返回结果
"""
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re

# 工具函数
def search_tool(query: str) -> str:#模拟搜索资料内容
    return f"🔍 搜索结果：'{query}' 是一门用于大模型开发的框架。"

def calc_tool(expr: str) -> str:#模拟数学运算。
    try:
        return str(eval(expr))
    except:
        return "❌ 无法计算表达式"

# 工具列表
tools = [
    Tool(name="SearchTool", func=search_tool, description="用于搜索资料内容"),
    Tool(name="Calculator", func=calc_tool, description="用于进行加减乘除等数学运算")
]

# Prompt 模板
prefix = """你是一个智能助手，可以使用以下工具：

{tools}

当你需要做出决定时，请遵循以下格式：

问题: {input}
思考: 你要如何处理这个问题？
操作: 工具名称
操作输入: 输入到工具的内容
观察: 工具的返回值
...（重复上述过程直到获得答案）
最终答案: 最终的回答

"""

suffix = """现在请开始回答：

问题: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)

# 输出解析器（解析 Action 或 Final Answer）
class CustomOutputParser(AgentOutputParser):
    def parse(self, text: str):
        if "最终答案:" in text:
            return AgentFinish(return_values={"output": text.split("最终答案:")[-1].strip()}, log=text)
        match = re.search(r"操作: (.*?)\n操作输入: (.*)", text, re.DOTALL)
        if match:
            return AgentAction(tool=match.group(1).strip(), tool_input=match.group(2).strip(), log=text)
        raise ValueError("解析失败：", text)

# LLM + 记忆
llm = ChatOpenAI(
        openai_api_key="sk-",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        temperature=0.7,
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 构建 Agent
agent = ZeroShotAgent(llm=llm, tools=tools, prompt=prompt, output_parser=CustomOutputParser())
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# 启动 Agent 问答
agent_executor.invoke({"input": "LangChain 是什么？"})
agent_executor.invoke({"input": "3 * (5 + 7) 等于多少？"})
agent_executor.invoke({"input": "你还记得我上一个问题问了什么吗？"})
"""
✅ 三、复杂功能你可以拓展：
	•	✅ 多轮上下文（加 Memory）
	•	✅ 函数组合使用（多个 Tool 串联）
	•	✅ 自定义格式（多语言、输出样式）
	•	✅ 插件式扩展（Web 搜索、数据库问答）
"""