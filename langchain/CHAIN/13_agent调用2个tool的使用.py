#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/6 16:28
@source from: 
"""
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.agents.agent import AgentOutputParser
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import re
from typing import List, Tuple, Dict, Any


# 1. 增强工具定义
def search_tool(query: str) -> str:
    """专业的技术文档搜索引擎"""
    return f"技术文档[ID-2025]显示：'{query}' 是面向大模型应用的开发框架，支持工具链集成"


def calc_tool(expr: str) -> str:
    """数学表达式计算器（支持加减乘除和括号）"""
    try:
        return f"计算结果：{eval(expr)}"
    except Exception as e:
        return f"计算错误：{str(e)}"


tools = [
    Tool(
        name="SearchTool",
        func=search_tool,
        description="技术文档搜索引擎，输入应为技术名词或框架名称。输出格式：'技术文档[ID-xxxx]显示：[内容]'"
    ),
    Tool(
        name="Calculator",
        func=calc_tool,
        description="数学计算器，输入应为有效表达式（如3*(5+7)），输出格式：'计算结果：[数值]' 或 '计算错误：[原因]'"
    )
]

# 2. 增强提示模板（参考网页7）
prefix = """你是一个专业的技术支持助手，需要根据对话历史和工具响应提供准确答案。以下是之前的对话：
{chat_history}

可用工具列表：

{tools}

请严格使用以下格式：
Question: 用户问题
Thought: 分析步骤（需用中文）
Action: 工具名称
Action Input: 工具输入参数
Observation: 工具返回结果
...（可重复多次）
Final Answer: 最终结论（需用中文）
"""

suffix = """开始！

Question: {input}
{agent_scratchpad}"""

# 修正后的提示模板初始化
prompt = ZeroShotAgent.create_prompt(
    tools=tools,  # 传入工具列表
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)


# 3. 健壮的输出解析（参考网页2）
class RobustOutputParser(AgentOutputParser):
    def parse(self, text: str) -> AgentAction | AgentFinish:
        text = re.sub(r'\s+', ' ', text).strip()

        # 优先处理最终答案
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].split("Action:")[0].strip()
            return AgentFinish(return_values={"output": answer}, log=text)

        # 增强action解析
        action_pattern = r"Action:\s*([a-zA-Z]+)\s*Action Input:\s*[\"']?(.*?)[\"']?(?=\s*Action:|$)"
        match = re.search(action_pattern, text, re.IGNORECASE | re.DOTALL)

        if match and match.group(1) in [t.name for t in tools]:
            return AgentAction(
                tool=match.group(1).strip(),
                tool_input=match.group(2).strip(),
                log=text
            )

        # 错误处理
        error_msg = f"无法解析代理响应：{text}"
        return AgentFinish(return_values={"output": error_msg}, log=text)


# 4. 带错误处理的初始化（参考网页1）
llm = ChatOpenAI(
    temperature=0,
    model_name="qwen-plus",
    openai_api_key="sk-",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_retries=3  # 添加重试机制
)
# 新增LLMChain构建步骤（网页3的核心修正点）
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)
tool_names = [tool.name for tool in tools]  # 生成工具名称列表

agent = ZeroShotAgent(
    llm_chain=llm_chain,  # 替换原来的llm参数
    tools=tools,

    allowed_tools=tool_names,  # 传入工具名称列表
    prompt=prompt,
    output_parser=RobustOutputParser(),
    max_iterations=5  # 防止无限循环
)

# 5. 带内存的执行器（参考网页8）
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=ConversationBufferWindowMemory(
        k=3,  # 保留最近3轮对话
        memory_key="chat_history",
        return_messages=True
    ),
    verbose=True,
    handle_parsing_errors=lambda e: f"解析错误，请检查格式：{str(e)}",  # 错误处理
    early_stopping_method="generate"  # 提前停止策略
)

# 6. 测试用例验证
queries = [
    {"input": "LangChain 的核心功能是什么"},
    {"input": "请计算 (3 + 5) * 2^3"},
    {"input": "我之前问了哪些问题？"}
]

for query in queries:
    try:
        result = agent_executor.invoke(query)
        print(f"Question: {query['input']}")
        print(f"Answer: {result['output']}\n{'-' * 50}")
    except Exception as e:
        print(f"执行错误: {str(e)}")