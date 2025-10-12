import re
from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from langchain_community.chat_models import ChatOpenAI


# ✅ 简单计算工具
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

# ✅ Prompt 模板
prompt_template = """你是一个智能助手，可以使用以下工具：

{tools}

请按照如下格式思考并回答问题：

问题: {input}
思考: 如何解决这个问题
操作: 工具名称（如 {tool_names}）
操作输入: 提供给工具的表达式
观察: 工具返回的结果
思考: 我已得出最终答案
最终答案: 最终答案内容
"""

# ✅ 输出解析器
class MyOutputParser(AgentOutputParser):
    def parse(self, text: str):
        if "最终答案:" in text:
            return AgentFinish(
                return_values={"output": text.split("最终答案:")[-1].strip()},
                log=text,
            )
        match = re.search(r"操作: (.*?)\n操作输入: (.*)", text, re.DOTALL)
        if match:
            return AgentAction(
                tool=match.group(1).strip(),
                tool_input=match.group(2).strip(),
                log=text,
            )
        raise ValueError("无法解析 LLM 输出")

# ✅ 构建 agent 并运行
def main():
    prompt = PromptTemplate(
        input_variables=["input", "tools", "tool_names"],
        template=prompt_template
    )
    llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0,
        api_key="你的key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt.partial(tools="\n".join(
    [f"{tool.name}: {tool.description}" for tool in tools]), tool_names=", ".join(tool_names)))

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=MyOutputParser(),
        stop=["\n观察:", "观察:"],
        allowed_tools=tool_names,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )
    result = agent_executor.run("3 * (5 + 7) 是多少？")
    print("🤖 最终答案:", result)

if __name__ == "__main__":
    main()


##对比差异性分析：AI给出的对比结果分析。
'''


1）手写的 ReAct Agent（LLMSingleActionAgent + 自定义 Parser）

**文件 1：**用自定义 Prompt + 自定义 AgentOutputParser + LLMSingleActionAgent
特点：
	•	你完全掌控提示词格式：要求 LLM 严格产出

问题/思考/操作/操作输入/观察/思考/最终答案


	•	你负责解析模型输出：MyOutputParser 用正则抽取“操作/操作输入”，或在出现“最终答案:”时结束。
	•	更高可定制性：停用词（stop=["观察:"]）、中文格式、工具说明都由你定。
	•	更容易出错：只要 LLM 没按你定义的格式说话，就会 ValueError("无法解析") 或无限循环（除非你自己加迭代上限）。
	•	工具设置了 return_direct=True：工具输出会被直接作为最终答案（绕过“思考 -> 最终答案”的一步）。

适合：有明确格式约束、需要完全掌控推理格式/停用词/多语种提示词，或要做“严格解析”的场景。

⸻

2）LangChain 内置初始化（initialize_agent(AgentType.ZERO_SHOT_REACT_DESCRIPTION)）

**文件 2：**用框架内置 ReAct 模板 & 解析器
特点：
	•	少代码，拿来即用：不给自定义 Parser，直接用 LangChain 内置的提示词与解析逻辑。
	•	工具选择靠描述：AgentType.ZERO_SHOT_REACT_DESCRIPTION 会让模型根据 Tool.description 自动决定是否调用工具。
	•	更稳健：内置解析器对 ReAct 模式有不少容错，你也设置了 handle_parsing_errors=True。
	•	约束力弱：提示词是通用英文模版；如果你需要中文、严格结构化或特定链路，改动空间比第一种小。
	•	你示例里第二段没有给 return_direct=True，因此一般是工具返回 → LLM再组织最终答复。

注意：initialize_agent 已被官方标记过时（Deprecation），建议用：
	•	create_react_agent(...) + AgentExecutor(...)
	•	或 create_structured_chat_agent(...)

⸻

关键对比（表）

维度	自写 ReAct（代码 1）	内置 ReAct（代码 2）
Prompt & 输出格式	你自定义（中文结构化）	官方预置模板（英文 ReAct）
输出解析	你自己写 Parser	框架内置解析
容错与健壮性	取决于你 Parser 的鲁棒性，易被模型“跑题格式”卡住	更稳，有 handle_parsing_errors 等兜底
工具结果直出	return_direct=True 可直接返回工具输出	默认先工具→再 LLM组织答案（除非你也设 return_direct）
可控性	最高（停用词、格式、语言、回合逻辑都能控）	较弱（更易用，改动少）
维护成本	高（要对付解析失败/循环）	低
未来建议	继续用就好	切到 create_react_agent 等新 API


⸻

哪个更“好”？
	•	想快跑/少踩坑：用 内置 ReAct（第 2 段），现在推荐直接用 create_react_agent 新 API。
	•	要强约束/中文格式/严格解析/可审计：用 自写 ReAct（第 1 段）。

⸻



一句话总结
	•	代码 1：你自己“造轮子”，完全可控、最灵活，但要处理解析/循环等坑；更适合严格格式与可审计的生产环境。
	•	代码 2：用 LangChain 的“轮子”，快速可靠；现在应改用 create_react_agent 等新接口。

'''