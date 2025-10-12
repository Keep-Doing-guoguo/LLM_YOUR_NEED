

## 🧩 一、什么是 Chain

Chain 就像流水线。

它定义了：

	•	输入 → 处理（Prompt + 模型 + 逻辑）→ 输出。
	•	可以是单步（如 LLMChain），也可以是多步（如 SequentialChain）。

举个简单例子👇：

```python


from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("请用一句话总结：{text}")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("LangChain 是一个用于构建 LLM 应用的框架。")
print(result)

#输出示例：

#LangChain 是一个帮助开发者快速构建基于大模型的应用框架。
```



## 🧠 二、Chain 的常见类型

| 类型 | 说明 | 示例 |
|------|------|------|
| **LLMChain** | 最基础的链：Prompt + LLM + 输出 | 文本总结、问答、改写 |
| **SimpleSequentialChain** | 串行执行多个链，一个输出是下一个输入 | 先总结 → 再翻译 |
| **SequentialChain** | 多输入多输出版本，可指定输入/输出变量 | 复杂管道 |
| **TransformChain** | 在链中做数据预处理/后处理（非LLM逻辑） | 解析 JSON、过滤文本 |
| **RouterChain** | 根据输入内容自动选择不同子链 | 多任务场景（分类 → 不同处理逻辑） |
| **ConversationalChain** | 保留对话上下文 | 聊天机器人 |
| **RetrievalQAChain** | 结合检索器（RAG）执行问答 | 知识库问答系统 |




## 🔗 三、链的组合方式

✅ 单链

最常见的 LLMChain，用于单步生成。

✅ 多链组合

```
SimpleSequentialChain 串行执行多个子链：

from langchain.chains import SimpleSequentialChain

chain = SimpleSequentialChain(chains=[chain_summary, chain_translate])
result = chain.run("LangChain 是一个开源框架，用于构建基于大语言模型的应用。")
```



## ⚙️ 四、自定义 Chain

你可以继承 Chain 自己写逻辑：
```
from langchain.chains.base import Chain

class MyChain(Chain):
    @property
    def input_keys(self):
        return ["text"]

    @property
    def output_keys(self):
        return ["result"]

    def _call(self, inputs):
        text = inputs["text"]
        result = text[::-1]  # 简单示例：反转字符串
        return {"result": result}

chain = MyChain()
print(chain.run("LangChain"))
# 输出: niahCgnaL
```



## 📘 五、使用 Chain 的典型场景


| 场景 | 使用 Chain 类型 |
|------|----------------|
| **问答系统（输入问题 → 查询知识 → 生成答案）** | `RetrievalQAChain` |
| **多轮对话** | `ConversationalChain` |
| **文本摘要 → 翻译** | `SimpleSequentialChain` |
| **智能 Agent（调用工具）** | `AgentExecutor`（底层也是 `Chain`） |
| **自定义业务流程** | 继承 `Chain` 自写 `_call()` |


## 六、llmchain和conversationchain

| 对比项 | LLMChain | ConversationalChain |
|--------|-----------|--------------------|
| **核心作用** | 单轮调用：Prompt + LLM → 输出 | 多轮对话：保留上下文历史 |
| **是否有记忆（Memory）** | ❌ 无记忆，每次输入独立 | ✅ 有记忆，会自动拼接历史对话 |
| **典型用途** | 问答、改写、摘要等单次任务 | 聊天机器人、多轮交互 |
| **输入结构** | `{ "input": "问题内容" }` | `{ "input": "用户当前问题" }` + 自动携带历史 |
| **是否自动管理上下文** | 否（需手动拼接历史） | 是（内置 `ConversationBufferMemory`） |
| **实现逻辑** | `prompt → llm → output` | `memory.load_memory_variables()` + prompt + llm |
| **适用场景** | 一次性推理任务 | 连续对话（客服、聊天助手） |

## 🚀 六、总结



| 概念 | 作用 |
|------|------|
| **Chain** | 管理流程（从输入到输出） |
| **LLMChain** | `Prompt + LLM` 的单步链 |
| **SequentialChain** | 串行执行多个链 |
| **Agent** | 运行时动态决定“下一步调用哪个工具”的智能链 |
