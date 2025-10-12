
## 🧠 什么是 Memory？

在 LangChain 中，Memory 用于在多轮对话中保存和管理上下文信息。
它会自动记录用户输入 (input) 与模型输出 (output)，
并在下一轮调用时自动把历史对话注入到 Prompt 中，让 LLM 有“记忆”能力。



### 📚 常见 Memory 类型对比

| Memory 类型 | 特点 | 适用场景 |
|--------------|------|-----------|
| **ConversationBufferMemory** | 保存完整的对话历史（按字符串拼接） | 简单聊天场景 |
| **ConversationBufferWindowMemory** | 只保留最近 N 轮对话 | 控制上下文长度，节省 token |
| **ConversationSummaryMemory** | 用 LLM 总结历史对话，再保存摘要 | 长对话（节省上下文空间） |
| **ConversationTokenBufferMemory** | 按 token 数量控制记忆范围 | 精确控制上下文长度 |
| **VectorStoreRetrieverMemory** | 将历史内容向量化存入向量库（如 Milvus/FAISS），支持语义检索 | 长期知识记忆、知识问答 |
| **EntityMemory** | 按“实体（人物/地点/事件）”追踪信息 | 角色扮演、多角色记忆场景 |




### ⚙️ 工作原理（以 ConversationBufferMemory 为例）
```python


from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(model="qwen-plus", temperature=0)
memory = ConversationBufferMemory()  # 保存全部历史记录

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print(conversation.predict(input="你好，我叫张三"))
print(conversation.predict(input="我刚才说我叫什么名字？"))
```
输出：

> Memory 会自动注入：
Human: 你好，我叫张三
AI: 你好张三！
Human: 我刚才说我叫什么名字？
AI: 你说你叫张三。




### 💡 使用 Memory 的好处

功能	效果
自动保存对话历史	模型可以回忆前文，不需手动拼接 prompt
控制上下文长度	使用 Window / Token Memory 可避免超长上下文
支持语义检索	Vector Memory 可实现长期记忆或知识库召回
提升多轮体验	让 Agent 或聊天机器人更像“有记忆的人”




### 🧩 在 Agent 中使用 Memory
```
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=[],
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

agent.run("你好")
agent.run("我刚才说了什么？")
```



✅ 一句话总结：

Memory 就是 LangChain 让模型“记住上下文”的机制。 可以选择不同类型的记忆方式（Buffer、Window、Summary、Vector）来平衡 记忆能力 vs 成本。