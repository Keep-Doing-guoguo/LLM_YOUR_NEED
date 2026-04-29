
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

| 功能 | 效果 |
|------|--------|
| 自动保存对话历史 | 模型可以回忆前文，不需手动拼接 prompt |
| 控制上下文长度 | 使用 Window / Token Memory 可避免超长上下文 |
| 支持语义检索 | Vector Memory 可实现长期记忆或知识库召回 |
| 提升多轮体验 | 让 Agent 或聊天机器人更像“有记忆的人” |



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


## 新版理解：Memory 本质是状态管理

早期 LangChain 里，Memory 常常和 `ConversationChain`、`ConversationBufferMemory` 绑定使用。新版应用里，更推荐把 Memory 理解成“状态管理”：

```text
当前用户是谁
当前会话 thread 是什么
之前说过什么
哪些信息要进入短期上下文
哪些信息要写入长期记忆
哪些记忆需要按需检索
```

也就是说，Memory 不只是“把历史对话拼到 Prompt 里”。


## 短期记忆

短期记忆通常指当前会话内的消息历史。

适合保存：

- 最近几轮对话
- 当前任务状态
- Agent 中间步骤
- 当前 thread 的上下文

不要无限保存完整历史，因为会导致：

- 上下文过长
- 成本升高
- 模型被旧信息干扰
- 响应变慢

常见策略：

| 策略 | 说明 |
|------|------|
| 最近 N 轮 | 只保留最近对话 |
| Token 限制 | 超过 token 后裁剪 |
| 摘要 | 用模型总结旧历史 |
| 检索式记忆 | 只取和当前问题相关的历史 |


## 长期记忆

长期记忆用于跨会话保存信息。

适合保存：

- 用户偏好
- 用户资料
- 长期事实
- 历史任务结果
- 可复用知识

长期记忆不应该无脑写入。要考虑：

```text
这条信息是否可靠？
是否用户明确授权？
是否会过期？
是否涉及隐私？
是否需要删除能力？
```


## Runtime Context

新版 Agent 可以通过 Runtime Context 注入用户信息、配置和依赖。

示例：

```python
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class Context:
    user_id: str

agent = create_agent(
    model=llm,
    tools=[],
    context_schema=Context,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "记住我喜欢简洁回答"}]},
    context=Context(user_id="u_001"),
)
```

Runtime Context 的价值是：不要把用户信息写成全局变量，而是在每次调用时显式传入。


## Agent 里的 Memory 设计

Agent 里的 Memory 要分层：

```text
messages：当前对话消息
state：当前任务状态
store：长期记忆
retriever：语义检索历史
context：用户身份和配置
```

简单聊天可以只用 messages。复杂 Agent 建议使用 LangGraph 管理 state 和 checkpoint。


## 生产环境注意点

| 问题 | 建议 |
|------|------|
| 历史太长 | 摘要或裁剪 |
| 用户隐私 | 明确授权和可删除 |
| 错误记忆 | 写入前校验 |
| 多用户串数据 | 用 user_id / tenant_id 隔离 |
| 成本高 | 检索式记忆和摘要 |
| 记忆污染 | 不把不可信内容直接写长期记忆 |


## 新版总结

Memory 的本质是状态和上下文管理。短期记忆解决当前会话连续性，长期记忆解决跨会话个性化，复杂场景建议用 LangGraph 的状态和持久化能力来管理。
