# LangChain Model 教学

Model 是 LangChain 应用的推理核心。Prompt 负责组织输入，Model 负责生成回复，OutputParser 负责把回复变成业务可用的数据。

## 1. Model 的作用

大模型可以完成：

- 文本生成
- 分类
- 摘要
- 翻译
- 信息抽取
- 工具调用
- 结构化输出
- 多模态理解
- 多步推理

LangChain 的价值是提供统一接口，让你更容易切换 OpenAI、Anthropic、Qwen、Gemini、Azure、Bedrock 等模型。

## 2. 初始化模型

推荐写法之一是使用具体 provider 的模型类。

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

response = model.invoke("LangChain 是什么？")
print(response.content)
```

如果使用 DashScope 的 OpenAI 兼容接口：

```python
import os
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

## 3. Message 消息格式

聊天模型通常使用消息列表，而不是单个字符串。

| 消息 | 作用 |
|------|------|
| `SystemMessage` | 系统规则、角色、边界 |
| `HumanMessage` | 用户输入 |
| `AIMessage` | 模型回复 |
| `ToolMessage` | 工具调用结果 |

```python
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是一个严谨的 AI 工程讲师。"),
    HumanMessage(content="用三句话解释 RAG。"),
]

response = model.invoke(messages)
print(response.content)
```

## 4. 常用调用方式

| 方法 | 说明 |
|------|------|
| `invoke` | 单次调用 |
| `stream` | 流式输出 |
| `batch` | 批量调用 |
| `ainvoke` | 异步单次调用 |
| `astream` | 异步流式输出 |

## 5. invoke

```python
response = model.invoke("解释一下 Tool Calling")
print(response.content)
```

## 6. stream

```python
for chunk in model.stream("请逐步解释 LangChain Agent"):
    print(chunk.content, end="")
```

适合聊天界面、长文本生成、实时反馈。

## 7. batch

```python
questions = [
    "什么是 PromptTemplate？",
    "什么是 Retriever？",
    "什么是 LangGraph？",
]

responses = model.batch(questions)

for response in responses:
    print(response.content)
```

适合离线批处理、评估集运行、批量摘要。

## 8. async

```python
import asyncio

async def main():
    response = await model.ainvoke("异步调用适合什么场景？")
    print(response.content)

asyncio.run(main())
```

适合 Web 服务、并发调用、长链路应用。

## 9. 常用参数

| 参数 | 说明 |
|------|------|
| `model` | 模型名称 |
| `temperature` | 随机性，越低越稳定 |
| `max_tokens` | 最大输出长度 |
| `timeout` | 超时时间 |
| `max_retries` | 最大重试次数 |
| `api_key` | 模型服务 API Key |
| `base_url` | OpenAI 兼容接口地址 |

建议：

```text
分类、抽取、RAG：temperature=0
创作、头脑风暴：temperature 可以稍高
生产服务：设置 timeout 和 max_retries
```

## 10. Model 在 Agent 中的作用

在 Agent 中，Model 不只是生成回答，还要判断：

```text
是否需要调用工具
调用哪个工具
工具参数是什么
是否继续调用工具
什么时候返回最终答案
```

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个助手。"
)
```

## 11. 选择模型时看什么

| 能力 | 影响 |
|------|------|
| 上下文窗口 | 能放多少历史和检索内容 |
| Tool Calling | Agent 是否稳定 |
| Structured Output | 结构化返回是否可靠 |
| 推理能力 | 复杂任务表现 |
| 价格 | 成本 |
| 延迟 | 用户体验 |
| 多模态 | 是否支持图片、音频、视频 |

## 12. 常见错误

| 问题 | 原因 |
|------|------|
| API Key 报错 | 环境变量未配置 |
| 模型不存在 | model 名称和 provider 不匹配 |
| 输出不稳定 | temperature 太高或 Prompt 约束不足 |
| 流式输出为空 | chunk 内容字段读取方式不对 |
| Agent 不调用工具 | 模型不支持工具调用或工具描述不清楚 |

## 13. 小结

Model 是 LangChain 的推理引擎。掌握 Model 后，后面 Prompt、Parser、LCEL、Agent、RAG 都是在它周围组织工程结构。

