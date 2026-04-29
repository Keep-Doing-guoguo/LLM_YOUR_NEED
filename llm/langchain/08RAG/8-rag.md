# RAG 教学

RAG 是 Retrieval-Augmented Generation，中文常叫检索增强生成。它让模型在回答前先检索外部知识，再基于检索结果生成答案。

## 1. 为什么需要 RAG

LLM 有两个天然限制：

| 限制 | 说明 |
|------|------|
| 上下文有限 | 不能一次塞进全部知识库 |
| 知识静态 | 模型训练后不知道最新或私有数据 |

RAG 的做法：

```text
用户问题
  -> 检索相关文档
  -> 把文档作为上下文
  -> 让模型基于上下文回答
```

## 2. 标准流程

```text
文档加载 Load
  -> 文档切分 Split
  -> 向量化 Embed
  -> 存入向量库 Store
  -> 用户提问 Query
  -> 检索 Retriever
  -> 组装 Prompt
  -> 模型生成 Answer
  -> 返回答案和引用来源
```

## 3. 2-Step RAG

2-Step RAG 是最稳定、最容易评估的架构。

```text
先检索，再回答
```

示例：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
请只根据上下文回答问题。如果上下文没有答案，就说“资料中没有找到”。

上下文：
{context}

问题：
{question}
""")

rag_chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough(),
} | prompt | model | StrOutputParser()

answer = rag_chain.invoke("LangChain Agent 是什么？")
print(answer)
```

## 4. 返回引用来源

RAG 不应该只返回答案，还要告诉用户依据来自哪里。

```python
question = "Milvus 是什么？"
docs = retriever.invoke(question)

answer = (prompt | model | StrOutputParser()).invoke({
    "context": format_docs(docs),
    "question": question,
})

sources = [doc.metadata.get("source") for doc in docs]

print(answer)
print("来源：", sources)
```

## 5. Agentic RAG

Agentic RAG 是让 Agent 自己决定什么时候检索。

```text
用户问题
  -> Agent 判断是否需要检索
  -> 调用检索工具
  -> 分析结果
  -> 必要时再次检索
  -> 回答
```

适合：

- 多跳问题
- 问题需要拆解
- 需要多个工具协作
- 检索条件不确定

风险：

- 延迟更高
- token 成本更高
- 工具调用可能循环
- 评估更复杂

## 6. 把 Retriever 封装成 Tool

```python
from langchain.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库，输入自然语言问题，返回相关文档片段。"""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)
```

然后交给 Agent：

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[search_knowledge_base],
    system_prompt="你是一个知识库问答助手，需要时调用知识库检索工具。",
)
```

## 7. RAG 优化方向

| 方向 | 说明 |
|------|------|
| 文档清洗 | 去页眉页脚、乱码、重复内容 |
| 切分策略 | 控制 chunk_size 和 overlap |
| Embedding 模型 | 根据语言和领域选择 |
| Metadata 过滤 | 按用户、租户、文档类型过滤 |
| Query Rewrite | 把用户问题改写成更适合检索的查询 |
| Multi Query | 生成多个查询提高召回 |
| Hybrid Search | 向量检索 + 关键词检索 |
| Rerank | 对候选文档二次排序 |
| Context Compression | 压缩上下文，减少无关内容 |
| Citation | 返回引用来源 |

## 8. 常见失败原因

| 现象 | 可能原因 |
|------|----------|
| 答案胡编 | 上下文没有答案，Prompt 没有限制 |
| 找不到相关内容 | chunk 太大或太小，embedding 不匹配 |
| 答案不完整 | top_k 太小，召回不足 |
| 答案很慢 | rerank 太重，上下文太长 |
| 答案引用错误 | metadata 丢失或 chunk 来源不清楚 |

## 9. RAG 评估

至少评估四类指标：

| 指标 | 含义 |
|------|------|
| 检索相关性 | 找到的文档是否相关 |
| 答案正确性 | 答案是否解决问题 |
| 忠实性 | 答案是否基于上下文 |
| 引用准确性 | 引用来源是否支撑答案 |

可以用 LangSmith 建数据集做离线评估。

## 10. 小结

RAG 的核心不是“接向量库”，而是让检索结果真正支撑模型回答。生产级 RAG 需要同时处理召回、重排、引用、权限、评估和成本。

