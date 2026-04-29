# LCEL 与 Runnable

LCEL 是 LangChain Expression Language，用来把 Prompt、Model、Parser、函数等组件组合成可执行流程。新版 LangChain 中，很多传统 Chain 的能力都可以用 LCEL 更清晰地表达。

## 1. 最核心写法

```python
chain = prompt | model | parser
```

含义：

```text
输入变量 -> Prompt -> Model -> Parser -> 输出
```

完整例子：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用三句话解释：{topic}")
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "RAG"})
print(result)
```

## 2. Runnable 是什么

Runnable 是 LangChain 中“可运行组件”的统一接口。

常见 Runnable：

| 组件 | 说明 |
|------|------|
| Prompt | 输入 dict，输出 messages |
| Model | 输入 messages，输出 AIMessage |
| Parser | 输入 AIMessage，输出字符串或结构 |
| RunnableLambda | 包装普通 Python 函数 |
| RunnableParallel | 并行执行多个分支 |
| RunnablePassthrough | 原样传递输入 |

## 3. Runnable 的通用方法

| 方法 | 作用 |
|------|------|
| `invoke` | 单次执行 |
| `batch` | 批量执行 |
| `stream` | 流式执行 |
| `ainvoke` | 异步执行 |

```python
chain.invoke({"topic": "Agent"})
chain.batch([{"topic": "RAG"}, {"topic": "Tool"}])
```

## 4. RunnableLambda

把普通函数放进链里。

```python
from langchain_core.runnables import RunnableLambda

def clean_text(text: str) -> str:
    return text.strip().replace("\n", " ")

cleaner = RunnableLambda(clean_text)

chain = cleaner | prompt | model | parser

result = chain.invoke("  LangChain\n是什么？  ")
```

## 5. RunnableParallel

并行生成多个输入字段。

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

setup = RunnableParallel({
    "question": RunnablePassthrough(),
    "context": lambda question: "这里是检索到的上下文：" + question,
})

chain = setup | prompt | model | parser

result = chain.invoke("LangChain Agent 是什么？")
```

输出给 Prompt 的结构是：

```python
{
    "question": "...",
    "context": "..."
}
```

## 6. RunnablePassthrough

保留原始输入，常用于 RAG。

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = {
    "context": retriever,
    "question": RunnablePassthrough(),
} | prompt | model | parser
```

含义：

```text
question 一路传给 Prompt
context 由 retriever 根据 question 检索出来
```

## 7. LCEL 和旧 Chain 的关系

旧写法：

```python
chain = LLMChain(llm=model, prompt=prompt)
chain.run(...)
```

新版更推荐：

```python
chain = prompt | model | parser
chain.invoke(...)
```

| 旧概念 | 新版替代 |
|--------|----------|
| LLMChain | `prompt | model | parser` |
| TransformChain | `RunnableLambda` |
| SequentialChain | LCEL 串联 |
| RouterChain | Runnable 分支 / LangGraph |
| RetrievalQA | Retriever + LCEL RAG |

## 8. 什么时候用 LCEL

适合：

- 摘要
- 翻译
- 分类
- 信息抽取
- 固定 RAG
- 多步骤但流程确定的任务

不适合：

- 复杂循环
- 多 Agent 协作
- 人工审批
- 长任务恢复

这些更适合 LangGraph。

## 9. 小结

LCEL 是新版 LangChain 的核心组合方式。固定流程优先用 LCEL，动态工具选择用 Agent，复杂状态编排用 LangGraph。

