# LangSmith 教学

LangSmith 用来调试、追踪、评估和监控 LLM 应用。LangChain 应用只跑通不够，还要知道它为什么这么回答、哪一步失败、成本是多少、版本更新后有没有退化。

## 1. LangSmith 解决什么问题

LLM 应用的问题通常不是简单异常，而是链路质量问题：

```text
Prompt 是否正确？
模型输入是什么？
检索到了哪些文档？
工具有没有调用错？
输出为什么幻觉？
哪个版本效果更好？
线上失败样本有哪些？
```

LangSmith 可以记录完整 trace，帮助你定位这些问题。

## 2. 基础配置

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="你的 LangSmith API Key"
export LANGSMITH_PROJECT="langchain-course"
```

配置后，LangChain / LangGraph 调用会自动记录 trace。

## 3. Trace 是什么

Trace 是一次请求的完整执行树。

它通常包含：

| 内容 | 说明 |
|------|------|
| 用户输入 | 原始问题 |
| Prompt | 最终发送给模型的内容 |
| Model Call | 模型名称、参数、输出 |
| Tool Call | 工具名称、参数、结果 |
| Retriever | 检索 query 和返回文档 |
| Token | 输入输出 token |
| Latency | 每一步耗时 |
| Error | 错误堆栈 |

## 4. 为什么 Trace 很重要

没有 trace 时，你只能看到最终答案。

有 trace 后可以判断：

```text
是 Prompt 写错？
是检索结果错？
是模型没遵守上下文？
是工具参数错？
是解析器失败？
是成本突然升高？
```

## 5. Debug RAG

RAG 出错时，优先看：

| 检查项 | 说明 |
|--------|------|
| query | 用户问题是否被正确传给 retriever |
| retrieved docs | 检索内容是否相关 |
| context | 上下文是否过长或包含无关内容 |
| prompt | 是否要求模型基于上下文回答 |
| answer | 是否忠实于上下文 |

常见定位：

```text
检索错：调 embedding、chunk、top_k、rerank
生成错：调 prompt、模型、结构化输出
权限错：查 metadata filter
```

## 6. Debug Agent

Agent 出错时，重点看：

| 检查项 | 说明 |
|--------|------|
| tool selection | 是否选错工具 |
| tool args | 工具参数是否正确 |
| tool result | 工具结果是否可用 |
| iteration count | 是否循环调用 |
| final answer | 是否正确吸收工具结果 |

如果 Agent 不调用工具，通常是：

- 工具 description 不清晰
- 模型工具调用能力弱
- system prompt 没明确要求
- 用户问题不需要工具

## 7. Dataset

Dataset 是评估集，包含输入和期望输出。

示例：

```text
question: "LangChain 是什么？"
expected: "用于构建 LLM 应用的框架"
```

数据集来源：

- 人工整理
- 线上失败样本
- 历史 trace
- 合成数据

## 8. Evaluation

评估分两类：

| 类型 | 说明 |
|------|------|
| Offline Evaluation | 上线前用固定数据集评估 |
| Online Evaluation | 线上真实流量中评估 |

评估器可以是：

- 代码规则
- 人工标注
- LLM-as-judge
- pairwise comparison

## 9. 常见评估指标

| 指标 | 说明 |
|------|------|
| Correctness | 答案是否正确 |
| Relevance | 是否回答了问题 |
| Faithfulness | 是否基于上下文 |
| Retrieval relevance | 检索文档是否相关 |
| Tool accuracy | 工具是否调用正确 |
| Latency | 延迟 |
| Cost | token 成本 |

## 10. LLM-as-judge 的风险

LLM 评估不是绝对真理。

风险：

- 评估模型也会幻觉
- 标准不稳定
- 对长答案偏好明显
- 对格式错误不敏感
- 可能和业务标准不一致

建议：

```text
关键指标用代码规则
复杂语义用 LLM-as-judge
高风险样本做人审
保留失败样本回归测试
```

## 11. 生产中怎么用

建议流程：

```text
开发阶段：打开 tracing，调 prompt 和 chain
测试阶段：构建 dataset，跑 offline eval
上线阶段：采集线上 trace，做 online eval
迭代阶段：把失败样本加入 regression dataset
```

## 12. 小结

LangSmith 让 LLM 应用从“感觉还行”变成“可观察、可评估、可回归”。只要做 RAG 或 Agent，就应该尽早接入 trace 和评估。

