# LangChain 学习总览

这套目录建议按“从模型调用到生产级应用”的顺序学习。LangChain 不只是封装模型 API，它更重要的价值是把模型、Prompt、工具、检索、记忆、工作流和评估组合成可维护的 LLM 应用。

## 1. 推荐学习路线

| 顺序 | 章节 | 重点 |
|------|------|------|
| 1 | `01PROMPT` | PromptTemplate、ChatPromptTemplate、few-shot |
| 2 | `02MODEL` | 模型初始化、消息格式、stream、batch、async |
| 3 | `03OUTPUT_PARSER` | 字符串解析、JSON、Pydantic、结构化输出 |
| 4 | `04LCEL_RUNNABLE` | `prompt | model | parser`、Runnable、并行和分支 |
| 5 | `02LOAD` | 文档加载器 |
| 6 | `03SPLITTER` | 文档切分策略 |
| 7 | `07EMBEDDING_VECTORSTORE` | Embedding、向量库、Retriever |
| 8 | `08RAG` | 2-Step RAG、Agentic RAG、RAG 优化 |
| 9 | `05CHAIN` | Chain 历史概念与新版 LCEL 对照 |
| 10 | `04AGENT` | Tool、Agent、工具调用 |
| 11 | `06MEMORY` | 短期记忆、长期记忆、会话状态 |
| 12 | `12LANGGRAPH` | 复杂 Agent 编排、状态图、人工介入 |
| 13 | `13LANGSMITH` | Trace、调试、评估、线上观测 |
| 14 | `14PRODUCTION` | 安全、权限、成本、超时、重试、评估 |

## 2. LangChain 生态分工

| 模块 | 定位 | 适合场景 |
|------|------|----------|
| LangChain | LLM 应用开发框架 | Prompt、模型、工具、Agent、RAG |
| LangGraph | 有状态 Agent 工作流 | 复杂流程、循环、人工审批、失败恢复 |
| LangSmith | 可观测与评估平台 | Trace、Debug、数据集评估、线上监控 |

简单判断：

```text
固定流程：LCEL / Runnable
动态工具选择：Agent
复杂状态编排：LangGraph
调试与评估：LangSmith
```

## 3. 一个完整 LLM 应用的结构

```text
用户请求
  -> 输入校验
  -> Prompt / Agent / RAG
  -> Model
  -> Tools / Retriever / Database
  -> Output Parser / Structured Output
  -> 权限、安全、审计
  -> LangSmith Trace / Evaluation
```

## 4. 学习时要抓住的主线

| 问题 | 对应模块 |
|------|----------|
| 模型怎么调用？ | Model |
| 输入怎么组织？ | Prompt |
| 输出怎么稳定？ | OutputParser / Structured Output |
| 多步骤怎么组合？ | LCEL / Runnable |
| 外部知识怎么接入？ | RAG / Retriever |
| 外部系统怎么调用？ | Tool / Agent |
| 多轮上下文怎么保存？ | Memory |
| 复杂工作流怎么控制？ | LangGraph |
| 效果怎么评估？ | LangSmith |
| 线上怎么稳定？ | Production |

## 5. 新版和旧版写法的关系

很多老教程会使用：

```python
LLMChain
ConversationChain
initialize_agent
AgentType
RetrievalQA
```

这些概念仍然有助于理解 LangChain 的历史设计，但新代码更建议优先使用：

```python
prompt | model | parser
create_agent(...)
retriever.invoke(...)
LangGraph StateGraph
LangSmith evaluation
```

学习时可以先理解旧概念，再用新版 API 实现。

