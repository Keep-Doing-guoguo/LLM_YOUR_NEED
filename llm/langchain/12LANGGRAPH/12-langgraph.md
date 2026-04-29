# LangGraph 教学

LangGraph 是 LangChain 生态中用于构建复杂 Agent 和有状态工作流的框架。简单 Agent 用 LangChain 的 `create_agent` 就够了；如果任务需要状态、循环、条件分支、人工审批、失败恢复，就应该考虑 LangGraph。

## 1. LangGraph 解决什么问题

普通 Chain 适合固定流程：

```text
Prompt -> Model -> Parser
```

普通 Agent 适合动态工具调用：

```text
用户问题 -> 模型决定工具 -> 工具结果 -> 最终回答
```

LangGraph 适合复杂流程：

```text
分类 -> 检索 -> 判断是否足够 -> 不足则继续检索 -> 人工审批 -> 执行操作 -> 结束
```

## 2. 适合 LangGraph 的场景

| 场景 | 说明 |
|------|------|
| 多步骤 Agent | 每一步有明确状态 |
| 循环执行 | 检索不够继续检索 |
| 条件分支 | 根据分类走不同路径 |
| 多 Agent 协作 | 规划 Agent、执行 Agent、审查 Agent |
| 人工介入 | 审批后继续执行 |
| 长任务 | 中断后恢复 |
| 生产工作流 | 可观测、可恢复、可控制 |

## 3. 核心概念

| 概念 | 说明 |
|------|------|
| State | 工作流共享状态 |
| Node | 一个处理节点 |
| Edge | 节点之间的流转 |
| Conditional Edge | 条件路由 |
| START / END | 起点和终点 |
| Checkpointer | 状态持久化 |
| Interrupt | 人工中断与恢复 |

## 4. 最小示例

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def call_model(state: MessagesState):
    return {
        "messages": [
            {"role": "ai", "content": "hello world"}
        ]
    }

graph = StateGraph(MessagesState)
graph.add_node("call_model", call_model)
graph.add_edge(START, "call_model")
graph.add_edge("call_model", END)

app = graph.compile()

result = app.invoke({
    "messages": [
        {"role": "user", "content": "hi"}
    ]
})

print(result)
```

## 5. 自定义 State

State 是每个节点共享和更新的数据。

```python
from typing import TypedDict

class QAState(TypedDict):
    question: str
    category: str
    context: str
    answer: str
```

节点函数接收 state，返回要更新的字段。

```python
def classify(state: QAState):
    question = state["question"]
    if "订单" in question:
        category = "order"
    else:
        category = "general"
    return {"category": category}
```

## 6. 条件分支

```python
def route_by_category(state: QAState):
    if state["category"] == "order":
        return "query_order"
    return "general_answer"

graph.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "query_order": "query_order",
        "general_answer": "general_answer",
    },
)
```

适合：

```text
问题分类
权限判断
是否需要检索
是否需要人工审批
是否继续循环
```

## 7. 一个 RAG 工作流示意

```text
START
  -> classify_question
  -> retrieve_docs
  -> judge_context_enough
  -> generate_answer
  -> END
```

如果上下文不够：

```text
judge_context_enough -> rewrite_query -> retrieve_docs
```

这就是 LangGraph 比普通 Chain 更适合复杂 RAG 的地方。

## 8. Durable Execution

Durable execution 指工作流能保存进度，中断后从上次状态恢复。

适合：

- 长任务
- 多步骤任务
- 人工审批
- 外部 API 不稳定
- 需要回放和审计

使用 checkpointer 后，LangGraph 可以保存每一步状态。

## 9. Human-in-the-loop

人工介入常见场景：

```text
发送邮件前审批
执行退款前审批
删除数据前审批
工具返回高风险结果时审批
```

思路：

```text
Agent 生成计划
  -> 暂停
  -> 人工查看 state
  -> 人工修改或批准
  -> 继续执行
```

## 10. LangChain Agent 和 LangGraph 的关系

| 对比项 | LangChain Agent | LangGraph |
|--------|-----------------|-----------|
| 抽象层级 | 高层封装 | 底层编排 |
| 上手难度 | 低 | 中高 |
| 状态控制 | 简单 | 强 |
| 条件分支 | 有限 | 强 |
| 持久化 | 依赖底层 | 原生支持 |
| 人工介入 | 可配置 | 更灵活 |
| 适合场景 | 简单工具调用 | 复杂 Agent 工作流 |

## 11. 使用建议

```text
简单问答：不用 LangGraph
固定流程：LCEL
简单工具 Agent：create_agent
复杂状态流程：LangGraph
需要恢复和人工审批：LangGraph
```

## 12. 小结

LangGraph 的核心是把 Agent 从“黑盒循环”变成“可控状态机”。当你开始关心状态、分支、循环、恢复、人工审批时，就应该学习 LangGraph。

