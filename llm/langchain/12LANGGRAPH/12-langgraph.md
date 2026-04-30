# LangGraph 常见组件与完整教学

LangGraph 是 LangChain 生态中专门用于构建 **复杂 Agent、有状态工作流、多步骤任务、人工审批、失败恢复** 的框架。

如果说 LCEL 适合固定链路：

```text
Prompt -> Model -> Parser
```

那么 LangGraph 更适合这种流程：

```text
用户输入
  -> 分类
  -> 检索
  -> 判断是否足够
  -> 不足则改写问题再次检索
  -> 调用工具
  -> 人工审批
  -> 生成最终结果
```

## 1. LangGraph 要学哪些内容？

| 模块 / 概念 | 功能说明 | 推荐场景 |
|-------------|----------|----------|
| `StateGraph` | 构建状态图的核心类 | 所有 LangGraph 工作流 |
| `State` | 保存图运行过程中的共享状态 | 问题、分类、上下文、答案、工具结果 |
| `Node` | 图中的一个执行步骤，本质是函数 | 分类、检索、生成、审批、工具调用 |
| `Edge` | 节点之间的固定连接 | 顺序流程 |
| `Conditional Edge` | 根据状态决定下一步走向 | 分支、路由、循环 |
| `START / END` | 图的起点和终点 | 定义流程开始和结束 |
| `Reducer` | 控制多个节点如何合并状态 | 消息追加、日志追加、结果合并 |
| `MessagesState` | 内置消息状态 | 聊天机器人、Agent |
| `ToolNode` | 执行模型请求的工具调用 | Tool Calling Agent |
| `Checkpointer` | 保存每一步状态 | 多轮对话、恢复、人审 |
| `interrupt / Command` | 暂停并恢复流程 | human-in-the-loop |
| `stream` | 流式观察图执行过程 | 前端展示、调试 |
| `Subgraph` | 子图复用 | 多模块复杂工作流 |
| Multi-Agent | 多 Agent 编排 | 规划、执行、审查、协作 |

## 2. LangGraph 和 Chain / Agent 的区别

| 类型 | 特点 | 适合场景 |
|------|------|----------|
| LCEL / Chain | 流程固定，输入输出明确 | 摘要、翻译、分类、普通 RAG |
| LangChain Agent | 模型自动决定是否调用工具 | 简单工具调用 |
| LangGraph | 显式定义状态、节点、分支、循环 | 复杂 Agent、长任务、人审、恢复 |

一句话：

```text
固定流程用 LCEL；
简单工具调用用 Agent；
复杂有状态流程用 LangGraph。
```

## 3. 安装

✅ 安装 LangGraph：

```bash
pip install -U langgraph
```

如果要结合 LangChain 模型：

```bash
pip install -U langchain langchain-openai
```

如果要使用 LangSmith 追踪：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="你的 LangSmith API Key"
```

📘 特点

- LangGraph 可以单独使用；
- 也可以和 LangChain 的模型、工具、Retriever 一起使用；
- LangChain 的新版 Agent 底层也和 LangGraph 生态关系很紧密。

## 4. StateGraph —— 构建图的核心类

`StateGraph` 是 LangGraph 最核心的类，用来创建一个“状态图”。

它需要一个 State 类型，用来声明整个图会维护哪些字段。

✅ 示例代码

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class SimpleState(TypedDict):
    input: str
    output: str


def answer_node(state: SimpleState):
    return {
        "output": f"你输入的是：{state['input']}"
    }


builder = StateGraph(SimpleState)

builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile()

result = graph.invoke({
    "input": "你好 LangGraph",
    "output": "",
})

print(result)
```

输出示例：

```python
{
    "input": "你好 LangGraph",
    "output": "你输入的是：你好 LangGraph"
}
```

📘 特点

- `StateGraph` 用来定义流程；
- `compile()` 之后才变成可以运行的 graph；
- 节点返回的字段会更新到 State 中；
- 图的执行入口一般是 `invoke()` 或 `stream()`。

## 5. State —— 图中的共享状态

`State` 是 LangGraph 的核心。

它表示整个工作流运行过程中会共享和更新的数据。

✅ 示例代码

```python
from typing_extensions import TypedDict


class QAState(TypedDict):
    question: str
    category: str
    context: str
    answer: str
```

这个 State 表示：

| 字段 | 含义 |
|------|------|
| `question` | 用户问题 |
| `category` | 问题分类 |
| `context` | 检索到的上下文 |
| `answer` | 最终答案 |

节点可以读取 State：

```python
def classify_node(state: QAState):
    question = state["question"]

    if "订单" in question:
        return {"category": "order"}

    return {"category": "general"}
```

📘 特点

- 节点不需要返回完整 State；
- 只返回要更新的字段即可；
- State 设计越清晰，图越容易维护；
- 不建议把超大文档、API Key、无关日志都塞进 State。

## 6. Node —— 图中的执行节点

Node 本质就是一个 Python 函数。

它接收当前 State，返回要更新的 State 字段。

✅ 示例代码

```python
def retrieve_node(state: QAState):
    question = state["question"]

    if "LangGraph" in question:
        context = "LangGraph 是用于构建有状态 Agent 工作流的框架。"
    else:
        context = "没有检索到相关资料。"

    return {"context": context}
```

再定义一个生成答案的节点：

```python
def generate_node(state: QAState):
    return {
        "answer": f"根据资料回答：{state['context']}"
    }
```

📘 特点

- 一个节点最好只做一件事；
- 分类、检索、生成、审批、工具调用都可以拆成独立节点；
- 节点越小，越容易调试和复用；
- 不建议写一个 `do_everything()` 节点把所有逻辑都塞进去。

## 7. Edge —— 固定流程连接

`Edge` 用来连接节点，表示固定执行顺序。

✅ 示例代码

```python
builder = StateGraph(QAState)

builder.add_node("classify", classify_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
```

执行顺序：

```text
START -> classify -> retrieve -> generate -> END
```

📘 特点

- 固定流程用普通 `add_edge`；
- 起点用 `START`；
- 终点用 `END`；
- 简单线性工作流只需要普通 Edge。

## 8. Conditional Edge —— 条件分支

当下一步取决于当前状态时，就用条件边。

比如：

```text
如果是订单问题 -> 查询订单
如果是普通问题 -> 普通回答
```

✅ 示例代码

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class RouteState(TypedDict):
    question: str
    category: str
    answer: str


def classify_node(state: RouteState):
    if "订单" in state["question"]:
        return {"category": "order"}
    return {"category": "general"}


def order_node(state: RouteState):
    return {"answer": "这是订单问题，订单 A1001 当前状态：已发货。"}


def general_node(state: RouteState):
    return {"answer": "这是普通问题，我可以直接回答。"}


def route_node(state: RouteState):
    if state["category"] == "order":
        return "order"
    return "general"


builder = StateGraph(RouteState)
builder.add_node("classify", classify_node)
builder.add_node("order", order_node)
builder.add_node("general", general_node)

builder.add_edge(START, "classify")

builder.add_conditional_edges(
    "classify",
    route_node,
    {
        "order": "order",
        "general": "general",
    },
)

builder.add_edge("order", END)
builder.add_edge("general", END)

graph = builder.compile()

result = graph.invoke({
    "question": "帮我查订单 A1001",
    "category": "",
    "answer": "",
})

print(result["answer"])
```

📘 特点

- `route_node` 返回的是路由名称；
- 路由名称必须和 mapping 中的 key 对上；
- 条件边适合分类、权限判断、是否检索、是否人工审批、是否继续循环。

## 9. 循环 —— 让图可以重复执行某些节点

LangGraph 可以显式定义循环。

典型场景：

```text
检索 -> 判断资料是否足够
  -> 足够：生成答案
  -> 不足：改写问题，再检索
```

✅ 示例代码

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    retry_count: int
    enough: bool
    answer: str


def retrieve_node(state: RAGState):
    context = f"用 query 检索资料：{state['query']}"
    return {"context": context}


def judge_node(state: RAGState):
    if "LangGraph" in state["context"]:
        return {"enough": True}

    if state["retry_count"] >= 2:
        return {"enough": True}

    return {"enough": False}


def rewrite_node(state: RAGState):
    return {
        "query": state["question"] + " LangGraph 官方文档",
        "retry_count": state["retry_count"] + 1,
    }


def generate_node(state: RAGState):
    return {
        "answer": f"基于上下文生成答案：{state['context']}"
    }


def route_after_judge(state: RAGState):
    if state["enough"]:
        return "generate"
    return "rewrite"


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("judge", judge_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "judge")
builder.add_conditional_edges(
    "judge",
    route_after_judge,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)

graph = builder.compile()

result = graph.invoke({
    "question": "LangGraph 的 checkpoint 是什么？",
    "query": "checkpoint 是什么",
    "context": "",
    "retry_count": 0,
    "enough": False,
    "answer": "",
})

print(result["answer"])
```

📘 特点

- 循环是 LangGraph 的强项；
- 必须设置退出条件；
- 常见退出条件是 `retry_count`、最大工具调用次数、置信度阈值；
- 不加限制容易死循环。

## 10. Reducer —— 控制状态如何合并

默认情况下，节点返回的新字段会覆盖旧字段。

如果你希望某个字段是“追加”，就需要 Reducer。

✅ 示例代码

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class LogState(TypedDict):
    logs: Annotated[list[str], operator.add]


def step_one(state: LogState):
    return {"logs": ["step one finished"]}


def step_two(state: LogState):
    return {"logs": ["step two finished"]}


builder = StateGraph(LogState)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)

builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)

graph = builder.compile()

result = graph.invoke({"logs": []})
print(result["logs"])
```

输出：

```python
[
    "step one finished",
    "step two finished"
]
```

📘 特点

- 默认是覆盖；
- Reducer 可以实现追加；
- 常用于消息列表、日志、中间步骤、多分支结果合并。

## 11. MessagesState —— 聊天 Agent 常用状态

`MessagesState` 是 LangGraph 内置的消息状态。

它适合聊天机器人、Agent、多轮对话。

✅ 示例代码

```python
from langgraph.graph import StateGraph, MessagesState, START, END


def chat_node(state: MessagesState):
    user_message = state["messages"][-1].content
    return {
        "messages": [
            {"role": "ai", "content": f"你刚才说：{user_message}"}
        ]
    }


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()

result = graph.invoke({
    "messages": [
        {"role": "user", "content": "你好"}
    ]
})

print(result["messages"][-1].content)
```

📘 特点

- `messages` 会自动追加；
- 不会每次覆盖旧消息；
- 很适合构建聊天 Agent；
- 如果需要额外字段，可以自定义消息状态。

自定义消息状态：

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
```

## 12. 接入真实 LLM

上面的例子没有真正调用模型。下面接入一个 OpenAI 兼容模型。

✅ 示例代码

```python
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END


model = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def call_model_node(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model_node)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

graph = builder.compile()

result = graph.invoke({
    "messages": [
        {"role": "user", "content": "用三句话解释 LangGraph"}
    ]
})

print(result["messages"][-1].content)
```

📘 特点

- LangGraph 只负责编排；
- 模型仍然使用 LangChain 的模型接口；
- 节点里可以调用模型、工具、Retriever、数据库。

## 13. ToolNode —— 工具调用节点

`ToolNode` 是 LangGraph 预置的工具执行节点。

适合构建 Tool Calling Agent。

✅ 示例代码

```python
from langchain.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    data = {
        "北京": "晴，24 摄氏度。",
        "上海": "多云，26 摄氏度。",
        "深圳": "阵雨，29 摄氏度。",
    }
    return data.get(city, f"{city} 暂无天气数据。")


tools = [get_weather]
model_with_tools = model.bind_tools(tools)


def agent_node(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

result = graph.invoke({
    "messages": [
        {"role": "user", "content": "上海天气怎么样？"}
    ]
})

print(result["messages"][-1].content)
```

流程：

```text
START -> agent
agent 判断是否需要工具
如果需要工具 -> tools
tools 执行后 -> 回到 agent
如果不需要工具 -> END
```

📘 特点

- `ToolNode` 负责执行工具；
- `tools_condition` 负责判断模型有没有请求工具；
- 这是 LangGraph 构建工具 Agent 的常见写法；
- 比黑盒 Agent 更容易控制流程。

## 14. Checkpointer —— 状态持久化

Checkpointer 用于保存每一步状态。

它可以支持：

- 多轮对话；
- 失败恢复；
- 人工审批；
- time travel；
- 长任务继续执行。

✅ 示例代码

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END


checkpointer = InMemorySaver()


def chat_node(state: MessagesState):
    return {
        "messages": [
            {"role": "ai", "content": "我会记住当前 thread 的历史。"}
        ]
    }


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "user-001"
    }
}

graph.invoke(
    {"messages": [{"role": "user", "content": "你好，我叫张三"}]},
    config=config,
)

graph.invoke(
    {"messages": [{"role": "user", "content": "我叫什么？"}]},
    config=config,
)
```

📘 特点

- `thread_id` 表示一次会话或任务；
- 同一个 `thread_id` 会共享历史状态；
- 没有 checkpointer，就不能很好地支持恢复和人审；
- 生产环境应使用持久化存储，而不是只用内存。

## 15. Human-in-the-loop —— 人工介入

人工介入用于让流程暂停，等待人确认后继续。

适合：

- 发送邮件前确认；
- 退款前审批；
- 删除数据前审批；
- 高风险工具调用前审批；
- 低置信度答案人工审核。

✅ 示例代码

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ApprovalState(TypedDict):
    draft: str
    approved: bool
    result: str


def draft_node(state: ApprovalState):
    return {"draft": "您好，订单已退款，请查收。"}


def approval_node(state: ApprovalState):
    decision = interrupt({
        "message": "请审批是否发送这封邮件",
        "draft": state["draft"],
    })
    return {"approved": decision["approved"]}


def send_node(state: ApprovalState):
    if not state["approved"]:
        return {"result": "人工拒绝，未发送。"}
    return {"result": f"已发送：{state['draft']}"}


builder = StateGraph(ApprovalState)
builder.add_node("draft", draft_node)
builder.add_node("approval", approval_node)
builder.add_node("send", send_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "approval")
builder.add_edge("approval", "send")
builder.add_edge("send", END)

graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "approval-001"}}

first_result = graph.invoke(
    {"draft": "", "approved": False, "result": ""},
    config=config,
)

second_result = graph.invoke(
    Command(resume={"approved": True}),
    config=config,
)

print(second_result["result"])
```

📘 特点

- `interrupt()` 会暂停图；
- `Command(resume=...)` 用于恢复；
- 必须配合 checkpointer；
- 很适合生产中的审批流。

## 16. Streaming —— 流式观察执行过程

LangGraph 支持流式输出图执行过程。

✅ 示例代码

```python
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "解释 LangGraph"}]},
    stream_mode="updates",
):
    print(chunk)
```

常见模式：

| stream_mode | 说明 |
|-------------|------|
| `updates` | 每个节点输出的增量更新 |
| `values` | 每一步后的完整 State |
| `messages` | 模型消息或 token 流 |

📘 特点

- 适合调试；
- 适合前端展示执行进度；
- 可以显示 Agent 当前执行到哪个节点；
- 对复杂工作流很有用。

## 17. Time Travel —— 状态回放

Time Travel 是基于 checkpoint 的能力，可以查看历史状态，甚至从某个历史状态重新执行。

✅ 示例代码

```python
history = list(graph.get_state_history(config))

for item in history:
    print(item)
```

📘 特点

- 适合 debug；
- 可以回看每一步 state；
- 可以做审计；
- 可以从某个 checkpoint 重新试验不同分支；
- 注意外部副作用可能被重复执行，所以生产环境要做幂等控制。

## 18. Subgraph —— 子图复用

复杂系统可以拆成多个子图。

例如：

```text
主图：
用户输入 -> 判断任务类型 -> 调用 RAG 子图 -> 审核 -> 返回

RAG 子图：
检索 -> 判断资料是否足够 -> 改写问题 -> 生成答案
```

✅ 示例结构

```python
rag_builder = StateGraph(RAGState)
rag_builder.add_node("retrieve", retrieve_node)
rag_builder.add_node("generate", generate_node)
rag_builder.add_edge(START, "retrieve")
rag_builder.add_edge("retrieve", "generate")
rag_builder.add_edge("generate", END)

rag_graph = rag_builder.compile()
```

在主图中可以把子图当作一个节点使用。

📘 特点

- 适合复杂项目拆分；
- 每个子图可以单独测试；
- 多人协作更清晰；
- 不要让子图依赖过多全局变量。

## 19. Multi-Agent —— 多 Agent 编排

LangGraph 很适合做多 Agent 编排。

常见模式：

| 模式 | 说明 |
|------|------|
| Planner-Executor | 规划 Agent 拆任务，执行 Agent 完成 |
| Supervisor | 主管 Agent 决定调用哪个 Agent |
| Reviewer | 一个 Agent 生成，一个 Agent 审查 |
| Debate | 多个 Agent 给出不同意见 |
| Pipeline | 多个 Agent 固定顺序协作 |

示意流程：

```text
START
  -> planner
  -> researcher
  -> writer
  -> reviewer
  -> 如果不通过，回 writer
  -> END
```

✅ 伪代码

```python
def planner_node(state):
    return {"plan": ["检索资料", "写答案", "审查答案"]}


def researcher_node(state):
    return {"research": "检索到的资料"}


def writer_node(state):
    return {"draft": "根据资料写出的初稿"}


def reviewer_node(state):
    return {"approved": True}
```

📘 特点

- 多 Agent 不是越多越好；
- 每个 Agent 要有清晰职责；
- 最好有 Reviewer 或 Supervisor 控制质量；
- 多 Agent 成本和延迟通常更高。

## 20. LangGraph RAG 完整示例

下面是一个更完整的 RAG 图结构：

```text
START
  -> retrieve
  -> judge
  -> 如果资料足够：generate
  -> 如果资料不足：rewrite
  -> rewrite 后回 retrieve
  -> END
```

✅ 示例代码

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class GraphRAGState(TypedDict):
    question: str
    query: str
    docs: list[str]
    answer: str
    retry_count: int
    enough: bool


def retrieve_node(state: GraphRAGState):
    # 真实项目中这里换成 retriever.invoke(state["query"])
    docs = [
        f"检索到与 {state['query']} 相关的文档片段。"
    ]
    return {"docs": docs}


def judge_node(state: GraphRAGState):
    if state["retry_count"] >= 1:
        return {"enough": True}

    if "LangGraph" in "\n".join(state["docs"]):
        return {"enough": True}

    return {"enough": False}


def rewrite_node(state: GraphRAGState):
    return {
        "query": state["question"] + " LangGraph 详细说明",
        "retry_count": state["retry_count"] + 1,
    }


def generate_node(state: GraphRAGState):
    context = "\n".join(state["docs"])
    return {
        "answer": f"根据上下文回答：{context}"
    }


def route_node(state: GraphRAGState):
    if state["enough"]:
        return "generate"
    return "rewrite"


builder = StateGraph(GraphRAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("judge", judge_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "judge")
builder.add_conditional_edges(
    "judge",
    route_node,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)

graph = builder.compile()

result = graph.invoke({
    "question": "LangGraph 是什么？",
    "query": "LangGraph",
    "docs": [],
    "answer": "",
    "retry_count": 0,
    "enough": False,
})

print(result["answer"])
```

📘 特点

- 比普通 2-Step RAG 更灵活；
- 可以加入判断、改写、循环；
- 适合复杂问答；
- 必须控制循环次数。

## 21. 实战建议

| 场景 | 推荐做法 |
|------|----------|
| 简单问答 | 不需要 LangGraph，直接模型调用 |
| 摘要 / 翻译 / 分类 | LCEL |
| 普通 RAG | LCEL + Retriever |
| 需要改写和多次检索的 RAG | LangGraph |
| 简单工具调用 | `create_agent` |
| 多工具、多步骤、可恢复 Agent | LangGraph |
| 高风险工具执行 | LangGraph + human-in-the-loop |
| 多 Agent 协作 | LangGraph |

## 22. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| State 没更新 | 节点没有返回对应字段 | 检查 return dict |
| 列表被覆盖 | 没有 reducer | 使用 `Annotated[list, operator.add]` |
| 条件边不跳转 | route 返回值和 mapping 不一致 | 检查 key |
| 图无限循环 | 没有退出条件 | 加 `retry_count` |
| 人审无法恢复 | 没有 checkpointer | compile 时传 checkpointer |
| 多轮记忆失效 | 没传 thread_id | config 里传 `thread_id` |
| 工具重复执行 | replay 或重试导致副作用重复 | 工具要幂等 |

## 23. 生产环境注意点

- State 不要过大；
- 节点职责要单一；
- 所有循环必须有退出条件；
- 高风险工具必须人工审批；
- 外部副作用要幂等；
- 每个 thread 要有稳定 `thread_id`；
- 接入 LangSmith 观察每一步执行；
- 工具调用要做权限校验；
- RAG 要做 metadata 权限过滤；
- 失败样本要进入评估集。

## 24. 一句话总结

LangGraph 就是把复杂 Agent 拆成一个可控的状态图：

```text
State 保存状态；
Node 执行步骤；
Edge 控制流转；
Conditional Edge 决定分支；
Checkpointer 支持恢复；
interrupt 支持人工介入。
```

当你的 LLM 应用开始出现 **多步骤、分支、循环、工具调用、人审、恢复、多 Agent** 这些需求时，就应该使用 LangGraph。

