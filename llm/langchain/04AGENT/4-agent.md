

## 1.🧰 常见 AgentType 列表（部分）

LangChain 中比较常用的几种 AgentType：

| AgentType | 特点 / 说明 |
|------------|-------------|
| **ZERO_SHOT_REACT_DESCRIPTION** | 零样本 ReAct 模式。模型根据工具说明（`description`）判断使用哪个工具，适合单轮任务。 |
| **CONVERSATIONAL_REACT_DESCRIPTION** | 支持对话历史 + ReAct，适合带上下文、多轮交互的场景。 |
| **ZERO_SHOT_REACT** | 类似于 `ZERO_SHOT_REACT_DESCRIPTION`，但依赖不同的 Prompt 模板。 |
| **STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION** | 强结构化输出 + ReAct。适合需要严格输出格式的任务。 |
| **CHAT_CONVERSATIONAL_REACT_DESCRIPTION** | 在聊天机制上做更多处理，适合更复杂的对话型 Agent。 |
| **PLAN_AND_EXECUTE_AGENT** | “先规划再执行”的复杂任务拆分模式。先生成计划（多个步骤），再逐步执行。 |




## 2. 入门介绍 & 区别说明

### 2.1.ZERO_SHOT_REACT_DESCRIPTION

	•	“Zero-shot” 表示模型 无需示例 就能用工具。
	•	“REACT” 模式就是经典的 ReAct 思考 + 行动循环：Thought → Action → Observation。
	•	“DESCRIPTION” 表示模型依据 Tool 的 description 选择工具。因为你给定工具时写了描述，模型读这个描述知道“Tool 是干啥的”。

适用场景：

单轮任务（一个问题对应一个回答），任务工具比较清晰。



### 2.2.CONVERSATIONAL_REACT_DESCRIPTION

	•	在 ZERO_SHOT 的基础上支持 对话历史。模型可以参考之前的上下文、工具调用历史，再决定下一步。
	•	适合聊天助手、问答机器人、带记忆场景。

举例：

用户问：

“刚才帮我查的那个短信是什么内容？”

模型可以回看之前的检索、工具调用、历史结果，再输出。



结构化 / 强格式输出 AgentType

这些 AgentType（如 STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION）通常会：

	•	限定输出 JSON / 指定字段格式，不让模型自由发挥。
	•	适合你希望工具调用 + 输出结果有严格格式的场景（比如 API 接口返回结构体）。



### 2.3.Plan & Execute 类型 Agent

有些任务复杂，一个问题可能分解成多个小子任务。PLAN_AND_EXECUTE_AGENT 类型的 Agent 会先让模型给出任务拆分规划（plan），然后按计划顺序执行（execute）。

举例：

“帮我生成一篇文章 + 配图 + 写摘要 + 发邮件给老板”

Agent 可能先产出：

	1.	写文章
	2.	生成图片
	3.	写摘要
	4.	构造邮件内容
	5.	发送邮件

然后一步一步执行这些小任务。

| 场景 | 推荐 AgentType |
|------|----------------|
| **简单问答或工具调用** | `ZERO_SHOT_REACT_DESCRIPTION` |
| **聊天机器人 / 多轮有上下文** | `CONVERSATIONAL_REACT_DESCRIPTION` |
| **需要严格格式输出** | `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION` |
| **复杂任务拆解 + 子任务执行** | `PLAN_AND_EXECUTE_AGENT` |

## 3.🛠 使用建议



在构建 Agent 时，需要考虑：

	•	是否有对话上下文？ → 要不就用 CONVERSATIONAL 系列
	•	是否需要拆任务？ → 要就用 Plan & Execute 类型
	•	工具是否多？ → 多工具场景比较适合 ReAct 模式
	•	输出格式是否严格？ → 用结构化 AgentType 强控制



版本更新之后，这些旧的 `AgentType` 写法已经逐步被新的 Agent API 替换。

随着 LangChain 版本迭代，部分 `initialize_agent + AgentType` 写法已经不再是首选。新代码建议优先使用 `create_agent`；如果任务进一步复杂到需要稳定状态、循环、人工审批、持久化恢复，再使用 LangGraph。


## 4. Agent 的一个完整案例代码

下面实现一个工具型 Agent，包含三个工具：

| 工具 | 作用 |
|------|------|
| `calculator` | 安全计算数学表达式 |
| `get_weather` | 查询模拟天气 |
| `query_orders` | 查询本地 SQLite 订单表，只允许 `SELECT` |

对应的可运行代码已放在同目录：

```bash
llm/langchain/04AGENT/agent_demo.py
```

运行方式：

```bash
cd llm/langchain/04AGENT
export DASHSCOPE_API_KEY="你的 DashScope API Key"
python agent_demo.py
```

如果要切换模型：

```bash
export AGENT_MODEL="qwen-plus"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
python agent_demo.py
```

核心代码如下：

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入必须是纯数学表达式，例如 3 * (5 + 7)。"""
    ...


@tool
def get_weather(city: str) -> str:
    """查询城市天气。输入城市名称，返回天气摘要。"""
    ...


@tool
def query_orders(sql: str) -> str:
    """查询订单 SQLite 表。只允许 SELECT 语句，例如 SELECT * FROM orders LIMIT 3。"""
    ...


llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    api_key="你的 DashScope API Key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

agent = create_agent(
    model=llm,
    tools=[calculator, get_weather, query_orders],
    system_prompt=(
        "你是一个工具型助手。需要计算、查天气或查订单时，优先调用合适的工具；"
        "回答要简洁，并说明你使用了哪个工具。"
    ),
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "3 * (5 + 7) 等于多少？"}
    ]
})

print(result["messages"][-1].content)
```

完整实现见：

```python
agent_demo.py
```

## 5. 这个案例说明了什么

Agent 的关键不是“调用一次模型”，而是让模型根据用户问题选择工具：

```text
用户问题
  -> 模型判断是否需要工具
  -> 选择 calculator / get_weather / query_orders
  -> 执行工具
  -> 读取 Observation
  -> 组织最终回答
```

这个案例里：

- 问数学题时，Agent 应该调用 `calculator`
- 问天气时，Agent 应该调用 `get_weather`
- 问订单时，Agent 应该调用 `query_orders`

## 6. 为什么不用旧的 AgentType 写法

旧版本常见写法是：

```python
initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
```

这种写法在很多老教程里还能看到，但新项目更建议使用：

```python
create_agent(
    model=llm,
    tools=tools,
    system_prompt="..."
)
```

原因是：

- 新 API 更贴近当前 LangChain 文档
- Agent 底层基于 LangGraph，后续更容易扩展状态、记忆、人工介入
- 工具、结构化输出、middleware、runtime context 等能力更容易组合

## 7. 生产环境注意点

上面的示例是教学代码。真实业务里要额外处理：

- 不要用 `eval` 直接执行用户输入，所以示例里用 AST 做了安全计算
- SQL 工具必须限制权限，最好只允许白名单查询
- 工具返回内容不能太长，否则会挤占上下文
- 工具调用要设置超时、重试和错误返回
- 涉及真实用户数据时要做权限校验和审计
- Agent 要设置最大调用轮数，避免循环调用工具
