

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




## 2.📖 入门介绍 & 区别说明

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



版本更新之后，这些已经被替换了。

随着 LangChain 版本迭代，部分 AgentType 已被弃用或替换。建议优先使用最新推荐的 AgentType，如 `CONVERSATIONAL_REACT_DESCRIPTION` 和 `PLAN_AND_EXECUTE_AGENT`。


agent的一个案例代码
