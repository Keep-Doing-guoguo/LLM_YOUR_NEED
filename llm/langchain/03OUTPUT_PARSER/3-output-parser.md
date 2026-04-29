# OutputParser 与结构化输出

LLM 默认返回自然语言，但业务系统通常需要稳定的数据结构。OutputParser 和 Structured Output 的目标就是让模型输出更可控、更容易接入程序。

## 1. 为什么需要 OutputParser

自然语言输出的问题：

- 格式不稳定
- 字段可能缺失
- JSON 可能不合法
- 难以直接写入数据库
- 难以被前端稳定渲染

所以很多场景需要把模型输出转成：

```text
字符串
JSON
Pydantic 对象
列表
分类标签
表格数据
```

## 2. StrOutputParser

最常用的解析器，把模型消息转成字符串。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("用一句话解释：{topic}")
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "LangChain"})
print(result)
```

执行流程：

```text
dict input -> prompt -> model -> AIMessage -> string
```

## 3. JSON 输出

可以通过 Prompt 要求模型输出 JSON。

```python
prompt = ChatPromptTemplate.from_template("""
请从文本中抽取信息，并输出 JSON。

文本：{text}

JSON 字段：
- name
- category
- summary
""")
```

但只靠 Prompt 有风险，模型可能输出解释文字或非法 JSON。

## 4. Pydantic 结构化输出

更推荐用 schema 描述目标结构。

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ProductInfo(BaseModel):
    name: str = Field(description="产品名称")
    category: str = Field(description="产品分类")
    summary: str = Field(description="一句话总结")

agent = create_agent(
    model=model,
    tools=[],
    response_format=ProductInfo,
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "LangChain 是一个构建 LLM 应用的框架。"}
    ]
})

print(result["structured_response"])
```

## 5. Structured Output 的优势

| 优势 | 说明 |
|------|------|
| 类型明确 | 字段和类型提前定义 |
| 易校验 | Pydantic 自动校验 |
| 易接入业务 | 可以直接传给 API、数据库、前端 |
| 更稳定 | 比纯 Prompt JSON 更可靠 |

## 6. Provider Strategy 和 Tool Strategy

LangChain 会根据模型能力选择结构化输出策略：

| 策略 | 说明 |
|------|------|
| Provider Strategy | 模型供应商原生支持结构化输出 |
| Tool Strategy | 通过工具调用的参数承载结构化数据 |

一般使用时可以直接传 schema：

```python
response_format=ProductInfo
```

LangChain 会自动选择合适策略。

## 7. 结构化输出适合什么

| 场景 | 示例 |
|------|------|
| 信息抽取 | 从合同中抽取甲方、乙方、金额 |
| 分类 | 判断工单类型、情绪倾向 |
| 前端渲染 | 返回卡片、表格、步骤列表 |
| API 接口 | 返回固定字段 |
| 评估 | 返回 score、reason、label |

## 8. 常见失败

| 问题 | 处理方式 |
|------|----------|
| 字段缺失 | 使用必填字段和描述 |
| 类型错误 | Pydantic 校验，失败重试 |
| 输出多个对象 | 明确 schema 是单对象还是列表 |
| 模型不支持 | 换支持结构化输出的模型或用 Tool Strategy |

## 9. 实战建议

- 简单文本生成：`StrOutputParser`
- 业务字段抽取：Pydantic structured output
- 需要前端渲染：结构化输出
- 强可靠系统：结构化输出 + 业务校验 + 失败重试

## 10. 小结

OutputParser 解决“模型输出怎么变成程序可用数据”的问题。生产系统里，不要只依赖自然语言输出，关键链路应使用结构化输出。

