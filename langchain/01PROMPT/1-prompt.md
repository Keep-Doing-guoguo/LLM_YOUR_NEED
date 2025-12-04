
## 1️⃣ PromptTemplate

👉 最基础的模板类（纯文本Prompt）

这是最早的、通用的模板类型，用于非聊天模型（比如 text-davinci-003 或 llama 类模型）。
```
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "请用一句话总结以下内容：{text}"
)
```
🔹 适用场景：

	•	单轮任务

	•	没有角色（system/user/assistant）概念

	•	传统“问→答”任务

🔹 输出内容：

"请用一句话总结以下内容：Python 是一种解释型语言。"




## 2️⃣ ChatPromptTemplate

👉 专门为聊天模型（如 ChatGPT、Qwen、Claude）设计的多角色模板。

用于组合多条消息（SystemMessage, HumanMessage, AIMessage），
支持上下文、系统提示、用户消息混合。
```
from langchain.prompts.chat import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template("""
你是一个智能助手，请根据以下内容回答问题：
{context}
问题：{question}
""")
```
或者多条消息版本：
```
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个友好的知识问答助手。"),
    HumanMessagePromptTemplate.from_template("请回答：{question}")
])
```
🔹 适用场景：

	•	聊天模型 (ChatOpenAI, ChatQwen, ChatGLM, 等)
	•	支持多轮对话和系统指令
	•	最常用在 LLMChain 和 ConversationalRetrievalChain 中

⸻

## 3️⃣ ChatMessagePromptTemplate

👉 是更底层的消息级模板，用于定义某个角色的一条消息。

举个例子：
```
from langchain.prompts.chat import ChatMessagePromptTemplate

msg = ChatMessagePromptTemplate.from_template(
    role="user",
    template="你好，我想了解一下 {topic}"
)
```
可以动态生成一个用户消息：
```
msg.format(topic="中国移动")
# => HumanMessage(content="你好，我想了解一下 中国移动")
```
🔹 适用场景：

	•	当你要自定义不同角色（user/system/assistant/其他自定义角色）；
	•	一般被 ChatPromptTemplate 包含；
	•	灵活控制一条消息的模板。



##  4️⃣ 其他衍生模板



| 类名 | 说明 | 主要用途 |
|------|------|----------|
| **SystemMessagePromptTemplate** | 系统指令模板 | 用于设定角色身份、风格、任务约束 |
| **HumanMessagePromptTemplate** | 用户消息模板 | 用户提问部分 |
| **AIMessagePromptTemplate** | AI 回复模板 | 模拟模型上一轮的回答，用于历史对话 |
| **FewShotPromptTemplate** | 少样本模板 | 在 prompt 里插入示例样本 |
| **FewShotChatMessagePromptTemplate** | 聊天版本的 FewShot | 在 Chat 模型中做 few-shot 例子提示 |


### 关系总结图
```
PromptTemplate
   └── ChatPromptTemplate
          ├── SystemMessagePromptTemplate
          ├── HumanMessagePromptTemplate
          ├── AIMessagePromptTemplate
          └── ChatMessagePromptTemplate (通用角色版本)
```



### 实战建议

| 场景 | 推荐使用 |
|------|------------|
| 单轮文本问答任务 | **PromptTemplate** |
| 聊天模型（Qwen / ChatGPT） | ✅ **ChatPromptTemplate** |
| 多角色 prompt（system / user / assistant） | **ChatPromptTemplate + 各种 MessagePromptTemplate** |
| 自定义 agent 对话历史 | **ChatMessagePromptTemplate** |
| few-shot 提示（带例子） | **FewShotPromptTemplate / FewShotChatMessagePromptTemplate** |




### 一句话总结：

PromptTemplate 是基础文本模板，

ChatPromptTemplate 是聊天模型专用模板，

ChatMessagePromptTemplate 是其中一条消息的模板。

它们的关系就像：

ChatPromptTemplate = 多条 ChatMessagePromptTemplate 的组合

### 案例：

```python
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# 1️⃣ 定义系统角色（设定身份和规则）
system_template = "你是一个专业的电信客服助手，回答要简洁、准确，只回答与中国移动相关的问题。"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# 2️⃣ 模拟上一轮对话：用户问过“怎么查话费”，AI 已经回答过
human_history = HumanMessagePromptTemplate.from_template("怎么查询我的话费余额？")
ai_history = AIMessagePromptTemplate.from_template("您可以发送短信'CXX'到10086，或登录中国移动APP查看。")

# 3️⃣ 当前用户的新问题（带变量）
current_human = HumanMessagePromptTemplate.from_template("那{service}怎么开通？")

# 4️⃣ 组合成完整的聊天模板（包含历史 + 当前问题）
chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_history,
    ai_history,
    current_human
])

# 5️⃣ 初始化聊天模型（使用 gpt-3.5-turbo）
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 6️⃣ 构建 LLMChain
chain = LLMChain(llm=chat_model, prompt=chat_prompt)

# 7️⃣ 调用链，传入变量
response = chain.run(service="5G套餐")

# 8️⃣ 输出结果
print("🤖 模型回复：")
print(response)
```