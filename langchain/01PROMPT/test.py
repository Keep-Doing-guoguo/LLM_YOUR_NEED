from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {
        "question": "什么是 YOLO？",
        "answer": "YOLO 是一种实时目标检测算法，全称是 You Only Look Once。"
    },
    {
        "question": "什么是向量数据库？",
        "answer": "向量数据库用于存储和检索向量数据，常用于相似度搜索和 RAG。"
    },
]

# 每个 few-shot 示例是一段 “人问 + AI 答”
example_chat_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("问：{question}"),
    AIMessagePromptTemplate.from_template("答：{answer}"),
])

fewshot_chat_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_chat_prompt,
    # 前缀：提醒模型接下来是示例
    prefix=ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("下面是一些问答示例：")
    ]),
    # 后缀：真正要问的问题
    suffix=ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("现在请回答我的问题：{question}")
    ]),
)

# 先把 few-shot 模板展开成消息列表，再丢给 llm
msgs = fewshot_chat_prompt.format_messages(
    question="Milvus 在实际业务中有哪些典型应用场景？"
)
resp_msg = llm.invoke(msgs)

print("【FewShotChatMessagePromptTemplate 业务问答】")
print(resp_msg.content)