#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/4 13:32
@source from: 
"""
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# 1️⃣ 定义 Prompt 模板（包含历史对话）
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
你是一个智能助手，以下是你和用户的对话历史：
{history}
用户: {input}
助手:"""
)
import os
os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 2️⃣ 配置 memory
memory = ConversationBufferMemory(memory_key="history", return_messages=False)

# 3️⃣ 实例化 LLM（可以换成 Qwen、ChatGLM）
llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        model_name="qwen-plus",
    )
# 4️⃣ 构建 LLMChain
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# 5️⃣ 调用 chain.call() 进行对话

res1 = chain.run("你好")
res2 = chain.run("你还记得我刚刚说了什么吗？")

print(res1)
print(res2)
# res1 = chain.call({"input": "你好"})
# print(res1["text"])
#
# res2 = chain.call({"input": "你还记得我刚刚说了什么吗？"})
# print(res2["text"])


