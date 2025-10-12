#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/13 14:17
@source from: 
"""
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI  # 也可用 HuggingFace

# 初始化 LLM
llm = ChatOpenAI(
        openai_api_key="sk-",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
)
print('debug')
# 初始化对话记忆
memory = ConversationBufferMemory(k=10)

# 创建带记忆的对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print("🤖 多轮对话机器人启动了！输入 exit 退出")

# 控制台循环输入
while True:
    user_input = input("👤 你说：")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 再见！")
        break

    response = conversation.predict(input=user_input)
    print("🤖 回复：" + response)