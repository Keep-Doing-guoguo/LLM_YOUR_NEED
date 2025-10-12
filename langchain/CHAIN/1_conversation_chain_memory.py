#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/4 12:47
@source from: 
"""
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI, Anthropic

# 阿里云通义千问 API，文档参考 https://help.aliyun.com/zh/dashscope/developer-reference/api-details
#     "qwen-api": {
#         "version": "qwen-turbo",  # 可选包括 "qwen-turbo", "qwen-plus"
#         "api_key": "sk-",  # 请在阿里云控制台模型服务灵积API-KEY管理页面创建
#         "provider": "QwenWorker",
#         "embed_model": "text-embedding-v1" # embedding 模型名称
#     },
llm = ChatOpenAI(
        api_key="sk-",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        stream=False
)
from langchain.chains import LLMChain
print('debug')

model1 = OpenAI(
        openai_api_key="sk-",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
    )
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="请用简洁的语言解释 {topic} 是什么。"
)

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run("区块链")
print(response)
print('debug')


#✅ 基本用法
from langchain.prompts import PromptTemplate

# 定义一个带变量的 prompt 模板
template = PromptTemplate(
    input_variables=["product"],
    template="请帮我写一段关于 {product} 的广告文案。"
)

# 填充变量，得到完整的 prompt
prompt_text = template.format(product="智能手表")
print(prompt_text)

#✅ 多个变量
template = PromptTemplate(
    input_variables=["language", "topic"],
    template="用 {language} 解释一下 {topic} 是什么。"
)

print(template.format(language="中文", topic="量子力学"))
#✅ 配合 LLM 使用
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


prompt = PromptTemplate(
    input_variables=["topic"],
    template="请用简洁的语言解释 {topic} 是什么。"
)

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run("区块链")
print(response)


#✅ 示例：记住上下文的对话机器人
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# 1️⃣ 定义 Prompt 模板（带历史记录变量）
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
你是一个智能助手，以下是你和用户的对话历史：
{history}
用户: {input}
助手:"""
)

# 2️⃣ 定义记忆（会自动记录历史对话）
memory = ConversationBufferMemory(memory_key="history", return_messages=False)

# 3️⃣ 实例化 LLM（可以换成 Qwen、ChatGLM 等）
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# 4️⃣ 创建对话链条
conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# 5️⃣ 启动对话
print(conversation.predict(input="你好"))
print(conversation.predict(input="你还记得我刚刚说了什么吗？"))

