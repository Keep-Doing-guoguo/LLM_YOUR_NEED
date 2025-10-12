#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/13 16:22
@source from: 
"""

from langchain_community.chat_models import ChatOpenAI
model = ChatOpenAI(
    temperature=0,
    openai_api_key="你的key",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name="qwen-plus",
)
print('abc')