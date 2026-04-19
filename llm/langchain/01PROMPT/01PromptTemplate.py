#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/12/3 14:44
@source from: 
"""
# 1️⃣ PromptTemplate：最基础的文本模板
#
# 场景：单轮任务 / 传统 completion 模型 / 直接拼字符串
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "请用一句话总结以下内容：{text}"
)


# 1）只用来格式化字符串
user_text = "这是一个关于 YOLOv8 火焰检测项目的 README，包含训练配置和性能评估。"
final_prompt_str = prompt.format(text=user_text)

print(final_prompt_str)
# 输出：请用一句话总结以下内容：这是一个关于 YOLOv8 火焰检测项目的 README，包含训练配置和性能评估。



#✅ 技巧：多个变量 + 默认参数封装
from langchain.prompts import PromptTemplate

qa_prompt = PromptTemplate.from_template(
    "请用{style}的语气回答下面的问题：\n"
    "问题：{question}\n"
    "上下文：{context}\n"
)

def build_prompt(question, context, style="严谨专业"):
    return qa_prompt.format(question=question, context=context, style=style)

print(build_prompt("什么是 Milvus？", "Milvus 是一个向量数据库..."))

'''
输出：

请用严谨专业的语气回答下面的问题：
问题：什么是 Milvus？
上下文：Milvus 是一个向量数据库...

'''
