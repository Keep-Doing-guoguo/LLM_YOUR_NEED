#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/14 08:54
@source from: 
"""
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import requests
from datetime import datetime
import re
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxx"

# 1️⃣ 定义 Prompt 模板（包含历史对话）

_PROMPT_TEMPLATE = """
用户会提出一个关于天气的问题，你的目标是拆分出用户问题中的区，市 并按照我提供的工具回答。
例如 用户提出的问题是: 上海浦东未来1小时天气情况？
则 提取的市和区是: 上海 浦东
如果用户提出的问题是: 上海未来1小时天气情况？
则 提取的市和区是: 上海 None
请注意以下内容:
1. 如果你没有找到区的内容,则一定要使用 None 替代，否则程序无法运行
2. 如果用户没有指定市 则直接返回缺少信息

问题: ${{用户的问题}}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。
```text

${{拆分的市和区，中间用空格隔开}}
```
... weathercheck(市 区)...
```output

${{提取后的答案}}
```
答案: ${{答案}}



这是一个例子：
问题: 上海浦东未来1小时天气情况？


```text
上海 浦东
```
...weathercheck(上海 浦东)...

```output
预报时间: 1小时后
具体时间: 今天 18:00
温度: 24°C
天气: 多云
风向: 西南风
风速: 7级
湿度: 88%
降水概率: 16%

Answer: 上海浦东一小时后的天气是多云。

现在，这是我的问题：

问题: {question}
"""
PROMPT = PromptTemplate(input_variables=["question"], template=_PROMPT_TEMPLATE)
# 3️⃣ 实例化 LLM（可以换成 Qwen、ChatGLM）
llm = ChatOpenAI(
        api_key="你的key",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
    )
# 4️⃣ 构建 LLMChain
chain = LLMChain(
    llm=llm,
    prompt=PROMPT,
    verbose=True
)
output = chain.run(question='北京的天气如何？')
match = re.search(r"```text(.*?)```", output, re.DOTALL)


print(output)