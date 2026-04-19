#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/6 18:01
@source from: 
"""
from langchain.tools import Tool
from calculate import calculate
from DataSource import fuge_data_source
from sql_tool import run_sqlite
from milvus_tool import milvus_search

## 请注意，如果你是为了使用AgentLM，在这里，你应该使用英文版本。

tools = [
    Tool(
        func=calculate,
        name="calculate",
        description="Useful for when you need to answer questions about simple calculations",
        #args_schema=CalculatorInput,
    ),
    # Tool(
    #     func=search_internet,
    #     name="search_internet",
    #     description="Use this tool to use bing search engine to search the internet",
    #     #args_schema=SearchInternetInput,
    # ),
    Tool(
        name="查询产品名称",
        func=fuge_data_source.find_product_description,
        description="通过产品名称找到产品描述时用的工具，输入应该是产品名称",
    ),
    Tool(
        name="复歌科技公司相关信息",
        func=fuge_data_source.find_company_info,
        description="当用户询问公司相关的问题，可以通过这个工具了解相关信息",
    ),
    Tool(
        name="SQLiteQuery",
        func=run_sqlite,
        description=(
            "用于执行SQLite数据库中的SQL查询，返回JSON结果。"
            "输入必须是完整的SELECT语句，例如：SELECT * FROM users LIMIT 3;"
        )
    ),
    Tool(
        name="MilvusSearch",
        func=milvus_search,
        description=(
            "用于语义相似检索（短信/文本语料）。输入自然语言查询，"
            "自动向量化并在 Milvus 中召回最相似的文本，返回 JSON（text, prob, similarity）。"
        )
    )

]

tool_names = [tool.name for tool in tools]
