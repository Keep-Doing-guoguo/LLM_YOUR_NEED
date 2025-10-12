#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/4 19:59
@source from: 
"""
from langchain.utilities import DuckDuckGoSearchAPIWrapper

# 创建一个 DuckDuckGo 搜索包装器实例
duckduckgo_search = DuckDuckGoSearchAPIWrapper()

# 进行搜索
results = duckduckgo_search.run("朱元璋")

# 输出搜索结果
# for result in results:
#     print(result["title"], result["link"])