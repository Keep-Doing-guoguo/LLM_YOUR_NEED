#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/6 17:45
@desc: 使用 DuckDuckGo 搜索 + LLMChain 实现问答
"""

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.docstore.document import Document
from pydantic import BaseModel, Field


# Step 1: 搜索工具
def duckduckgo_search(text, result_len=5):
    search = DuckDuckGoSearchAPIWrapper()
    return search.results(text, result_len)


# Step 2: 转换为 Document 对象
def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(
            page_content=result.get("snippet", ""),
            metadata={
                "source": result.get("link", ""),
                "filename": result.get("title", "")
            }
        )
        docs.append(doc)
    return docs


# Step 3: 主调用函数（控制台执行）
def search_internet(query: str) -> str:
    results = duckduckgo_search(query)
    docs = search_result2docs(results)
    context = "\n".join([doc.page_content for doc in docs]) or "无可用上下文"

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
<指令>这是我搜索到的互联网信息，请你根据这些信息进行提取并有调理、简洁地回答问题。
如果无法从中得到答案，请说 “无法搜索到能回答问题的内容”。</指令>

<已知信息>
{context}
</已知信息>

<问题>
{question}
</问题>
"""
    )

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key="你的key",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
    )

    chain = LLMChain(prompt=prompt_template, llm=llm)
    answer = chain.run({"context": context, "question": query})
    return answer


# ✅ 定义 pydantic 输入模型（可选用于 agent）
class SearchInternetInput(BaseModel):
    location: str = Field(description="Query for Internet search")


# Step 4: 主入口
if __name__ == "__main__":
    query = "今天星期几"
    result = search_internet(query)
    print("🧠 答案:", result)