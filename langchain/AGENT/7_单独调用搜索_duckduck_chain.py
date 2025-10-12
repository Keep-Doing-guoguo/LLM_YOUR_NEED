#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/6 17:45
@source from: 
"""
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
#from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.docstore.document import Document

import time
from typing import List
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

def duckduckgo_search1(
    text: str,
    result_len: int = 5,
    backend: str = "lite",       # 关键：用 lite，抗 202 Ratelimit
    retries: int = 3,
    backoff: float = 1.5,
) -> List[str]:
    """
    返回网页结果的简短摘要列表；内部带重试与后端切换。
    """
    last_err = None
    for attempt in range(retries):
        try:
            search = DuckDuckGoSearchAPIWrapper(
                backend=backend,     # 可选: "lite" / "html" / "api" / "auto"
                region="us-en",      # 按需要换地区
                safesearch="moderate",
                time="y",            # 近一年，可选 "d","w","m","y"
                max_results=result_len,
            )
            return search.results(text, result_len)
        except Exception as e:
            last_err = e
            # 第一次失败：换后端再试
            if attempt == 0 and backend != "html":
                backend = "html"
            # 第二次失败：再换
            elif attempt == 1 and backend != "api":
                backend = "api"
            time.sleep(backoff ** attempt)

    # 全部失败，给个兜底
    raise RuntimeError(f"DuckDuckGo search failed after retries: {last_err}")
# Step 1: 搜索工具
def duckduckgo_search(text, result_len=5):

    return duckduckgo_search1(text, result_len=5)


# Step 2: 转换为文档对象
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


# Step 3: 主函数
def main():
    query = "今天星期几"

    # 搜索
    results = duckduckgo_search(query)
    docs = search_result2docs(results)
    context = "\n".join([doc.page_content for doc in docs]) or "无可用上下文"

    # Prompt 模板
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
    source_documents = [
        f"""出处 [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
        for inum, doc in enumerate(docs)
    ]
    if len(source_documents) == 0:  # 没有找到相关资料（不太可能）
        source_documents.append(f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")
    # 模型
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key="你的key",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
    )

    chain = LLMChain(prompt=prompt_template, llm=llm)

    print("\n🤖 正在生成回答...\n")
    answer = chain.run({"context": context, "question": query})
    print("🧠 答案：", answer)
    print("======context=======")
    print(context)
    print("======context=======")


if __name__ == "__main__":
    main()