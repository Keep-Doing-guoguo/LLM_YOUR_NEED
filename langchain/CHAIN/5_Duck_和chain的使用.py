#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/4 21:07
@source from: 
"""
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# 1. DuckDuckGo Search Wrapper
search = DuckDuckGoSearchAPIWrapper()


# 2. Search Function (gets top K documents for a query)
def search_query(query: str, num_results: int = 5):
    search_results = search.run(query)
    search_results = search.results(query, num_results)# Run the search query
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs



# 3. Define PromptTemplate for the Chat
prompt_template = """
你是一个智能助手，以下是你从网上找到的相关信息：
{docs}
用户的问题是：{query}
请根据这些信息给出准确的回答：
"""

# 4. Set up ChatOpenAI model (You can replace this with your own model like qwen or ChatGLM)
llm = ChatOpenAI(
        openai_api_key="sk-",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        temperature=0.7,
)

# 5. Create the PromptTemplate and LLMChain
chat_prompt = ChatPromptTemplate.from_template(prompt_template)
chain = LLMChain(llm=llm, prompt=chat_prompt)


# 6. Final function for querying, searching, and generating response
def answer_query(query: str):
    # Step 1: Get documents from DuckDuckGo based on the query
    docs = search_query(query)

    # Step 2: Prepare the documents to be inserted into the prompt
    doc_contents = "\n\n".join([doc.page_content for doc in docs])
    #context = "\n".join([doc.page_content for doc in docs])#在这里获取到查询的结果了。
    # Step 3: Get the response from the model using LLMChain
    response = chain.run(query=query, docs=doc_contents)
    # response_1 = chain.acall(query=query, docs=doc_contents)
    # response_2 = chain.acall({"context": doc_contents, "question": query})
    print('debug')
    return response


# Test the application with a query
query = "What is the capital of France?"
answer = answer_query(query)
print("Answer:", answer)