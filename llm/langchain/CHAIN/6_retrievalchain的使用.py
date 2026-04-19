#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/4 22:07
@source from: 
"""
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import faiss

# ✅ 1. 构造文档（模拟语料库）
raw_text = """
LangChain 是一个用于开发由大语言模型驱动的应用的框架。它为构建问答系统、聊天机器人、智能搜索引擎等提供模块化组件。
它集成了 LLM、Memory、Prompt、Agent、VectorStore 等模块，可以让你高效构建复杂的 AI 应用。
"""

# ✅ 2. 文本切割成块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.create_documents([raw_text])

# ✅ 3. 向量化
embedding_model = OpenAIEmbeddings(openai_api_key="你的 key")

# ✅ 4. 创建 FAISS 向量数据库
vectorstore = FAISS.from_documents(docs, embedding_model)

# ✅ 5. 构造 Retriever
retriever = vectorstore.as_retriever()

# ✅ 6. 构建 LLM（也可以换成 Qwen）
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

# ✅ 7. 构建 RetrievalQA 链
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ✅ 8. 测试问答
query = "LangChain 能用来做什么？"
res = rag_chain.invoke({"query": query})

print("🧠 答案：", res["result"])
print("📚 来源：", res["source_documents"][0].page_content)