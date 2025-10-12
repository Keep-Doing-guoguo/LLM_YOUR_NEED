#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/4 22:23
@source from: 
"""
# ✅ 通用RAG框架：支持 FAISS / Chroma / Weaviate / Milvus / Pinecone

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma

# 可选：你可以自行替换为 Pinecone、Weaviate、Milvus 导入
# from langchain.vectorstores import Weaviate, Milvus, Pinecone

# ✅ 向量化模型（通用嵌入器）
embedding_model = HuggingFaceEmbeddings()

# ✅ 文本切分器
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def load_and_split(file_path: str) -> List[Document]:
    # 从文件中读取内容（简写，替换为实际加载）
    with open(file_path, encoding="utf-8") as f:
        raw_text = f.read()
    docs = splitter.create_documents([raw_text])
    return docs


# ✅ 向量库选择器（根据 name 决定存储方式）
def build_vectorstore(name: str, docs: List[Document]):
    if name == "faiss":
        return FAISS.from_documents(docs, embedding_model)
    elif name == "chroma":
        return Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db")
    # elif name == "weaviate":
    #     return Weaviate.from_documents(...)
    # elif name == "milvus_l":
    #     return Milvus.from_documents(...)
    # elif name == "pinecone":
    #     return Pinecone.from_documents(...)
    else:
        raise ValueError("Unsupported vectorstore")


# ✅ 检索 + 问答（使用 LangChain Retriever + LLM）
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain


# ✅ 演示流程
def rag_pipeline(file_path: str, store_name: str):
    docs = load_and_split(file_path)
    vec_store = build_vectorstore(store_name, docs)
    qa_chain = build_qa_chain(vec_store)

    while True:
        query = input("💬 请输入你的问题：")
        result = qa_chain.run(query)
        print("🤖 回答：", result)


# ✅ 使用示例（传入txt文件路径 + 向量库名）
if __name__ == "__main__":
    rag_pipeline("./test.txt", store_name="faiss")  # 可选：chroma / faiss / milvus_l / pinecone
