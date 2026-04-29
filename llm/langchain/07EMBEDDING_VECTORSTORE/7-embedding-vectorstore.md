# Embedding、VectorStore 与 Retriever

Embedding 和 VectorStore 是 RAG 的基础。Embedding 把文本转成向量，VectorStore 负责存储和相似度检索，Retriever 提供统一的检索接口。

## 1. Embedding 是什么

Embedding 是一组数字向量，用来表示文本语义。

```text
"我要查订单" -> [0.12, -0.03, 0.88, ...]
```

语义相近的文本，向量距离更近。

## 2. Embedding 的用途

| 场景 | 说明 |
|------|------|
| 语义搜索 | 按含义找相关文本 |
| RAG | 检索知识片段交给模型回答 |
| 推荐系统 | 用户和物品向量匹配 |
| 聚类去重 | 找相似文本 |
| 长期记忆 | 保存用户偏好和历史事实 |

## 3. 初始化 Embedding 模型

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector = embeddings.embed_query("LangChain 是什么？")
print(len(vector))
```

批量向量化：

```python
vectors = embeddings.embed_documents([
    "LangChain 是 LLM 应用框架",
    "Milvus 是向量数据库",
])
```

## 4. VectorStore 是什么

VectorStore 是向量数据库接口，负责：

```text
存储文本 chunk
存储 embedding
存储 metadata
执行相似度搜索
返回相关 Document
```

常见向量库：

| 向量库 | 特点 |
|--------|------|
| FAISS | 本地轻量，适合 demo 和单机 |
| Chroma | 本地开发友好 |
| Milvus | 分布式，适合生产 |
| Pinecone | 云服务，托管方便 |
| Weaviate | 支持混合搜索和 schema |

## 5. 用 FAISS 构建向量库

```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

docs = [
    Document(page_content="LangChain 用于构建 LLM 应用。", metadata={"source": "intro"}),
    Document(page_content="Milvus 是向量数据库。", metadata={"source": "milvus"}),
]

vectorstore = FAISS.from_documents(docs, embeddings)

results = vectorstore.similarity_search("什么是向量数据库？", k=2)

for doc in results:
    print(doc.page_content, doc.metadata)
```

## 6. Retriever 是什么

Retriever 是统一检索接口。

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke("LangChain 可以做什么？")
```

和 VectorStore 的区别：

| 概念 | 说明 |
|------|------|
| VectorStore | 负责存储和搜索向量 |
| Retriever | 面向 RAG 的检索接口 |

RAG 中一般把 Retriever 接到 LCEL 链里。

## 7. Metadata 过滤

metadata 可以保存来源、用户、时间、权限等信息。

```python
Document(
    page_content="订单系统说明",
    metadata={"source": "order.md", "tenant_id": "t1"}
)
```

生产系统必须考虑权限过滤，不能让用户检索到不属于自己的文档。

## 8. 相似度度量

| 度量 | 说明 |
|------|------|
| COSINE | 文本语义检索常用 |
| L2 | 欧氏距离，图像或数值特征常见 |
| IP | 内积，推荐系统常见 |

注意：建索引和检索时的 metric 要一致。

## 9. 常见问题

| 问题 | 原因 |
|------|------|
| 检索不到 | chunk 切分差、embedding 不合适、k 太小 |
| 检索结果不准 | 只靠向量相似度，缺少 rerank |
| 成本高 | 重复 embedding，没有缓存 |
| 数据串租户 | metadata 权限过滤缺失 |
| 更新困难 | 缺少文档 ID 和版本管理 |

## 10. 实战建议

- Demo 用 FAISS 或 Chroma
- 生产用 Milvus、Pinecone、Weaviate 等
- 为每个 chunk 保存 `source`、`page`、`doc_id`
- 大文档要设计增量更新和删除策略
- RAG 不要只依赖向量检索，重要场景加 rerank 和 metadata filter

