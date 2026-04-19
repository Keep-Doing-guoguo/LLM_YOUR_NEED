#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/10/11 17:51
@source from: 
"""
# ====== 4) 工具三：Milvus 语义检索 ======
from pymilvus import connections, Collection
import os
import json
from typing import List
import requests
from langchain.agents import Tool

MILVUS_HOST = os.getenv("MILVUS_HOST", "10.40.100")
MILVUS_PORT = os.getenv("MILVUS_PORT", "9997")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "sms_messages_bge1024")
EMBED_URL = os.getenv("EMBED_URL", "http://127.0.0.1:8000/embedding")  # 你的 embedding 接口

_connections_done = False
_collection_cache = None

def _get_collection() -> Collection:
    global _connections_done, _collection_cache
    if not _connections_done:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        _connections_done = True
    if _collection_cache is None:
        _collection_cache = Collection(MILVUS_COLLECTION)
        _collection_cache.load()
    return _collection_cache

def _embed(texts: List[str]) -> List[List[float]]:
    """调用本地 embedding 服务"""
    r = requests.post(EMBED_URL, json={"texts": texts}, timeout=60)
    r.raise_for_status()
    return r.json()["embeddings"]

def milvus_search(query: str, top_k: int = 5) -> str:
    """Milvus 相似文本检索"""
    try:
        embs = _embed([query])
        col = _get_collection()
        res = col.search(
            data=embs,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 32}},
            limit=top_k,
            output_fields=["text", "prob"]
        )
        hits = res[0] if res else []
        out = []
        for h in hits:
            out.append({
                "text": h.entity.get("text"),
                "prob": float(h.entity.get("prob") or 0.0),
                "distance": float(h.distance),
                "similarity": round(1.0 - float(h.distance), 4)
            })
        return json.dumps(out, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Milvus检索失败: {e}"

tool_milvus = Tool(
    name="MilvusSearch",
    func=milvus_search,
    description=(
        "用于语义相似检索（短信/文本语料）。输入自然语言查询，"
        "自动向量化并在 Milvus 中召回最相似的文本，返回 JSON（text, prob, similarity）。"
    )
)

if __name__ == '__main__':
    print(milvus_search('帮我在短信库里找到与“您的申通快递已到站，请凭手机号取件”相似的记录，给我前 5 条。',5))
    pass