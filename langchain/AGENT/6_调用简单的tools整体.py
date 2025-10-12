#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/6 19:03
@source from: 
"""
#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/6
@desc: 简化版 Agent 控制台程序，串行执行
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import sqlite3
import requests
from typing import List

from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import Tool  # 0.2+ 推荐
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# ========== 1) LLM ==========
LLM = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    api_key="你的key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ========== 2) 工具：Calculator ==========
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"计算错误: {e}"

tool_calculator = Tool(
    name="Calculator",
    description="用于数学表达式计算，例如：2 + 2、3 * (5+7)。输入必须是可执行的 Python 数学表达式字符串。",
    func=calculator,   # 注意这里是函数对象，不是字符串
)

# ========== 3) 工具：SQLite 查询 ==========
SQLITE_PATH = os.getenv("SQLITE_PATH", "./example.db")

def run_sqlite(sql_text: str) -> str:
    sql = sql_text.strip().rstrip(";")
    if not sql.lower().startswith("select"):
        return "❌ 仅允许执行 SELECT 查询。"
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        data = [dict(zip(cols, r)) for r in rows]
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"SQLite执行失败: {e}"

tool_sqlite = Tool(
    name="SQLiteQuery",
    description="执行 SQLite 的 SELECT 语句，返回 JSON。例如：SELECT id, name FROM users LIMIT 3;",
    func=run_sqlite,
)

# ========== 4) 工具：Milvus 语义检索 ==========
from pymilvus import connections, Collection

MILVUS_HOST = os.getenv("MILVUS_HOST", "10.40.100.16")
MILVUS_PORT = os.getenv("MILVUS_PORT", "9997")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "sms_messages_bge1024")
EMBED_URL = os.getenv("EMBED_URL", "http://127.0.0.1:8000/embedding")

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
    r = requests.post(EMBED_URL, json={"texts": texts}, timeout=60)
    r.raise_for_status()
    return r.json()["embeddings"]

def milvus_search(query: str) -> str:
    try:
        embs = _embed([query])
        col = _get_collection()
        res = col.search(
            data=embs,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 32}},
            limit=5,
            output_fields=["text", "prob"],
        )
        hits = res[0] if res else []
        out = []
        for h in hits:
            out.append({
                "text": h.entity.get("text"),
                "prob": float(h.entity.get("prob") or 0.0),
                "distance": float(h.distance),
                "similarity": round(1.0 - float(h.distance), 4),
            })
        return json.dumps(out, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Milvus检索失败: {e}"

tool_milvus = Tool(
    name="MilvusSearch",
    description="语义相似检索：输入自然语言，返回 Milvus 中最相似的文本（JSON 格式）。",
    func=milvus_search,
)

# ========== 5) 组装 Agent (React) ==========
tools = [tool_calculator, tool_sqlite, tool_milvus]  # ⚠️ 确保里面全是 Tool 实例

# 标准 ReAct 提示词（从 Hub 拉取官方模板）
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=LLM,
    tools=tools,
    prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# ========== 6) 测试 ==========
def demo():
    print("\n=== 计算器 ===")
    print(agent_executor.invoke({"input": "3 * (5 + 7) 等于多少？"})["output"])

    print("\n=== SQLite ===")
    # 先准备个简单表测试（只需第一次执行）
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
    cur.execute("INSERT INTO users (name) VALUES ('Alice'),('Bob'),('Carol')")
    conn.commit()
    conn.close()

    print(agent_executor.invoke({"input": "请用SQL查询：SELECT id, name FROM users ORDER BY id LIMIT 3;"})["output"])

    print("\n=== Milvus 检索 ===")
    print(agent_executor.invoke({"input": "帮我在短信库里找到与“您的申通快递已到站，请凭手机号取件”相似的记录，给我前 5 条。"})["output"])

if __name__ == "__main__":
    demo()