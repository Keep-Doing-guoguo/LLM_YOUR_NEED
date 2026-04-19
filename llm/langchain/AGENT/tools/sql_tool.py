#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/10/11 17:49
@source from: 
"""
# ====== 3) 工具二：SQLite 查询 ======
# ✅ 使用 SQLite 替代 MySQL
import sqlite3
import os
import json

from langchain.agents import Tool


# 指定本地 SQLite 文件路径
SQLITE_PATH = os.getenv("SQLITE_PATH", "/Volumes/PSSD/未命名文件夹/donwload/chat/knowledge_base/info.db")

def run_sqlite(sql_text: str, limit: int = 50) -> str:
    """
    执行 SQLite 查询语句。
    仅支持 SELECT 查询，防止误操作。
    """
    sql_text = sql_text.strip().rstrip(";")
    if not sql_text.lower().startswith("select"):
        return "❌ 仅允许执行 SELECT 查询。"

    try:
        conn = sqlite3.connect(SQLITE_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_text)
        rows = cursor.fetchmany(limit)
        col_names = [desc[0] for desc in cursor.description] if cursor.description else []
        data = [dict(zip(col_names, row)) for row in rows]
        conn.close()

        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"SQLite执行失败: {e}"

tool_sql = Tool(
    name="SQLiteQuery",
    func=run_sqlite,
    description=(
        "用于执行SQLite数据库中的SQL查询，返回JSON结果。"
        "输入必须是完整的SELECT语句，例如：SELECT * FROM users LIMIT 3;"
    )
)

if __name__ == '__main__' :
    print(run_sqlite('SELECT * FROM message LIMIT 10',10))
    pass
