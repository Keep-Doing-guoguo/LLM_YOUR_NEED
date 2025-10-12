#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/4 14:09
@source from: 
"""
# -- 创建 messages 表
# CREATE TABLE IF NOT EXISTS messages (
#     id TEXT PRIMARY KEY,
#     conversation_id TEXT NOT NULL,
#     query TEXT NOT NULL,
#     response TEXT,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
import sqlite3
from typing import List, Dict, Optional

DB_PATH = "chat_memory.db"  # SQLite 文件路径


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()


def insert_message(message_id: str, conversation_id: str, query: str, response: Optional[str] = None):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (id, conversation_id, query, response) 
            VALUES (?, ?, ?, ?)
        ''', (message_id, conversation_id, query, response))
        conn.commit()


def update_message(message_id: str, response: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages SET response = ? WHERE id = ?
        ''', (response, message_id))
        conn.commit()


def filter_message(conversation_id: str, limit: int = 10) -> List[Dict]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT query, response FROM messages 
            WHERE conversation_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (conversation_id, limit))
        rows = cursor.fetchall()

    return [{"query": row[0], "response": row[1]} for row in rows]

#init_db()

import sqlite3
import pandas as pd

# 数据库路径
db_path = '/Volumes/PSSD/未命名文件夹/donwload/Langchain-Chatchat-0.2.9/learning_all/langchain_l/chat_memory.db'  # ⬅️ 改成你的实际路径，比如 './mydb.sqlite'

# 要查看的表名
table_name = 'messages'       # ⬅️ 改成你实际的表名，比如 'message'

# 建立连接
conn = sqlite3.connect(db_path)

# 查询所有数据
df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# 打印内容
print(df)

# 关闭连接
conn.close()