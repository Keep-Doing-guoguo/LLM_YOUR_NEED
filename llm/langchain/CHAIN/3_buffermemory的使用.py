#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/4 14:03
@source from: 
"""
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
import sqlite3
import uuid

# ========== 数据库处理 ============
DB_PATH = "chat_memory.db"

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

def insert_message(message_id, conversation_id, query, response=None):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (id, conversation_id, query, response) VALUES (?, ?, ?, ?)
        ''', (message_id, conversation_id, query, response))
        conn.commit()

def update_message(message_id, response):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE messages SET response = ? WHERE id = ?
        ''', (response, message_id))
        conn.commit()

def show_all():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM messages")
        for row in cursor.fetchall():
            print(row)

# ========== LangChain 设置 ============
class DBCallbackHandler(BaseCallbackHandler):
    def __init__(self, conversation_id, message_id, query):
        self.conversation_id = conversation_id
        self.message_id = message_id
        self.query = query
        insert_message(message_id, conversation_id, query)  # 插入问题

    def on_llm_end(self, response: LLMResult, **kwargs):
        answer = response.generations[0][0].text.strip()
        update_message(self.message_id, answer)  # 回答落库

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
你是一个有帮助的智能助手。
以下是用户和你的历史对话：
{history}
用户: {input}
助手:"""
)

memory = ConversationBufferWindowMemory(memory_key="history", return_messages=False, k=10)

def get_llm(callbacks):
    return ChatOpenAI(
        openai_api_key="sk-",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        temperature=0.7,
        callbacks=callbacks
    )

def build_chain(callbacks):
    return LLMChain(llm=get_llm(callbacks), prompt=prompt, memory=memory, verbose=True)

if __name__ == "__main__":
    init_db()
    conversation_id = "conv-001"
    inputs = ["你好", "你是谁", "你能做什么"]

    for question in inputs:
        message_id = str(uuid.uuid4())
        handler = DBCallbackHandler(conversation_id, message_id, question)
        chain = build_chain(callbacks=[handler])
        res = chain.invoke({"input": question})
        print("🤖", res["text"].strip())

    print("\n✅ 数据库内容如下:")
    show_all()
