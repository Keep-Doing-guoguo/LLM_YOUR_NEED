#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/5 12:18
@source from: 
"""
#好嘞！以下是使用 LLMChain + ChatOpenAI 实现 大模型对话 + 流式输出（SSE） 的完整代码 💡


# app.py
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
import uvicorn

app = FastAPI()

# ✅ Prompt 模板
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
你是一个聪明的助手，以下是你和用户的历史对话：
{history}
用户: {input}
助手:"""
)

# ✅ Memory 缓存
memory = ConversationBufferMemory(memory_key="history", return_messages=False)

# ✅ LLM 初始化函数（带 callback）
def get_llm(callbacks=None):
    return ChatOpenAI(
        openai_api_key="sk-",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        streaming=True,
        callbacks=callbacks,
        temperature=0.7
    )

# ✅ 构造 LLMChain
def get_chain(callbacks=None):
    return LLMChain(
        llm=get_llm(callbacks),
        prompt=prompt,
        memory=memory,
        verbose=True
    )

# ✅ FastAPI 路由：流式 or 普通输出
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query")
    stream = data.get("stream", False)

    callback = AsyncIteratorCallbackHandler()
    chain = get_chain(callbacks=[callback])

    task = chain.ainvoke({"input": query})  # 异步触发执行（重要）

    if stream:
        async def event_generator():
            async for token in callback.aiter():
                yield f"data: {json.dumps({'answer': token}, ensure_ascii=False)}\n\n"
            yield f"data: [DONE]\n\n"
            await task
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        answer = ""
        async for token in callback.aiter():
            answer += token
        await task
        return JSONResponse(content={"answer": answer})


# ✅ 主函数入口
if __name__ == "__main__":
    uvicorn.run("1002:app", host="0.0.0.0", port=8011, reload=True)