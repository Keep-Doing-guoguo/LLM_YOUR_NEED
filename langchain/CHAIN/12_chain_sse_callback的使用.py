#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/6 12:31
@source from: 
"""
from fastapi import Body
from langchain_community.chat_models import ChatOpenAI
from sse_starlette.sse import EventSourceResponse
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
import json
from fastapi import FastAPI, Body

from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from langchain.prompts import PromptTemplate

# ✅ 创建 FastAPI 应用实例
app = FastAPI()
# ✅ 定义流式响应路由
@app.post("/chat")
async def chat(query: str = Body(..., embed=True)):

    async def chat_iterator() -> AsyncIterable[str]:
        stream = True
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]

        model = ChatOpenAI(
            openai_api_key="sk-",
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model_name="qwen-plus",
            callbacks = callbacks
        )

        # 构造提示模板
        prompt = PromptTemplate(
            input_variables=["input"],
            template="你是一个有用的助手，用户问：{input}"
        )

        chain = LLMChain(prompt=prompt, llm=model)

        # Begin a task that runs in the background.
        # task = asyncio.create_task(
        #     chain.acall({"input": query}),
        #     callback.done
        # )
        # 异步执行 LLMChain
        task = asyncio.create_task(chain.arun(input=query))
        print('到这里了！')
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                print(token)
                yield json.dumps({"text": token, "message_id": 111},ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": 111},ensure_ascii=False)

        await task

    return EventSourceResponse(chat_iterator())
# ✅ 主函数入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("1005:app", host="0.0.0.0", port=8001, reload=True)
    """
    curl -X POST "http://127.0.0.1:8001/chat" \
-H "Content-Type: application/json" \
-d '{"query": "你好"}' -v
    """