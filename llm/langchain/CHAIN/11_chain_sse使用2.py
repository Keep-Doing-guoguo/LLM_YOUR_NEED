#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/5 12:35
@source from: 
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import AsyncCallbackHandler

import uuid
import json

app = FastAPI()
from langchain.callbacks import AsyncIteratorCallbackHandler

#https://blog.csdn.net/q506610466/article/details/132790633



# ✅ 构造 LLMChain
def get_chain(callback):
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
        callbacks=[callback],
        openai_api_key="sk-",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus"
    )

    prompt = PromptTemplate(
        input_variables=["input"],
        template="你是一个聪明的助手，用户问你：{input}"
    )

    return LLMChain(llm=llm, prompt=prompt)


# ✅ 接口：支持 stream=True（流式） 或 False（非流）
@app.get("/chat")
async def stream_chat(request: Request, query: str, stream: bool = True):
    message_id = str(uuid.uuid4())
    # ✅ 自定义流式回调器
    callback = AsyncIteratorCallbackHandler()

    chain = get_chain(callback)


    async def chat_iterator():
        task = chain.arun(input=query)
        if stream:
            async for token in callback.aiter():
                yield f"data: {json.dumps({'text': token, 'message_id': message_id}, ensure_ascii=False)}\n\n"
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield f"data: {json.dumps({'text': answer, 'message_id': message_id}, ensure_ascii=False)}\n\n"
        await task



    return EventSourceResponse(chat_iterator())


# ✅ 主函数入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("1004:app", host="0.0.0.0", port=8011, reload=True)

#curl -N "http://localhost:8011/chat?query=LangChain是干嘛的&stream=true"
"""
curl -N -H "Accept: text/event-stream" --get \
  --data-urlencode "query=介绍一下朱元璋" \
  --data-urlencode "stream=true" \
  "http://localhost:8011/chat"
"""

