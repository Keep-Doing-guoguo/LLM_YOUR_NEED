#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/5 12:24
@source from: 
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import time

app = FastAPI()

@app.get("/stream")
async def stream_response():


    async def fake_llm_stream():
        text = "你好，我是大模型，很高兴见到你。"
        for char in text:
            yield f"data: {char}\n\n"
            time.sleep(0.2)
        yield "data: [DONE]\n\n"


    return StreamingResponse(fake_llm_stream(), media_type="text/event-stream")



# ✅ 主函数入口（仅运行该脚本时生效）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("1003:app", host="0.0.0.0", port=8011, reload=True)

#curl -N http://localhost:8011/stream

# data: 你
#
# data: 好
#
# data: ，
#
# data: 我
#
# data: 是
#
# data: 大
#
# data: 模
#
# data: 型
#
# data: ，
#
# data: 很
#
# data: 高
#
# data: 兴
#
# data: 见
#
# data: 到
#
# data: 你
#
# data: 。
#
# data: [DONE]

