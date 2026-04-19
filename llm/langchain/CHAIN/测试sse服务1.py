#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/9 09:57
@source from: 
"""
import requests
import json

def request_stream():
    url = "http://localhost:8001/chat"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "query": "介绍一下朱元璋",
        "stream": True
    }

    # 发送 POST 请求，开启流式响应
    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)

    # 检查状态码
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                try:
                    # SSE 每条数据是以 "data: " 开头的
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data:"):
                        data = decoded.replace("data: ", "")
                        print("🧠 返回片段：", json.loads(data)["text"])
                except Exception as e:
                    print("❌ 解码失败：", e)
    else:
        print("❌ 请求失败，状态码：", response.status_code)

# 执行
request_stream()