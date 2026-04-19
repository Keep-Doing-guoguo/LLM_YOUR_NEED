#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/5 12:48
@source from: 
"""
import requests

def _test_sse_chat():
    url = "http://localhost:8011/chat"
    params = {
        "query": "LangChain 是干嘛的",
        "stream": "true"
    }

    headers = {
        "Accept": "text/event-stream"
    }

    with requests.get(url, params=params, headers=headers, stream=True) as response:
        print("📡 Connected to SSE stream...")
        for line in response.iter_lines(decode_unicode=True):
            if line.strip() == "":
                continue
            if line.startswith("data: "):
                json_data = line.replace("data: ", "")
                print("🧠 Token:", json_data)


if __name__ == "__main__":
    _test_sse_chat()