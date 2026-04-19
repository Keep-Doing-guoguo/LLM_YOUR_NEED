#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2025/4/13 18:18
@source from: 
"""
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import requests
from datetime import datetime
import re
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxx"
model = ChatOpenAI(
    temperature=0,
    openai_api_key="你的key",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name="qwen-plus",
)
# 1️⃣ 定义 Prompt 模板（包含历史对话）
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
你是一个智能助手，以下是你和用户的对话历史：
{history}
用户: {input}
助手:"""
)
_PROMPT_TEMPLATE = """
用户会提出一个关于天气的问题，你的目标是拆分出用户问题中的区，市 并按照我提供的工具回答。
例如 用户提出的问题是: 上海浦东未来1小时天气情况？
则 提取的市和区是: 上海 浦东
如果用户提出的问题是: 上海未来1小时天气情况？
则 提取的市和区是: 上海 None
请注意以下内容:
1. 如果你没有找到区的内容,则一定要使用 None 替代，否则程序无法运行
2. 如果用户没有指定市 则直接返回缺少信息

问题: ${{用户的问题}}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。
```text

${{拆分的市和区，中间用空格隔开}}
```
... weathercheck(市 区)...
```output

${{提取后的答案}}
```
答案: ${{答案}}



这是一个例子：
问题: 上海浦东未来1小时天气情况？


```text
上海 浦东
```
...weathercheck(上海 浦东)...

```output
预报时间: 1小时后
具体时间: 今天 18:00
温度: 24°C
天气: 多云
风向: 西南风
风速: 7级
湿度: 88%
降水概率: 16%

Answer: 上海浦东一小时后的天气是多云。

现在，这是我的问题：

问题: {question}
"""
PROMPT = PromptTemplate(input_variables=["question"], template=_PROMPT_TEMPLATE)

def get_city_info(location, adm, key):
    base_url = 'https://geoapi.qweather.com/v2/city/lookup?'
    params = {'location': location, 'adm': adm, 'key': key}
    response = requests.get(base_url, params=params)
    return response.json()

def format_weather_data(data, place):
    hourly_forecast = data['hourly']
    now = datetime.now()
    formatted_data = f"\n 这是查询到的关于{place}未来24小时的天气信息: \n"
    for forecast in hourly_forecast:
        forecast_time = datetime.strptime(forecast['fxTime'], '%Y-%m-%dT%H:%M%z')
        tz = forecast_time.tzinfo
        now = datetime.now(tz)
        days_diff = (forecast_time.date() - now.date()).days
        if days_diff == 0:
            date_str = '今天'
        elif days_diff == 1:
            date_str = '明天'
        else:
            date_str = f"{days_diff}天后"
        time_str = date_str + ' ' + forecast_time.strftime('%H:%M')
        hours_diff = (forecast_time - now).total_seconds() // 3600
        if hours_diff < 1:
            hours_str = '1小时后'
        elif hours_diff >= 24:
            hours_str = f"{int(hours_diff // 24)}天"
        else:
            hours_str = f"{int(hours_diff)}小时"
        formatted_data += f"预报时间: {time_str}  距离现在有: {hours_str}\n温度: {forecast['temp']}°C\n天气: {forecast['text']}\n风向: {forecast['windDir']}\n风速: {forecast['windSpeed']}级\n湿度: {forecast['humidity']}%\n降水概率: {forecast['pop']}%\n\n"
    return formatted_data

def get_weather(key, location_id, place):
    url = "https://devapi.qweather.com/v7/weather/24h?"
    params = {'location': location_id, 'key': key}
    response = requests.get(url, params=params)
    return format_weather_data(response.json(), place)

def split_query(query):
    parts = query.split()
    adm = parts[0]
    location = parts[1] if len(parts) > 1 and parts[1] != 'None' else adm
    return location, adm

def weather(query):
    location, adm = split_query(query)
    key = "kp4d922ray.re.qweatherapi.com"
    try:
        city_info = get_city_info(location, adm, key)
        location_id = city_info['location'][0]['id']
        place = adm + "市" + location + "区"
        return get_weather(key, location_id, place)
    except KeyError:
        try:
            city_info = get_city_info(adm, adm, key)
            location_id = city_info['location'][0]['id']
            place = adm + "市"
            return get_weather(key, location_id, place) + "（重要提示：只返回市级天气）\n"
        except KeyError:
            return "输入的地区不存在，无法提供天气预报"


def weathercheck(user_query):
    chain = LLMChain(llm=model, prompt=PROMPT)
    output = chain.run(question=user_query)

    match = re.search(r"```text(.*?)```", output, re.DOTALL)
    if not match:
        return "无法识别位置格式"

    expression = match.group(1).strip()
    weather_output = weather(expression)
    return "Answer: " + weather_output


if __name__ == "__main__":
    print(weathercheck(user_query='北京的天气如何'))
    print('debug')