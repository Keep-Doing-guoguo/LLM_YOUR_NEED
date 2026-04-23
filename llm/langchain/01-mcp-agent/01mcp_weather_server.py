#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2026/1/8 16:20
@source from: 
"""
# mcp_weather_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherMCP")

@mcp.tool()
def get_weather(city: str) -> dict:
    """
    Return current weather for a city (demo stub).
    Replace this with a real weather API call if you want.
    """
    city = city.strip()
    # 这里写死一份示例数据（最小可跑）
    fake = {
        "beijing": {"weather": "sunny", "temperature_c": 28.3},
        "shanghai": {"weather": "cloudy", "temperature_c": 25.1},
        "jinan": {"weather": "rain", "temperature_c": 22.8},
        "zhengzhou": {"weather": "sunny", "temperature_c": 27.0},
    }
    key = city.lower()
    data = fake.get(key, {"weather": "unknown", "temperature_c": None})
    return {"city": city, **data}

if __name__ == "__main__":
    # 默认 stdio 方式运行（Agent 会用子进程拉起它）
    mcp.run()