#!/usr/bin/env python
# coding=utf-8

"""
MCP Server - 和风天气 Demo（使用 city lookup 动态查询城市 ID）
工具：
  get_weather(city: str) -> dict

说明：
- city 可以传中文（“北京”“济南”“郑州”）、拼音（beijing/jinan），
  我们通过和风的「城市查询」接口把它转成 location id，
  然后再调用「实时天气」接口查天气。
"""

from typing import Dict, Any
import requests
from mcp.server.fastmcp import FastMCP

# ====== 你的和风 API KEY ======
HEFENG_API_KEY = "757cf7ccc6f84582827da7b11596ce8e"

# ====== 你的专属域名（已验证可用）======
BASE_URL = "https://kp4d922ray.re.qweatherapi.com"

# 城市查询（City Lookup），注意路径是 geo/v2/city/lookup
GEO_URL = f"{BASE_URL}/geo/v2/city/lookup"
# 实时天气
NOW_URL = f"{BASE_URL}/v7/weather/now"

mcp = FastMCP("HeFengWeatherMCP")


def _lookup_location_id(city_name: str) -> str:
    """
    使用和风 City Lookup 根据城市名查询 location id
    - city_name: 可以是中文名（济南）、拼音（jinan）、模糊（beij）
    - 成功返回第一个匹配城市的 id
    - 失败抛异常
    """
    city_name = city_name.strip()
    if not city_name:
        raise ValueError("city_name 不能为空")

    # 按你 curl 的方式：Header 传 API Key，Query 传 location
    headers = {
        "X-QW-Api-Key": HEFENG_API_KEY
    }
    params = {
        "location": city_name
    }

    resp = requests.get(GEO_URL, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    # City Lookup 成功 code 一般为 "200"
    code = data.get("code")
    if code != "200":
        raise RuntimeError(f"City Lookup error: {code} - {data.get('message', '')}")

    locations = data.get("location") or []
    if not locations:
        raise RuntimeError(f"未找到城市：{city_name}")

    # 取第一个匹配
    first = locations[0]
    return first["id"]


def _get_now_weather(location_id: str) -> Dict[str, Any]:
    """
    直接调用和风 v7 实况天气接口
    - location_id: 形如 101120101 的城市 ID
    """
    params = {
        "location": location_id,
        "key": HEFENG_API_KEY,
    }
    resp = requests.get(NOW_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    code = data.get("code")
    if code != "200":
        # 和风错误码 + message
        raise RuntimeError(f"Weather API error: {code} - {data.get('message', '')}")

    return data.get("now", {})


@mcp.tool()
def get_weather(city: str) -> Dict[str, Any]:
    """
    查询某个城市的实时天气（和风天气）

    参数：
      - city: 城市名称，例如 "济南"、"郑州"、"北京" 或 "jinan"、"beijing"

    返回：
      - 包含温度、体感温度、天气现象、风力、湿度等信息的字典
    """
    city = city.strip()
    if not city:
        raise ValueError("city 不能为空")

    # 1）城市名 -> location id
    location_id = _lookup_location_id(city)

    # 2）location id -> 实况天气
    now = _get_now_weather(location_id)

    # 3）整理输出字段，方便 Agent 使用
    return {
        "city": city,
        "location_id": location_id,
        "obs_time": now.get("obsTime"),
        "text": now.get("text"),           # 天气现象，如 "多云"
        "temp": now.get("temp"),           # 温度，摄氏度
        "feels_like": now.get("feelsLike"),
        "humidity": now.get("humidity"),   # 相对湿度 %
        "wind_dir": now.get("windDir"),
        "wind_scale": now.get("windScale"),
        "pressure": now.get("pressure"),   # 大气压 hPa
        "vis": now.get("vis"),             # 能见度 km
    }


# ========= 调试 / MCP 两种模式 =========
if __name__ == "__main__":
    # 1）本地直接测试，不通过 Agent：
    print(">>> 测试：济南")
    print(get_weather("济南"))
    print(">>> 测试：jinan")
    print(get_weather("jinan"))
    print(">>> 测试：北京")
    print(get_weather("北京"))

    # 2）如果要给 Agent 调用，就把上面几行 print 注释掉，改用：
    # mcp.run()