#!/usr/bin/env python
# coding=utf-8

"""
@author: zgw
@date: 2026/1/12 14:55
@source from: 
"""
#!/usr/bin/env python
# coding=utf-8

import requests

# 替换成你的 API KEY
API_KEY = "757cf7ccc6f84582827da7b11596ce8e"

# 要查询的城市
city_name = "济南"

# city lookup 接口地址（GeoAPI）
url = "https://kp4d922ray.re.qweatherapi.com/v7/city/lookup"

params = {
    "location": city_name,
    "key": API_KEY,
}
'''
curl --compressed \
-H "X-QW-Api-Key: 757cf7ccc6f84582827da7b11596ce8e" \
'https://kp4d922ray.re.qweatherapi.com/v7/weather/now?location=101010100'


curl --compressed \
  -H "X-QW-Api-Key: 757cf7ccc6f84582827da7b11596ce8e" \
  "https://kp4d922ray.re.qweatherapi.com/geo/v2/city/lookup?location=beij"

'''
try:
    resp = requests.get(url, params=params, timeout=10)
    print("请求 URL:", resp.url)
    print("状态码:", resp.status_code)
    print("返回内容:", resp.text)
except Exception as e:
    print("请求失败:", str(e))