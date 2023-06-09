# -*- coding: utf-8 -*-
"""산학협력_태양전지.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CCXb94Oq9cFWOPK1tQneg8k-xy4JSl0H
"""

import pandas as pd
import math

from bs4 import BeautifulSoup
import requests
import json
import re

import warnings
warnings.filterwarnings(action='ignore')

# 온도, 습도 크롤링
url = 'https://weather.naver.com/today/09290600?cpName=KMA'

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'}
html = requests.get(url, headers = headers)
soup = BeautifulSoup(html.text, "html.parser")

a = str(soup).index('hourlyFcastListJson')
b = str(soup)[a:].index(']')
data = str(soup)[a:a+b+1]

d = data[len('hourlyFcastListJson = '):]
d = d.replace('null', '"null"')
d = d.replace('false', '"false"')
d = d.replace('true', '"true"')
d = json.loads(d)

data = pd.DataFrame(d) # 데이터 프레임으로 날씨 정보 모두 확인 가능
data = data.drop(data.index[-1])
data['dewpoint'] = 0

# aplYmd: 날짜
# aplTm: 시간
# tmpr: 기온
# humd 습도

a = 243.12
b = 17.62

for i in range(len(data)):
  gamma = (b * data['tmpr'][i] / (a + data['tmpr'][i])) + math.log(data['humd'][i] / 100.0)
  dewpoint = (a * gamma) / (b - gamma)
  data['dewpoint'][i] = dewpoint

# 날짜별로 이슬점이 가장 낮은 시각 표시
min_dewpoints = data.groupby('aplYmd')['dewpoint'].idxmin()
min_dwp_time = data.loc[min_dewpoints, ['aplYmd', 'aplTm']]

print("날짜별로 이슬점이 가장 낮은 시각은 다음과 같습니다:")
for _, row in min_dwp_time.iterrows():
  print(f"{row['aplYmd']}: {row['aplTm']}시")


min_dwp_times = data.groupby('aplYmd').apply(lambda x: x.nsmallest(3, 'dewpoint'))

print()
print("날짜별로 이슬점이 가장 낮은 top 3 시각은 다음과 같습니다: ")

for i, row in min_dwp_times.iterrows():
  dp = format(row['dewpoint'],".4f")
  print(f"{row['aplYmd']}: {row['aplTm']}시에 이슬점이 {dp}로 낮습니다.")

data.head() # 데이터 프레임으로 날씨 정보 모두 확인 가능

