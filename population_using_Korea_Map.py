# -*- coding: utf-8 -*-
"""
세 번째 주제 : Folium을 얼마나 다룰 수 있는지....
우리나라 인구 소멸 위기 지역 분석

인구 소멸 위기 지역:
‘한국의 ‘지방소멸’에 관한 7가지 분석’ 보고서를 쓴 
이상호 한국고용정보원 부연구위원의 분석 방법을 이용. 

65세 이상 노인 인구와 20∼39세 여성 인구를 비교해 
젊은 여성 인구가 노인 인구의 절반에 미달할 경우, 
‘소멸 위험 지역’으로 분류하는 방식.
"""
# 모듈 import
import pandas as pd
import numpy as np

import platform

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

import folium
import json

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

path = "c:/Windows/Fonts/malgun.ttf"

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')    

plt.rcParams['axes.unicode_minus'] = False

#----------- 여기까지 사전 설정 부분 ------------------

### 1. 인구 데이터 확보하고 정리 : 05. population_raw_data.xlsx
population = pd.read_excel("./data/05. population_raw_data.xlsx", header=1)

# NaN 부분을 해당 컬럼의 상위 데이터로 채우기 : DataFrame.fillna(method='pad')
population.fillna(method="pad", inplace=True)


# 컬럼명이 길거나 특수기호가 포함된 컬럼명을 변경
population.rename(columns={"행정구역(동읍면)별(1)":"광역시도", "행정구역(동읍면)별(2)":"시도", "계":"인구수"}, inplace=True)

population = population[(population["시도"] != "소계")]

population.rename(columns={"항목":"구분"}, inplace=True)

population.loc[population["구분"] ==  "총인구수 (명)", "구분" ] = "합계"
population.loc[population["구분"] ==  "남자인구수 (명)", "구분" ] = "남자"
population.loc[population["구분"] ==  "여자인구수 (명)", "구분" ] = "여자"



### 2. 인구 소멸 위기 지역 계산하고 데이터 정리
# 1. 20~30 대 여성 인구수 파악
population['20-39세'] = population['20 - 24세'] + population['25 - 29세'] + population['30 - 34세'] + population['35 - 39세']

# 2. 65세 이상의 노인 인구수 파악
population['65세이상'] = population['65 - 69세'] + population['70 - 74세'] + \
                        population['75 - 79세'] + population['80 - 84세'] + \
                        population['85 - 89세'] + population['90 - 94세'] + \
                        population['95 - 99세'] + population['100+']

pop = pd.pivot_table(population,
                     index=["광역시도", "시도"],
                     columns=["구분"],
                     values=["인구수", "20-39세", "65세이상"])

# 3. 인구소멸지역인지 파악
# 인구소멸지역 계산을 위한 '소멸비율' 컬럼 추가
'''
소멸비율이 1보다 작으면 인구소멸지역으로 판단
'''
pop["소멸비율"] = pop['20-39세', '여자'] / (pop["65세이상", "합계"] / 2)

# 소멸위기지역 컬럼추가 : 소멸위기지역 여부를 boolean 으로 지정
pop["소멸위기지역"]  = pop["소멸비율"] < 1.0

pop[pop["소멸위기지역"] == True].index.get_level_values(1)

# 여기까지 정리된 데이터프레임의 기존 index를 초기화 하여 
# 현재 광역시도, 시도 부분(index)의 데이터를 실제 데이터로 변경
pop.reset_index(inplace=True)

# 기존 데이터프레임으로부터 컬럼명을 정리하여 리스트에 저장.
tmp_coloumns = [pop.columns.get_level_values(0)[n] +  pop.columns.get_level_values(1)[n]  for n in range(0,len(pop.columns.get_level_values(0)))]

pop.columns
'''
MultiIndex([(  '광역시도',   ''),
            (    '시도',   ''),
            ('20-39세', '남자'),
            ('20-39세', '여자'),
            ('20-39세', '합계'),
            ( '65세이상', '남자'),
            ( '65세이상', '여자'),
            ( '65세이상', '합계'),
            (   '인구수', '남자'),
            (   '인구수', '여자'),
            (   '인구수', '합계'),
            (  '소멸비율',   ''),
            ('소멸위기지역',   '')],
           names=[None, '구분'])
'''

pop.columns = tmp_coloumns
pop.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 264 entries, 0 to 263
Data columns (total 13 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   광역시도      264 non-null    object 
 1   시도        264 non-null    object 
 2   20-39세남자  264 non-null    float64
 3   20-39세여자  264 non-null    float64
 4   20-39세합계  264 non-null    float64
 5   65세이상남자   264 non-null    float64
 6   65세이상여자   264 non-null    float64
 7   65세이상합계   264 non-null    float64
 8   인구수남자     264 non-null    float64
 9   인구수여자     264 non-null    float64
 10  인구수합계     264 non-null    float64
 11  소멸비율      264 non-null    float64
 12  소멸위기지역    264 non-null    bool   
dtypes: bool(1), float64(10), object(2)
memory usage: 25.1+ KB
'''

## 지도 시각화 작업을 위한 지역별 고유 ID 생성
# 고유 ID 생성 이유 : 05. skorea_municipalities_geo_simple.json 
pop["시도"].unique()

si_name = [None] * len(pop)

# 샘플링 데이터 
tmp_gu_dict = {'수원':['장안구', '권선구', '팔달구', '영통구'], 
               '성남':['수정구', '중원구', '분당구'], 
               '안양':['만안구', '동안구'], 
               '안산':['상록구', '단원구'], 
               '고양':['덕양구', '일산동구', '일산서구'], 
               '용인':['처인구', '기흥구', '수지구'], 
               '청주':['상당구', '서원구', '흥덕구', '청원구'], 
               '천안':['동남구', '서북구'], 
               '전주':['완산구', '덕진구'], 
               '포항':['남구', '북구'], 
               '창원':['의창구', '성산구', '진해구', '마산합포구', '마산회원구'], 
               '부천':['오정구', '원미구', '소사구']}
'''
만약, '광역시도' 컬럼의 데이터가 '광역시', '특별시', '자치시' 를 포함하지 않고,
    만약, '시도'컬럼의 데이터가 '고성'으로 끝나고  '광역시도' 컬럼의 데이터가 '강원도' 끝나면
          '고성('강원)'
    그렇지않고 만약,  '시도'컬럼의 데이터가 '고성'으로 끝나고  '광역시도' 컬럼의 데이터가 '경상남도' 끝나면
          '고성('경남)'
    그렇지않으면
          '고성'

그헣지안으면
          
'''
for n in pop.index:
    if pop['광역시도'][n][-3:] not in ['광역시', '특별시', '자치시']:
        if pop['시도'][n][:-1]=='고성' and pop['광역시도'][n]=='강원도':
            si_name[n] = '고성(강원)'
        elif pop['시도'][n][:-1]=='고성' and pop['광역시도'][n]=='경상남도':
            si_name[n] = '고성(경남)'
        else:
             si_name[n] = pop['시도'][n][:-1]
                
        for keys, values in tmp_gu_dict.items():
            if pop['시도'][n] in values:
                if len(pop['시도'][n])==2:
                    si_name[n] = keys + ' ' + pop['시도'][n]
                elif pop['시도'][n] in ['마산합포구','마산회원구']:
                    si_name[n] = keys + ' ' + pop['시도'][n][2:-1]
                else:
                    si_name[n] = keys + ' ' + pop['시도'][n][:-1]
        
    elif pop['광역시도'][n] == '세종특별자치시':
        si_name[n] = '세종'
        
    else:
        if len(pop['시도'][n])==2:
            si_name[n] = pop['광역시도'][n][:2] + ' ' + pop['시도'][n]
        else:
            si_name[n] = pop['광역시도'][n][:2] + ' ' + pop['시도'][n][:-1]


pop["ID"] = si_name

del pop['20-39세남자']
del pop['65세이상남자']
del pop['65세이상여자']

# 4. 한국지도를 이용한 시각화 : 05. draw_korea_raw.xlsx
draw_korea_raw = pd.read_excel('./data/05. draw_korea_raw.xlsx')


# 각 행정구역의 화면상 좌표를 얻기 위해서 
# pivot_table의 정 반대 개념인 stack() 를 사용하여 새로운 데이터프레임 생성
draw_korea_raw_stacked = pd.DataFrame(draw_korea_raw.stack())

## stack() 함수를 이용하여생성된 데이터프레임의 컬럼명 변경
# index를 초기화
draw_korea_raw_stacked.reset_index(inplace=True)

# 컬럼명 변경
draw_korea_raw_stacked.rename(columns={"level_0":"x", "level_1":"y", 0:"ID"},
                              inplace=True)


draw_korea = draw_korea_raw_stacked

# ID 컬럼으로 지도에 표시할 때,
#시이름, 구이름으로 줄을 나누기 위해 데이터 분리를 위한 리스트 생성
BORDER_LINES = [
    [(5, 1), (5,2), (7,2), (7,3), (11,3), (11,0)], # 인천
    [(5,4), (5,5), (2,5), (2,7), (4,7), (4,9), (7,9), (7,7), (9,7), (9,5), (10,5), (10,4), (5,4)], # 서울
    [(1,7), (1,8), (3,8), (3,10), (10,10), (10,7), (12,7), (12,6), (11,6), (11,5), (12, 5), (12,4), (11,4), (11,3)], # 경기도
    [(8,10), (8,11), (6,11), (6,12)], # 강원도
    [(12,5), (13,5), (13,4), (14,4), (14,5), (15,5), (15,4), (16,4), (16,2)], # 충청북도
    [(16,4), (17,4), (17,5), (16,5), (16,6), (19,6), (19,5), (20,5), (20,4), (21,4), (21,3), (19,3), (19,1)], # 전라북도
    [(13,5), (13,6), (16,6)], # 대전시
    [(13,5), (14,5)], #세종시
    [(21,2), (21,3), (22,3), (22,4), (24,4), (24,2), (21,2)], #광주
    [(20,5), (21,5), (21,6), (23,6)], #전라남도
    [(10,8), (12,8), (12,9), (14,9), (14,8), (16,8), (16,6)], #충청북도
    [(14,9), (14,11), (14,12), (13,12), (13,13)], #경상북도
    [(15,8), (17,8), (17,10), (16,10), (16,11), (14,11)], #대구
    [(17,9), (18,9), (18,8), (19,8), (19,9), (20,9), (20,10), (21,10)], #부산
    [(16,11), (16,13)], #울산
    [(27,5), (27,6), (25,6)]
]

plt.figure(figsize=(8,11))

# 지역명 표시
for idx, row in draw_korea.iterrows():
    
    # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. 
    # (중구, 서구)
    if len(row['ID'].split())==2:
        dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
    elif row['ID'][:2]=='고성':
        dispname = '고성'
    else:
        dispname = row['ID']

    # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
    if len(dispname.splitlines()[-1]) >= 3:
        fontsize, linespacing = 9.5, 1.5
    else:
        fontsize, linespacing = 11, 1.2

    plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                 fontsize=fontsize, ha='center', va='center', 
                 linespacing=linespacing)
  

# 시도의 경계그리기
for path in BORDER_LINES:
    ys, xs = zip(*path)
    plt.plot(xs, ys, c='black', lw=1.5)

plt.gca().invert_yaxis()


plt.axis('off')

plt.tight_layout()
plt.show()

'''
인구 대한 분석결과인 pop 데이터프레임과 지도를 그리기 위한 draw_korea 
이 두가지 데이터를 합할 경우 
사용할 ID 컬럼네 문제가 없는지 확인!
'''
set(draw_korea['ID'].unique()) - set(pop['ID'].unique())
set(pop['ID'].unique()) - set(draw_korea['ID'].unique())
'''
Out[102]: {'고양', '부천', '성남', '수원', '안산', '안양', '용인', '전주', '창원', '천안', '청주', '포항'}
'''
'''
위의 결과에 따르면
pop에 행정구를 가진 시들의 데이터가 더 있다는 것이 확인 되었지만
지도에는 표시할 수 없기 때문에 삭제!
'''
tmp_list =  list(set(pop['ID'].unique()) - set(draw_korea['ID'].unique()))

for tmp in tmp_list:
    pop = pop.drop(pop[pop['ID'] == tmp].index)

# 삭제후, 데이터 재확인
set(pop['ID'].unique()) - set(draw_korea['ID'].unique())
'''
Out[105]: set()
'''

### 인구 대한 분석결과인 pop 데이터프레임과 지도를 그리기 위한 draw_korea 
#    두 데이터프레임 병합 : merge()
#    병합 기준 : ID 컬럼
pop = pd.merge(pop, draw_korea, on=['ID'], how='left')

mapdata = pop.pivot_table(index='y', columns='x', values='인구수합계')

masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)

# 위의 내용과 colormap을 생성하는 사용자 함수선언
def drawKorea(targetData, blockedMap, cmapname):

    whitelabelmin = (max(blockedMap[targetData]) - min(blockedMap[targetData]))*0.25 +  min(blockedMap[targetData])

    datalabel = targetData

    vmin = min(blockedMap[targetData])
    vmax = max(blockedMap[targetData])

    mapdata = blockedMap.pivot_table(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(9, 11))
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=cmapname, 
               edgecolor='#aaaaaa', linewidth=0.5)

    # 지역 이름 표시
    for idx, row in blockedMap.iterrows():
        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. 
        #(중구, 서구)
        if len(row['ID'].split())==2:
            dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
        elif row['ID'][:2]=='고성':
            dispname = '고성'
        else:
            dispname = row['ID']

        # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 10.0, 1.1
        else:
            fontsize, linespacing = 11, 1.

        annocolor = 'white' if row[targetData] > whitelabelmin else 'black'
        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                     fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)

    # 시도 경계 그린다.
    for path in BORDER_LINES:
        xs, ys = zip(*path)
        plt.plot(xs, ys, c='black', lw=2)

    plt.gca().invert_yaxis()

    plt.axis('off')

    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    plt.show()

#----- 사용자 함수 END ----- #


## 선언된 사용자 함수를 이용하여 인구현황 및 인구 소멸지역 확인
# 인구현황
drawKorea('인구수합계', pop, 'Blues')

# 인구 소멸 위기지역
pop['소멸위기지역'] = [ 1 if con else 0  for con in pop['소멸위기지역']]
drawKorea('소멸위기지역', pop, 'Reds')


### 인구현황에서 여성 인구 비율 확인
#----- 사용자 정의 함수 수정 START -----#

def drawKorea(targetData, blockedMap, cmapname):

    whitelabelmin = 20.

    datalabel = targetData

    tmp_max = max([ np.abs(min(blockedMap[targetData])), 
                                  np.abs(max(blockedMap[targetData]))])
    vmin, vmax = -tmp_max, tmp_max

    mapdata = blockedMap.pivot_table(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(9, 11))
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=cmapname, 
               edgecolor='#aaaaaa', linewidth=0.5)

    # 지역 이름 표시
    for idx, row in blockedMap.iterrows():
        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. 
        #(중구, 서구)
        if len(row['ID'].split())==2:
            dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
        elif row['ID'][:2]=='고성':
            dispname = '고성'
        else:
            dispname = row['ID']

        # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 10.0, 1.1
        else:
            fontsize, linespacing = 11, 1.

        annocolor = 'white' if np.abs(row[targetData]) > whitelabelmin else 'black'
        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                     fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)

    # 시도 경계 그린다.
    for path in BORDER_LINES:
        xs, ys = zip(*path)
        plt.plot(xs, ys, c='black', lw=2)

    plt.gca().invert_yaxis()

    plt.axis('off')

    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    plt.show()

#----- 사용자 정의 함수 수정 END -----#

# 여성비율
pop['여성비'] = (pop['인구수여자'] / pop['인구수합계'] - 0.5) * 100

# 여성비율 확인
drawKorea('여성비', pop, 'RdBu')


# 20-30 대 여성비
pop['2030여성비'] = (pop['20-39세여자'] / pop['20-39세합계'] - 0.5) * 100

# 20-30 대 여성비율 확인
drawKorea('2030여성비', pop, 'RdBu')


## folium을 이용한 최종 시각화 : 05. skorea_municipalities_geo_simple.json
# pop 데이터프레임의 index를 'ID' 컬럼으로 설정
pop_folium = pop.set_index('ID')

# json 데이터 로드
geo_path = './data/05. skorea_municipalities_geo_simple.json'
geo_str = json.load(open(geo_path, encoding='utf-8'))


## 인구수 합계
map = folium.Map(location=[36.2002, 127.054], 
                 zoom_start=7)

map.choropleth(geo_data=geo_str, 
               data=pop_folium['인구수합계'], 
               columns=[pop_folium.index, pop_folium['인구수합계']],
               fill_color='YlGnBu',
               key_on='feature.id')
map.save('인구수합계.html')


## 인구 소멸 위기 지역
map = folium.Map(location=[36.2002, 127.054], 
                 zoom_start=7)

map.choropleth(geo_data=geo_str, 
               data=pop_folium['소멸위기지역'], 
               columns=[pop_folium.index, pop_folium['소멸위기지역']],
               fill_color='PuRd',
               key_on='feature.id')
map.save('소멸위기지역.html')


import webbrowser
webbrowser.open_new('인구수합계.html')
webbrowser.open_new('소멸위기지역.html')









