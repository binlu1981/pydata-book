# -*- coding: utf-8 -*-
#来自URL缩短服务bit.ly的1.usa.gov数据
path = './usagov_bitly_data2012-03-16-1331923249.txt'
print(open(path).readline())
#使用json模块逐行加载数据文件
"""
open(path).readline()
Out[100]: '{ "a": "Mozilla\\/5.0 (Windows NT 6.1; WOW64) AppleWebKit\\/535.11 (KHTML, like Gecko) Chrome\\/17.0.963.78 Safari\\/535.11", "c": "US", "nk": 1, "tz": "America\\/New_York", "gr": "MA", "g": "A6qOVH", "h": "wfLQtf", "l": "orofrog", "al": "en-US,en;q=0.8", "hh": "1.usa.gov", "r": "http:\\/\\/www.facebook.com\\/l\\/7AQEFzjSi\\/1.usa.gov\\/wfLQtf", "u": "http:\\/\\/www.ncbi.nlm.nih.gov\\/pubmed\\/22415991", "t": 1331923247, "hc": 1331822918, "cy": "Danvers", "ll": [ 42.576698, -70.954903 ] }\r\n'
"""
import json
#列表推导式
#在一个打开的文件句柄上进行迭代获得一个由行组成的序列，这里为字典列表
records = [json.loads(line) for line in open(path)]
print(records[0].get('tz'))
"""
records[0]
Out[101]: 
{u'a': u'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.78 Safari/535.11',
 u'al': u'en-US,en;q=0.8',
 u'c': u'US',
 u'cy': u'Danvers',
 u'g': u'A6qOVH',
 u'gr': u'MA',
 u'h': u'wfLQtf',
 u'hc': 1331822918,
 u'hh': u'1.usa.gov',
 u'l': u'orofrog',
 u'll': [42.576698, -70.954903],
 u'nk': 1,
 u'r': u'http://www.facebook.com/l/7AQEFzjSi/1.usa.gov/wfLQtf',
 u't': 1331923247,
 u'tz': u'America/New_York',
 u'u': u'http://www.ncbi.nlm.nih.gov/pubmed/22415991'}
 """
#获悉最常出现的时区
#列表推导式取出一组时区
time_zones = [rec.get('tz') for rec in records]
print(time_zones[:10])
"""
time_zones[:10]
Out[105]: 
[u'America/New_York',
 u'America/Denver',
 u'America/New_York',
 u'America/Sao_Paulo',
 u'America/New_York',
 u'America/New_York',
 u'Europe/Warsaw',
 u'',
 u'',
 u'']
 """
#计数
def get_counts(lst):
    counts={}
    for x in lst:
        if x in counts:
            counts[x]+=1
        else:
            counts[x]=1
    return counts
#计数字典
counts = get_counts(time_zones)
print(list(counts.items())[:10])
"""
list(counts.items())[:10]
Out[111]: 
[(u'', 521),
 (u'Europe/Lisbon', 8),
 (u'Asia/Calcutta', 9),
 (u'Europe/Skopje', 1),
 (u'Europe/Copenhagen', 5),
 (u'Europe/Amsterdam', 22),
 (u'America/Phoenix', 20),
 (u'Europe/Moscow', 10),
 (u'Europe/Madrid', 35),
 (u'Asia/Dubai', 4)]
 """
#前10个最多的时区
def top_counts(count_dict, n=10):
    v_k_list = [(v,k) for k, v in count_dict.items()]
    return sorted(v_k_list,reverse=True)[:n]

print(top_counts(counts))
"""
top_counts(counts)
Out[107]: 
[(1251, u'America/New_York'),
 (521, u''),
 (400, u'America/Chicago'),
 (382, u'America/Los_Angeles'),
 (191, u'America/Denver'),
 (120, None),
 (74, u'Europe/London'),
 (37, u'Asia/Tokyo'),
 (36, u'Pacific/Honolulu'),
 (35, u'Europe/Madrid')]
 """
#使用collections.Counter类更简单
from collections import Counter
counts = Counter(time_zones)
print(counts.most_common(10))
"""
counts.most_common(10)
Out[112]: 
[(u'America/New_York', 1251),
 (u'', 521),
 (u'America/Chicago', 400),
 (u'America/Los_Angeles', 382),
 (u'America/Denver', 191),
 (None, 120),
 (u'Europe/London', 74),
 (u'Asia/Tokyo', 37),
 (u'Pacific/Honolulu', 36),
 (u'Europe/Madrid', 35)]
 """
#使用pandas计数
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
#创建表格数据结构 key是列名标签 value是值 每个字典是一行
frame = DataFrame(records)
print(frame['tz'][:10])
"""
frame.head(3)
Out[117]: 
   _heartbeat_                                                  a  \
0          nan  Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKi...   
1          nan                             GoogleMaps/RochesterNY   
2          nan  Mozilla/4.0 (compatible; MSIE 8.0; Windows NT ...   

               al   c          cy       g  gr       h                hc  \
0  en-US,en;q=0.8  US     Danvers  A6qOVH  MA  wfLQtf 1,331,822,918.000   
1             NaN  US       Provo  mwszkS  UT  mwszkS 1,308,262,393.000   
2           en-US  US  Washington  xxr3Qb  DC  xxr3Qb 1,331,919,941.000   

          hh   kw        l                        ll    nk  \
0  1.usa.gov  NaN  orofrog   [42.576698, -70.954903] 1.000   
1       j.mp  NaN    bitly  [40.218102, -111.613297] 0.000   
2  1.usa.gov  NaN    bitly     [38.9007, -77.043098] 1.000   

                                                   r                 t  \
0  http://www.facebook.com/l/7AQEFzjSi/1.usa.gov/... 1,331,923,247.000   
1                           http://www.AwareMap.com/ 1,331,923,249.000   
2                               http://t.co/03elZC4Q 1,331,923,250.000   

                 tz                                                  u  
0  America/New_York        http://www.ncbi.nlm.nih.gov/pubmed/22415991  
1    America/Denver        http://www.monroecounty.gov/etc/911/rss.php  
2  America/New_York  http://boxer.senate.gov/en/press/releases/0316...  
"""
#使用value_counts方法计数
tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])
"""
tz_counts[:10]
Out[120]: 
America/New_York       1251
Unknown                 521
America/Chicago         400
America/Los_Angeles     382
America/Denver          191
Missing                 120
Europe/London            74
Asia/Tokyo               37
Pacific/Honolulu         36
Europe/Madrid            35
Name: tz, dtype: int64
"""
#缺失值替换为Missing
clean_tz = frame['tz'].fillna('Missing')
#print(clean_tz == '')
#未知值（空字符）通过布尔型数组索引替换为Unknown
clean_tz[clean_tz ==''] ='Unknown'
tz_counts = clean_tz.value_counts()
print(tz_counts.head())

import matplotlib
import pylab
#柱状图最常出现的时区
tz_counts[:10].plot(kind='barh',rot=0)  #rot y axis label 角度
#pylab.show()

xvalues = pylab.arange(0,10,0.1)
yvalues = pylab.sin(xvalues)
pylab.plot(xvalues,yvalues)
#pylab.show()

#提取浏览器、设备等信息
print(frame['a'][1])
#浏览器信息
frame_a_list = [x.split()[0] for x in frame.a.dropna()] #保留空格前信息
results = Series(frame_a_list)
print(frame_a_list[:5])
"""
frame_a_list[:5]
Out[133]: 
[u'Mozilla/5.0',
 u'GoogleMaps/RochesterNY',
 u'Mozilla/4.0',
 u'Mozilla/5.0',
 u'Mozilla/5.0']
 """
print(results[:5])
"""
results[:5]
Out[134]: 
0               Mozilla/5.0
1    GoogleMaps/RochesterNY
2               Mozilla/4.0
3               Mozilla/5.0
4               Mozilla/5.0
dtype: object
"""
#计数
print(results.value_counts()[:8])
"""
results.value_counts()[:8]
Out[135]: 
Mozilla/5.0                 2594
Mozilla/4.0                  601
GoogleMaps/RochesterNY       121
Opera/9.80                    34
TEST_INTERNET_AGENT           24
GoogleProducer                21
Mozilla/6.0                    5
BlackBerry8520/5.0.0.681       4
dtype: int64
"""
#进一步统计WIN用户和非Win用户的时区信息，先移除缺失数据
cframe = frame[frame.a.notnull()]
# == frame.a.dropna()
print(np.where(cframe.a.str.contains('Windows')))
"""
np.where(cframe.a.str.contains('Windows'))
Out[155]: (array([   0,    2,    4, ..., 3435, 3436, 3439]),)
"""
#根据a值计算出各行是否是WIN
operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')
print(operating_system[:5])
#根据时区和最新的得到的操作系统列表对数据进行分组
by_tz_os = cframe.groupby(['tz',operating_system])
"""
by_tz_os
Out[167]: <pandas.core.groupby.DataFrameGroupBy object at 0x1232cde10>
"""
print(by_tz_os.head())
#通过size（类似value_counts)对分组结果进行计数,然后利用unstack对计数结果进行重塑
agg_counts = by_tz_os.size().unstack().fillna(0)
#print(agg_counts[:10])
"""
agg_counts[:10]
Out[169]: 
                                Not Windows  Windows
tz                                                  
                                    245.000  276.000
Africa/Cairo                          0.000    3.000
Africa/Casablanca                     0.000    1.000
Africa/Ceuta                          0.000    2.000
Africa/Johannesburg                   0.000    1.000
Africa/Lusaka                         0.000    1.000
America/Anchorage                     4.000    1.000
America/Argentina/Buenos_Aires        1.000    0.000
America/Argentina/Cordoba             0.000    1.000
America/Argentina/Mendoza             0.000    1.000
"""
#选取最常出现的时区，WIN+NOT WIN,然后打印排位索引
agg_counts[:10].sum(1) 
#axis=0是按列求和 xis=1 是按行求和
"""
agg_counts[:10].sum(1)
Out[171]: 
tz
                                 521.000
Africa/Cairo                       3.000
Africa/Casablanca                  1.000
Africa/Ceuta                       2.000
Africa/Johannesburg                1.000
Africa/Lusaka                      1.000
America/Anchorage                  5.000
America/Argentina/Buenos_Aires     1.000
America/Argentina/Cordoba          1.000
America/Argentina/Mendoza          1.000
dtype: float64
"""
indexer = agg_counts.sum(1).argsort()
print(indexer[:10])
""""
indexer[:10]
Out[172]: 
tz
                                  24
Africa/Cairo                      20
Africa/Casablanca                 21
Africa/Ceuta                      92
Africa/Johannesburg               87
Africa/Lusaka                     53
America/Anchorage                 54
America/Argentina/Buenos_Aires    57
America/Argentina/Cordoba         26
America/Argentina/Mendoza         55
dtype: int64
"""
#使用take根据排位截取最后10行
count_subset = agg_counts.take(indexer)[-10:]
#print(count_subset)
#生成堆积条形图
count_subset.plot(kind = 'barh', stacked =True)
"""
count_subset
Out[173]: 
                     Not Windows  Windows
tz                                       
America/Sao_Paulo         13.000   20.000
Europe/Madrid             16.000   19.000
Pacific/Honolulu           0.000   36.000
Asia/Tokyo                 2.000   35.000
Europe/London             43.000   31.000
America/Denver           132.000   59.000
America/Los_Angeles      130.000  252.000
America/Chicago          115.000  285.000
                         245.000  276.000
America/New_York         339.000  912.000
"""
#相对比例图
normed_subset = count_subset.div(count_subset.sum(1),axis =0)
#print(normed_subset)
normed_subset.plot(kind='barh',stacked=True)
#pylab.show()

#http://codingpy.com/article/a-quick-intro-to-pandas/

#读取英国降雨量数据
df = pd.read_csv('./uk_rain_2014.csv',header=0)
print(df.head(5))
"""
  ﻿Water Year  Rain (mm) Oct-Sep  Outflow (m3/s) Oct-Sep  Rain (mm) Dec-Feb  \
0     1980/81               1182                    5408                292   
1     1981/82               1098                    5112                257   
2     1982/83               1156                    5701                330   
3     1983/84                993                    4265                391   
4     1984/85               1182                    5364                217   

   Outflow (m3/s) Dec-Feb  Rain (mm) Jun-Aug  Outflow (m3/s) Jun-Aug  
0                    7248                174                    2212  
1                    7316                242                    1936  
2                    8567                124                    1802  
3                    8905                141                    1078  
4                    5813                343                    4313  
"""
df.columns = ['water_year','rain_octsep', 'outflow_octsep','rain_decfeb', 'outflow_decfeb', 'rain_junaug', 'outflow_junaug']
print(df.head(5))
print(len(df))
df.columns = ['water_year','rain_octsep', 'outflow_octsep','rain_decfeb', 'outflow_decfeb', 'rain_junaug', 'outflow_junaug']
print(df.head(5))
pd.options.display.float_format = '{:,.3f}'.format
print(df.describe())
"""
df.describe()
Out[179]: 
       rain_octsep  outflow_octsep  rain_decfeb  outflow_decfeb  rain_junaug  \
count       33.000          33.000       33.000          33.000       33.000   
mean     1,129.000       5,019.182      325.364       7,926.545      237.485   
std        101.900         658.588       69.995       1,692.800       66.168   
min        856.000       3,479.000      206.000       4,578.000      103.000   
25%      1,053.000       4,506.000      268.000       6,690.000      193.000   
50%      1,139.000       5,112.000      309.000       7,630.000      229.000   
75%      1,182.000       5,497.000      360.000       8,905.000      280.000   
max      1,387.000       6,391.000      484.000      11,486.000      379.000   

       outflow_junaug  
count          33.000  
mean        2,439.758  
std         1,025.914  
min         1,078.000  
25%         1,797.000  
50%         2,142.000  
75%         2,959.000  
max         5,261.000  
"""
#当我们提取列的时候，会得到一个 series ，而不是 dataframe 
# Creating a series of booleans based on a conditional
print(df['rain_octsep'].head())
print(df.rain_octsep.head())
"""
df['rain_octsep'].head()
Out[181]: 
0    1182
1    1098
2    1156
3     993
4    1182
Name: rain_octsep, dtype: int64
"""
#返回一个由布尔值构成的 dataframe。True 表示在十月-九月降雨量小于 1000 mm，False 表示大于等于 1000 mm
print(df.rain_octsep < 1000)
"""
(df.rain_octsep < 1000).head()
Out[187]: 
0    False
1    False
2    False
3     True
4    False
Name: rain_octsep, dtype: bool
"""
#用这些条件表达式来过滤现有的 dataframe
# Using a series of booleans to filter
print(df[df.rain_octsep < 1000])
"""
df[df.rain_octsep < 1000].head()
Out[182]: 
   water_year  rain_octsep  outflow_octsep  rain_decfeb  outflow_decfeb  \
3     1983/84          993            4265          391            8905   
8     1988/89          976            4330          309            6465   
15    1995/96          856            3479          245            5515   

    rain_junaug  outflow_junaug  
3           141            1078  
8           200            1440  
15          172            1439  
"""
# Filtering by multiple conditionals
print(df[(df.rain_octsep < 1000) & (df.outflow_octsep < 4000)]) # Can't use the keyword 'and'
# Filtering by string methods
print(df[df.water_year.str.startswith('199')])
# Getting a row via a numerical index
#行标签 iloc 只对数字型的标签有用。它会返回给定行的 series，行中的每一列都是返回 series 的一个元素
print(df.iloc[30])
#也许你的数据集中有年份或者年龄的列，你可能想通过这些年份或者年龄来引用行，这个时候我们就可以设置一个（或者多个）新的索引
# Setting a new index from an existing column
df = df.set_index(['water_year'])
print(df.head(5))
"""
df.head()
Out[189]: 
            rain_octsep  outflow_octsep  rain_decfeb  outflow_decfeb  \
water_year                                                             
1980/81            1182            5408          292            7248   
1981/82            1098            5112          257            7316   
1982/83            1156            5701          330            8567   
1983/84             993            4265          391            8905   
1984/85            1182            5364          217            5813   
"""
#上例中我们设置的索引列中都是字符型数据，这意味着我们不能继续使用 iloc 来引用，那我们用什么呢？用 loc
print(df.loc['2000/01'])
#和 iloc 一样，loc 会返回你引用的行，唯一一点不同就是此时你使用的是基于字符串的引用，而不是基于数字的
#ix 是基于标签的查询方法，但它同时也支持数字型索引作为备选
# Getting a row via a label-based or numerical index
print(df.ix['1999/00']) # Label based with numerical index fallback *Not recommended
#降序排序
print(df.sort_index(ascending=False).head(5)) #inplace=True to apply the sorting in place

#当你将一列设置为索引的时候，它就不再是数据的一部分了。如果你想将索引恢复为数据，调用 set_index 相反的方法 reset_index 即可
# Returning an index to data将索引恢复成数据形式
df = df.reset_index('water_year')
print(df.head(5))


#对数据集应用函数
"""
>>> df = pd.DataFrame(['05SEP2014:00:00:00.000'],columns=['Mycol'])
>>> df
                    Mycol
0  05SEP2014:00:00:00.000
>>> import datetime as dt
>>> df['Mycol'] = df['Mycol'].apply(lambda x: 
                                    dt.datetime.strptime(x,'%d%b%Y:%H:%M:%S.%f'))
>>> df
       Mycol
0 2014-09-05
"""

#有时你想对数据集中的数据进行改变或者某种操作。比方说，你有一列年份的数据，你需要新的一列来表示这些年份对应的年代。Pandas 中有两个非常有用的函数，apply 和 applymap
# Applying a function to a column
def base_year(wateryear):
    base_year = wateryear[:4]
    base_year = pd.to_datetime(base_year).year
    return base_year
df['year'] = df.water_year.apply(base_year)
print(df.head(5))
#上面的代码创建了一个叫做 year 的列，它只将 water_year 列中的年提取了出来。这就是 apply 的用法，即对一列数据应用函数。如果你想对整个数据集应用函数，就要使用 applymap


#操作数据集的结构
#另一常见的做法是重新建立数据结构，使得数据集呈现出一种更方便并且（或者）有用的形式
#Manipulating structure (groupby, unstack, pivot)
# Grouby
print(df.groupby(df.year//10*10))
"""
df.groupby(df.year//10*10)
Out[197]: <pandas.core.groupby.DataFrameGroupBy object at 0x13376f410>
"""
print(df.groupby(df.year//10*10).max())
"""
df.groupby(df.year//10*10).max()
Out[196]: 
     water_year  rain_octsep  outflow_octsep  rain_decfeb  outflow_decfeb  \
year                                                                        
1980    1989/90         1210            5701          470           10520   
1990    1999/00         1268            5824          484           11486   
2000    2009/10         1387            6391          437           10926   
2010    2012/13         1285            5500          350            9615   
"""
#groupby 会按照你选择的列对数据集进行分组。上例是按照年代分组。不过仅仅这样做并没有什么用，我们必须对其调用函数，比如 max 、 min 、mean 等等
# Grouping by multiple columns
decade_rain = df.groupby([df.year // 10 * 10, df.rain_octsep // 1000 * 1000])[['outflow_octsep','outflow_decfeb', 'outflow_junaug']].mean()
print(decade_rain)
"""
decade_rain
Out[198]: 
                  outflow_octsep  outflow_decfeb  outflow_junaug
year rain_octsep                                                
1980 0                 4,297.500       7,685.000       1,259.000
     1000              5,289.625       7,933.000       2,572.250
1990 0                 3,479.000       5,515.000       1,439.000
     1000              5,064.889       8,363.111       2,130.556
2000 1000              5,030.800       7,812.100       2,685.900
2010 1000              5,116.667       7,946.000       3,314.333
"""
#接下来是 unstack ，最开始可能有一些困惑，它可以将一列数据设置为列标签。最好还是看看实际的操作：
print(decade_rain.unstack(0))
"""
decade_rain.unstack(0)
Out[199]: 
            outflow_octsep                               outflow_decfeb  \
year                  1980      1990      2000      2010           1980   
rain_octsep                                                               
0                4,297.500 3,479.000       nan       nan      7,685.000   
1000             5,289.625 5,064.889 5,030.800 5,116.667      7,933.000   
"""
#这条语句将上例中的 dataframe 转换为下面的形式。它将第 0 列，也就是 year 列设置为列的子标签。
print(decade_rain.unstack(1))
"""
decade_rain.unstack(1)
Out[200]: 
            outflow_octsep           outflow_decfeb           outflow_junaug  \
rain_octsep           0         1000           0         1000           0      
year                                                                           
1980             4,297.500 5,289.625      7,685.000 7,933.000      1,259.000   
1990             3,479.000 5,064.889      5,515.000 8,363.111      1,439.000   
2000                   nan 5,030.800            nan 7,812.100            nan   
2010                   nan 5,116.667            nan 7,946.000            nan   
"""

# Create a new dataframe containing entries which 
# has rain_octsep values of greater than 1250
high_rain = df[df.rain_octsep > 1250]
print(high_rain)
#轴旋转其实就是我们之前已经看到的那些操作的一个集合。首先，它会设置一个新的索引（set_index()），然后对索引排序（sort_index()），最后调用 unstack 。以上的步骤合在一起就是 pivot 。
#Pivoting
#does set_index, sort_index and unstack in a row
print(high_rain.pivot('year', 'rain_octsep')[['outflow_octsep', 'outflow_decfeb', 'outflow_junaug']].fillna(''))
#注意，最后有一个 .fillna('') 。pivot 产生了很多空的记录，也就是值为 NaN 的记录。我个人觉得数据集里面有很多 NaN 会很烦，所以使用了 fillna('') 。你也可以用别的别的东西，比方说 0 。
#我们也可以使用 dropna(how = 'any') 来删除有 NaN 的行，不过这样就把所有的数据都删掉了，所以不这样做
#上面的 dataframe 展示了所有降雨超过 1250 的 outflow
#‘year'做行，'rain_octsep'做子列
"""
high_rain.pivot('year', 'rain_octsep')[['outflow_octsep', 'outflow_decfeb', 'outflow_junaug']].fillna('')
Out[202]: 
            outflow_octsep                     outflow_decfeb            \
rain_octsep           1268      1285      1387           1268      1285   
year                                                                      
1998             5,824.000                          8,771.000             
2006                                 6,391.000                            
2011                       5,500.000                          7,630.000   
"""

#合并数据集
# Merging two datasets together
"""
rain_jpn = pd.read_csv('jpn_rain.csv')
rain_jpn.columns = ['year', 'jpn_rainfall']

uk_jpn_rain = df.merge(rain_jpn, on='year')
uk_jpn_rain.head(5)
"""
#首先你需要通过 on 关键字来指定需要合并的列。通常你可以省略这个参数，Pandas 将会自动选择要合并的列。
#如下图所示，两个数据集在年份这一类上合并了。jpn_rain 数据集只有年份和降雨量两列，通过年份列合并之后，jpn_rain 中只有降雨量那一列合并到了 UK_rain 数据集中。

# Saving your data to a csv
df.to_csv('./uk_rain.csv')


#http://codingpy.com/article/a-quick-intro-to-matplotlib/
#十分钟入门Matplotlib
import matplotlib.pyplot as plt
import numpy as np
# 简单的绘图 画出一个简单的正弦曲线
x = np.linspace(0, 2 * np.pi, 50)
#np.linspace(0, 2 * np.pi, 50) 这段代码将会生成一个包含 50 个元素的数组，这 50 个元素均匀的分布在 [0, 2pi] 的区间上
print(x)
plt.plot(x, np.sin(x)) # 如果没有第一个参数 x，图形的 x 坐标默认为数组的索引
plt.show() # 显示图形




