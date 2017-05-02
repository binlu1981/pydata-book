# -*- coding: utf-8 -*-
path = 'D:/megasync/research/ngs/GitHub/pydata_book/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
print(open(path).readline())
#使用json模块逐行加载数据文件
import json
#列表推导式
#在一个打开的文件句柄上进行迭代获得一个由行组成的序列，这里为字典列表
records = [json.loads(line) for line in open(path)]
print(records[:5])
from pandas import DataFrame, Series
frame = DataFrame(records)
print(frame['tz'][:10])
#使用value_counts方法计数
tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])
#缺失值替换为Missing
clean_tz = frame['tz'].fillna('Missing')
#print(clean_tz == '')
#未知值（空字符）通过布尔型数组索引替换为Unknown
clean_tz[clean_tz ==''] ='Unknown'
tz_counts = clean_tz.value_counts()
print(clean_tz)
import matplotlib
import pylab
#柱状图最常出现的时区
tz_counts[:10].plot(kind='barh',rot=0)
#提取浏览器、设备等信息
print(frame['a'][1])
#浏览器信息
frame_a_list = [x.split()[0] for x in frame.a.dropna()]
results = Series(frame_a_list)
print(frame_a_list[:5])
print(results[:5])
#计数
print(results.value_counts()[:8])
import pandas as pd
import numpy as np
#进一步统计WIN用户和非Win用户的时区信息，先移除缺失数据
cframe = frame[frame.a.notnull()]
print(np.where(cframe.a.str.contains('Windows')))
#根据a值计算出各行是否是WIN
operating_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')
print(operating_system[:5])
#根据时区和最新的得到的操作系统列表对数据进行分组
by_tz_os = cframe.groupby(['tz',operating_system])
print(by_tz_os)

#通过size（类似value_counts)对分组结果进行计数,然后利用unstack对计数结果进行重塑
agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts[:10])
#选取最常出现的时区，WIN+NOT WIN,然后打印排位索引
agg_counts[:10].sum(1) 
#axis=0是按列求和 xis=1 是按行求和
indexer = agg_counts.sum(1).argsort()
print(indexer[:10])
#使用take根据排位截取最后10行
count_subset = agg_counts.take(indexer)[-10:]
#print(count_subset)
#生成堆积条形图
count_subset.plot(kind = 'barh', stacked =True)