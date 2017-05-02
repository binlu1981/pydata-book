# -*- coding: utf-8 -*-
#来自URL缩短服务bit.ly的1.usa.gov数据
path = 'D:/megasync/research/ngs/GitHub/pydata_book/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
print(open(path).readline())
#使用json模块逐行加载数据文件
import json
#列表推导式
#在一个打开的文件句柄上进行迭代获得一个由行组成的序列，这里为字典列表
records = [json.loads(line) for line in open(path)]
print(records[0].get('tz'))

#获悉最常出现的时区
#列表推导式取出一组时区
time_zones = [rec.get('tz') for rec in records]
print(time_zones)
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
print(counts)
#前10个最多的时区
def top_counts(count_dict, n=10):
    v_k_list = [(v,k) for k, v in count_dict.items()]
    return sorted(v_k_list,reverse=True)[:n]

print(top_counts(counts))
#使用collections.Counter类更简单
from collections import Counter
counts = Counter(time_zones)
print(counts.most_common(10))

#使用pandas计数
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
#创建表格数据结构 key是列名标签 value是值 每个字典是一行
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
print(results[:5])
#计数
print(results.value_counts()[:8])

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
#print(agg_counts[:10])
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

#相对比例图
normed_subset = count_subset.div(count_subset.sum(1),axis =0)
#print(normed_subset)
normed_subset.plot(kind='barh',stacked=True)
#pylab.show()

#http://codingpy.com/article/a-quick-intro-to-pandas/

#读取英国降雨量数据
df = pd.read_csv('D:/megasync/research/ngs/GitHub/pydata_book/ch02/uk_rain_2014.csv',header=0)
print(df.head(5))
df.columns = ['water_year','rain_octsep', 'outflow_octsep','rain_decfeb', 'outflow_decfeb', 'rain_junaug', 'outflow_junaug']
print(df.head(5))
print(len(df))
df.columns = ['water_year','rain_octsep', 'outflow_octsep','rain_decfeb', 'outflow_decfeb', 'rain_junaug', 'outflow_junaug']
print(df.head(5))
pd.options.display.float_format = '{:,.3f}'.format
print(df.describe())
#当我们提取列的时候，会得到一个 series ，而不是 dataframe 
# Creating a series of booleans based on a conditional
print(df['rain_octsep'])
print(df.rain_octsep)
#返回一个由布尔值构成的 dataframe。True 表示在十月-九月降雨量小于 1000 mm，False 表示大于等于 1000 mm
print(df.rain_octsep < 1000)
#用这些条件表达式来过滤现有的 dataframe
# Using a series of booleans to filter
print(df[df.rain_octsep < 1000])
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
print(df.groupby(df.year//10*10).max())
#groupby 会按照你选择的列对数据集进行分组。上例是按照年代分组。不过仅仅这样做并没有什么用，我们必须对其调用函数，比如 max 、 min 、mean 等等
# Grouping by multiple columns
decade_rain = df.groupby([df.year // 10 * 10, df.rain_octsep // 1000 * 1000])[['outflow_octsep','outflow_decfeb', 'outflow_junaug']].mean()
print(decade_rain)
#接下来是 unstack ，最开始可能有一些困惑，它可以将一列数据设置为列标签。最好还是看看实际的操作：
print(decade_rain.unstack(0))
#这条语句将上例中的 dataframe 转换为下面的形式。它将第 0 列，也就是 year 列设置为列的标签。
print(decade_rain.unstack(1))


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
df.to_csv('D:/megasync/research/ngs/GitHub/pydata_book/ch02/uk_rain.csv')


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




