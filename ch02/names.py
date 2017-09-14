# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
names1880 = pd.read_csv('./names/yob1880.txt',names=['name','sex','births'])
"""
names1880.head()
Out[22]: 
        name sex  births
0       Mary   F    7065
1       Anna   F    2604
2       Emma   F    2003
3  Elizabeth   F    1939
4     Minnie   F    1746
"""
#sex分组统计birth
print(names1880.groupby('sex').births.sum())
"""
sex
F     90993
M    110493
"""

#将所有年份数据组装到一个DataFrame里面，并加上year列，使用pandas.concat
years = range(1880,2011)
pieces = []
for year in years:
    path = './names/yob%d.txt' % year
    frame = pd.read_csv(path,names=['name','sex','births'])
    frame['year'] = year
    pieces.append(frame)
#print(pieces)
#合并所有数据，不保留read_csv返回的原始行号
names = pd.concat(pieces,ignore_index=True)
print(names.head())
"""
names.head()
Out[23]: 
        name sex  births  year
0       Mary   F    7065  1880
1       Anna   F    2604  1880
2       Emma   F    2003  1880
3  Elizabeth   F    1939  1880
4     Minnie   F    1746  1880
"""
#使用groupby或pivot_table在year和sex级别上对其进行聚合
total_births = pd.pivot_table(names, values = 'births',index = ['year'], columns= ['sex'], aggfunc = sum)
print(total_births.head())
"""
total_births.head()
Out[24]: 
sex        F       M
year                
1880   90993  110493
1881   91955  100748
1882  107851  113687
1883  112322  104632
1884  129021  114445
"""
import matplotlib.pyplot as plt
import pylab
total_births.plot(title='Total births by sex and year')
plt.show()
#插入一个prop列用于存放指定名字的婴儿出生比例
def add_prop(group):
    births = group.births.astype(float)
    group['prop'] = births/births.sum()
    return group
#按照year和sex分组，然后将prop列加到各个分组上,计算出每年份和性别组内姓名    
names = names.groupby(['year','sex']).apply(add_prop)
"""
names.groupby(['year','sex']).head()
Out[20]: 
              name sex  births  year
0             Mary   F    7065  1880
1             Anna   F    2604  1880
2             Emma   F    2003  1880
3        Elizabeth   F    1939  1880
4           Minnie   F    1746  1880
942           John   M    9655  1880
943        William   M    9533  1880
944          James   M    5927  1880
945        Charles   M    5348  1880
946         George   M    5126  1880
2000          Mary   F    6919  1881
"""
print(names[:10])
"""
names.head()
Out[26]: 
        name sex  births  year      prop
0       Mary   F    7065  1880  0.077643       
1       Anna   F    2604  1880  0.028618
2       Emma   F    2003  1880  0.022013
3  Elizabeth   F    1939  1880  0.021309
4     Minnie   F    1746  1880  0.019188

for names1880 group
sex
F     90993
M    110493

mary 7065/90993 = 0.077643
"""
pieces = []
for year, df in names.groupby(["year","sex"]):
    df['prop'] = df.births/df.births.sum()
    pieces.append(df)
names1 = pd.concat(pieces)
print(names1)
"""              name sex  births  year      prop
0             Mary   F    7065  1880  0.077643
1             Anna   F    2604  1880  0.028618
2             Emma   F    2003  1880  0.022013
3        Elizabeth   F    1939  1880  0.021309
"""
#有效性检验，总和应该足够近似于1
print(np.allclose(names.groupby(['year','sex']).prop.sum(),1))   

print(names.sort_index(by='births',ascending=False)[:10])
#print(names.groupby(['year','sex']).sort_index(by='birth',ascend=False)[:10])
#AttributeError: Cannot access callable attribute 'sort_index' of 'DataFrameGroupBy' objects, try using the 'apply' method
def get_top1000(group):
    return group.sort_values(by='births',ascending=False)[:1000]
#取出每对sex/year组合前1000个名字进行后续分析
top1000=names.groupby(['year','sex']).apply(get_top1000)
print(top1000.head())
"""
top1000.head()
Out[41]: 
                 name sex  births  year      prop
year sex                                         
1880 F   0       Mary   F    7065  1880  0.077643
         1       Anna   F    2604  1880  0.028618
         2       Emma   F    2003  1880  0.022013
         3  Elizabeth   F    1939  1880  0.021309
         4     Minnie   F    1746  1880  0.019188
"""
#print(names.groupby(['year','sex']))
#分析命名趋势
boys = top1000[top1000.sex=='M']
"""
boys.head()
Out[63]: 
                 name sex  births  year      prop
year sex                                         
1880 M   942     John   M    9655  1880  0.087381
         943  William   M    9533  1880  0.086277
         944    James   M    5927  1880  0.053641
         945  Charles   M    5348  1880  0.048401
         946   George   M    5126  1880  0.046392
"""
girls = top1000[top1000.sex == 'F']
#制作时间序列图表，每年叫做john,harry,mary的婴儿出生数，
#year和name统计的总出生数透视表
total_births = top1000.pivot_table('births',index='year',columns='name',aggfunc=sum)
print(total_births.head())
"""
total_births.head()
Out[39]: 
name  Aaden  Aaliyah  Aarav  Aaron  Aarush  Ab  Abagail  Abb  Abbey  Abbie  \
year                                                                         
1880    NaN      NaN    NaN  102.0     NaN NaN      NaN  NaN    NaN   71.0   
1881    NaN      NaN    NaN   94.0     NaN NaN      NaN  NaN    NaN   81.0   
1882    NaN      NaN    NaN   85.0     NaN NaN      NaN  NaN    NaN   80.0   
1883    NaN      NaN    NaN  105.0     NaN NaN      NaN  NaN    NaN   79.0   
1884    NaN      NaN    NaN   97.0     NaN NaN      NaN  NaN    NaN   98.0   
"""
subset = total_births[['John','Harry','Mary']]
subset.plot(subplots=True,figsize=(12,10),grid=False,title='Number of births per year')
plt.show()

#评估命名多样性的增长
table = top1000.pivot_table('prop',index='year',columns='sex',aggfunc='sum')
#方法1：前1000项的比例降低
table.plot(title='sum of table10.prop by year and sex',yticks=np.linspace(0,1.2,13),xticks=range(1880,2020,10))
plt.show()
"""
table.head()
Out[40]: 
sex          F         M
year                    
1880  0.091954  0.908046
1881  0.106796  0.893204
1882  0.065693  0.934307
1883  0.053030  0.946970
1884  0.107143  0.892857
"""
#方法2：计算占总出生人数前50%的不同名字的数量
df = boys[boys.year==2010]
print(df[:10])
"""
df[:10]
Out[64]: 
                       name sex  births  year      prop
year sex                                               
2010 M   1676644      Jacob   M   21875  2010  0.011523
         1676645      Ethan   M   17866  2010  0.009411
         1676646    Michael   M   17133  2010  0.009025
         1676647     Jayden   M   17030  2010  0.008971
         1676648    William   M   16870  2010  0.008887
         1676649  Alexander   M   16634  2010  0.008762
"""
#先对prop降序排列，计算prop的累积和cumsum
prop_cumsum = df.sort_values(by='prop',ascending=False).prop.cumsum()
print(prop_cumsum[:10])
"""
prop_cumsum[:10]
Out[65]: 
year  sex         
2010  M    1676644    0.011523
           1676645    0.020934
           1676646    0.029959
           1676647    0.038930
           1676648    0.047817
           1676649    0.056579
           1676650    0.065155
           1676651    0.073414
           1676652    0.081528
           1676653    0.089621
Name: prop, dtype: float64
"""
#通过searchsorted方法找出50%所在的位置
print(prop_cumsum.searchsorted(0.5))
"""
prop_cumsum.searchsorted(0.5)
Out[66]: array([116])
"""

def get_quantitle_count(group,q=0.5):
    group = group.sort_values(by='prop',ascending=False)
    return group.prop.cumsum().searchsorted(q)+1
#对year/sex组合执行计算
diversity = top1000.groupby(['year','sex']).apply(get_quantitle_count)
print(diversity.head())
"""
diversity.head()
Out[68]: 
year  sex
1880  F      [38]
      M      [14]
1881  F      [38]
      M      [14]
1882  F      [38]
"""
#按sex分成两列，取浮点数
diversity = diversity.unstack('sex').astype('float')
print(diversity.head())
"""
diversity.head()
Out[70]: 
sex      F     M
year            
1880  38.0  14.0
1881  38.0  14.0
1882  38.0  15.0
1883  39.0  15.0
1884  39.0  16.0
"""
diversity.plot()
plt.show()

#最后一个字母的变革
names['last_letters']=names.name.map(lambda x: x[-1])
print(names.head())
"""
names.head()
Out[75]: 
        name sex  births  year      prop last_letters
0       Mary   F    7065  1880  0.077643            y
1       Anna   F    2604  1880  0.028618            a
2       Emma   F    2003  1880  0.022013            a
3  Elizabeth   F    1939  1880  0.021309            h
4     Minnie   F    1746  1880  0.019188            e
"""
#先将全部数据在年代、性别和末字母上进行聚合
table = names.pivot_table('births',index='last_letters',columns=['sex','year'],aggfunc=sum)
print(table.head())
"""
table.head()
Out[77]: 
sex                 F                                                        \
year             1880     1881     1882     1883     1884     1885     1886   
last_letters                                                                  
a             31446.0  31581.0  36536.0  38330.0  43680.0  45408.0  49100.0   
b                 NaN      NaN      NaN      NaN      NaN      NaN      NaN   
c                 NaN      NaN      5.0      5.0      NaN      NaN      NaN   
d               609.0    607.0    734.0    810.0    916.0    862.0   1007.0   
e             33378.0  34080.0  40399.0  41914.0  48089.0  49616.0  53884.0   
"""
#选出具有代表性的三年，并输出前面几行
subtable = table.reindex(columns=[1910,1960,2010],level='year')
print(subtable.head())
"""
subtable.head()
Out[78]: 
sex                  F                            M                    
year              1910      1960      2010     1910      1960      2010
last_letters                                                           
a             108376.0  691247.0  670605.0    977.0    5204.0   28438.0
b                  NaN     694.0     450.0    411.0    3912.0   38859.0
c                  5.0      49.0     946.0    482.0   15476.0   23125.0
d               6750.0    3729.0    2607.0  22111.0  262112.0   44398.0
e             133569.0  435013.0  313833.0  28655.0  178823.0  129012.0
"""
#按总出生数对表格进行规范化处理
print(subtable.sum())
"""
subtable.sum()
Out[79]: 
sex  year
F    1910     396416.0
     1960    2022062.0
     2010    1759010.0
M    1910     194198.0
     1960    2132588.0
     2010    1898382.0
"""
#计算各性别、各末字母占总出生人数的比例
letter_prop = subtable/subtable.sum()
print(letter_prop.head())
"""
letter_prop.head()
Out[82]: 
sex                  F                             M                    
year              1910      1960      2010      1910      1960      2010
last_letters                                                            
a             0.273390  0.341853  0.381240  0.005031  0.002440  0.014980
b                  NaN  0.000343  0.000256  0.002116  0.001834  0.020470
c             0.000013  0.000024  0.000538  0.002482  0.007257  0.012181
d             0.017028  0.001844  0.001482  0.113858  0.122908  0.023387
e             0.336941  0.215133  0.178415  0.147556  0.083853  0.067959
"""
#生成各年度，各性别的条形图，rot横轴标注的角度
import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,1,figsize=(10,8))
letter_prop['M'].plot(kind='bar',rot=0,ax=axes[1],title='Male')
letter_prop['F'].plot(kind='bar',rot=90,ax=axes[0],title='Female')
plt.show()
#回到table，在男孩名字末字母选取几个字母，进行转置以便将各个列做成一个时间序列
letter_prop = table/table.sum()
"""
letter_prop.head()
Out[82]: 
sex                  F                             M                    
year              1910      1960      2010      1910      1960      2010
last_letters                                                            
a             0.273390  0.341853  0.381240  0.005031  0.002440  0.014980
b                  NaN  0.000343  0.000256  0.002116  0.001834  0.020470
c             0.000013  0.000024  0.000538  0.002482  0.007257  0.012181
d             0.017028  0.001844  0.001482  0.113858  0.122908  0.023387
e             0.336941  0.215133  0.178415  0.147556  0.083853  0.067959
"""
dny_ts = letter_prop.ix[['d','n','y'],'M'].T
print(dny_ts.head())
"""
dny_ts.head()
Out[84]: 
last_letters         d         n         y
year                                      
1880          0.083055  0.153213  0.075760
1881          0.083247  0.153214  0.077451
1882          0.085340  0.149560  0.077537
1883          0.084066  0.151646  0.079144
1884          0.086120  0.149915  0.080405
"""
dny_ts.plot()
plt.show()
#变成女孩名字的男孩名字（以及相反的情况)
#回到top1000数据集，找出其中以lesl开头的一组名字


#利用这个结果过滤掉其他的名字，并按名字分组计算出生数以查看相对频率
all_names = top1000.name.unique()
print(all_names[:10])
mask=np.array(['lesl' in x.lower() for x in all_names])
"""
mask
Out[90]: array([False, False, False, ..., False, False, False], dtype=bool)
"""
lesley_like = all_names[mask]
print(lesley_like)
"""
lesley_like
Out[89]: array(['Leslie', 'Lesley', 'Leslee', 'Lesli', 'Lesly'], dtype=object)
"""
filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered.head())
"""
filtered.head()
Out[91]: 
                 name sex  births  year      prop
year sex                                         
1880 F   654   Leslie   F       8  1880  0.000088
     M   1108  Leslie   M      79  1880  0.000715
1881 F   2523  Leslie   F      11  1881  0.000120
     M   3072  Leslie   M      92  1881  0.000913
1882 F   4593  Leslie   F       9  1882  0.000083
"""
print(filtered.groupby('name').births.sum())
"""
filtered.groupby('name').births.sum()
Out[92]: 
name
Leslee      1082
Lesley     35022
Lesli        929
Leslie    370429
Lesly      10067
Name: births, dtype: int64
"""
#按性别和年度进行聚合，并按照年度进行规范化处理
table = filtered.pivot_table('births',index='year',columns='sex',aggfunc=sum)
print(table.head())
"""
table.head()
Out[94]: 
sex      F      M
year             
1880   8.0   79.0
1881  11.0   92.0
1882   9.0  128.0
1883   7.0  125.0
1884  15.0  125.0
"""
table = table.div(table.sum(1),axis=0)
print(table.head())
"""
print(table.head())
sex          F         M
year                    
1880  0.091954  0.908046
1881  0.106796  0.893204
1882  0.065693  0.934307
1883  0.053030  0.946970
1884  0.107143  0.892857
"""
table.plot(style={'M':'k-','F':'k--'})
plt.show()

