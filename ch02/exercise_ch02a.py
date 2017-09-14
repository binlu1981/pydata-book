#-*- coding: utf-8 -*-
path = './usagov_bitly_data2012-03-16-1331923249.txt'

import json

records = [json.loads(line) for line in open(path)]
print(records[0])
time_zone = [x.get('tz') for x in records]
def get_count(lst):
    count_lst = {}
    for x in lst:
        if x in count_lst:
            count_lst[x] += 1
        else:
            count_lst[x] = 1
    return count_lst
            
tz_count = get_count(time_zone)
vk = [(v, k) for k, v in tz_count.items()]
print(sorted(vk,reverse = True)[:10])
from collections import Counter
tz_count1 = Counter(time_zone)
print(tz_count1.most_common(10))
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

df = DataFrame(records)
tz_count2 = df['tz'].value_counts()
import matplotlib
import pylab
clean_tz = df['tz'].fillna("Missing")
clean_tz[clean_tz == ''] = 'Unknown'
tz_count3 = clean_tz.value_counts()
tz_count3[:10].plot(kind = 'barh',rot = 0)
#pylab.show()

explorer = [x.split()[0] for x in df.a.dropna()]
print(explorer[:10])
print(Series(explorer[:10]))
#print(explorer.value_counts()[:9])
#/Users/binlu/MEGA/research/ngs/GitHub/pydata_book/ch02/exercise_ch02.py
print(Series(explorer).value_counts()[:9])

cdf = df[df.a.notnull()]
system = np.where(cdf.a.str.contains('Windows|windows'),"Windows","Not Windows")
print(Series(system)[:5])

#by_tz_os = cdf.groupby(['tz',system]).value_counts()
#AttributeError: 'DataFrameGroupBy' object has no attribute 'value_couts' 
by_tz_os = cdf.groupby(['tz',system]).size().unstack().fillna(0)
print(by_tz_os[:10])
indexer=by_tz_os.sum(1).argsort()
print(indexer)
count_subset = by_tz_os.take(indexer)
print(count_subset)            
count_subset[-10:].plot(kind='barh',stacked=False)
pylab.show()
normed_subset = count_subset.div(count_subset.sum(1),axis=0)
normed_subset[-10:].plot(kind='barh',stacked=True)
pylab.show()

"""
unames = ['user_id','gender','age','occupation','zip']
users = pd.read_table('./movielens/users.dat',sep='::',header=None,names=unames,engine='python')
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('./movielens/ratings.dat',sep='::',header=None,names=rnames,engine='python')
mnames = ['movie_id','title','genres']
movies = pd.read_table('./movielens/movies.dat',sep='::',header=None,names=mnames,engine='python')

df = pd.merge(pd.merge(ratings,users),movies)
print(df.ix[0])
print(df.head())
by_sex = df.pivot_table('rating',index='title',columns='gender',aggfunc='mean')
#print(by_sex.sum(1).argsort())

rating_by_title = df.groupby('title').size()
active_title = rating_by_title.index[rating_by_title > 250]
by_sex = by_sex.ix[active_title]
print(by_sex)

top_female_rating = by_sex.sort_values(by='F',ascending=False)
rating_std_by_title = df.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.ix[active_title]

print(rating_std_by_title.head())
"""


years = range(1880,2011)
pieces =[]
for year in years:
    path='./names/yob%d.txt' % year
    frame = pd.read_csv(path,names=['name','sex','births'])
    frame = frame.assign(year=Series(year,index=frame.index))
    pieces.append(frame)
names= pd.concat(pieces,ignore_index=True)

total_births = names.pivot_table(values = 'births',index='year',columns='sex',aggfunc='sum')

print(total_births.head())
total_births.plot(kind = 'line')
pylab.show()

def add_prop(df):
    births = df.births.astype(float)
    df['prop'] = births/births.sum(0)
    return df
    
names1 = names.groupby(['year','sex']).apply(add_prop)
print(names.head()) 
print(np.allclose(names1.groupby(['year','sex']).prop.sum(),1)) 

def get_top1000(df):
    return df.sort_values(by='births',ascending = False)[:1000]
    
top1000 = names1.groupby(['year','sex']).apply(get_top1000)
print(top1000.head())

pieces1 =[ ]
for year, group in names1.groupby(['year','sex']):
    pieces1.append(group.sort_values(by='births',ascending=False)[:1000])
top1000a = pd.concat(pieces1,ignore_index=True)
print(top1000a[:10])

boys = top1000[top1000.sex == 'M']
grils = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births',index='year',columns='name',aggfunc='sum')
subset = total_births[['John','Harry','Mary','Marilyn']]
subset.plot(subplots=True)
pylab.show()
table = top1000.pivot_table(values='prop',index='year',columns='sex',aggfunc='sum')
table.plot()
pylab.show()
"""
def get_half_position(df,position=0.5):
    df['prop_sum']=df.sort_values(by='prop',ascending=False).prop.cumsum()
    idx = df['prop_sum'].searchsorted(position)
    return idx+1
diversity = top1000.groupby(['year','sex']).apply(get_half_position).unstack('sex')
print(diversity.head())
diversity.astype(float).plot()
pylab.show()


get_last_letter = lambda x: x[-1]
names['last_letters'] = names.name.map(get_last_letter)
table1 = names.pivot_table('births',index='last_letters',columns=['sex','year'],aggfunc=sum)

subtable1 = table1.reindex(columns=[1910,1960,2010],level='year')
print(subtable1.head())
letter_prop = subtable1 / subtable1.sum().astype(float)
print(letter_prop.head())
import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,1,figsize=(10,8)) #draw 轮廓
letter_prop['M'].plot(kind='bar',rot=0,ax=axes[0],title='male')
letter_prop['F'].plot(kind='bar',rot=0,ax=axes[1],title='female')
plt.show()

letter_prop1 = table1/ table1.sum(0).astype(float)
dny_ts = letter_prop1.ix[['d','n','y'],'M'].T
print(dny_ts.head()) 
dny_ts.plot()   
pylab.show()


all_names = top1000.name.unique()
mask = ['lesl' in x.lower() for x in all_names]
#lesley_like=all_names[mask]
#print(lesley_like)
#['Mary' 'Mary' 'Mary' ..., 'Mary' 'Mary' 'Mary']
lesley_like=all_names[np.array(mask)]
filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered.groupby('name').births.sum())
table2 = filtered.pivot_table('births',index='year',columns='sex',aggfunc='sum')
table3 = table2.div(table2.sum(1),axis=0)
print(table3)
table3.plot()
pylab.show()


import random
position = 0
walk = [position]
steps =1000
for i in range(steps):
    step = 1 if random.randint(0,1) else -1
    position +=step
    walk.append(position)
Series(walk).plot()
pylab.show()
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0,2,size=(nwalks,nsteps))
steps = np.where(draws > 0, 1,-1)
walks =steps.cumsum(1)
hit30 = (np.abs(walks)>=30).any(1)
hit30.sum()
crossing_times = (np.abs(walks[hit30])).argmax(1)
print(crossing_times.mean())

"""




    
    
