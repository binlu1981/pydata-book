#-*- coding: utf-8 -*-
import json
path = './usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]
print(records[0])
time_zones = [x.get('tz') for x in records]

def get_counts(lst):
    dict = {}
    for x in lst:
        if x in dict:
            dict[x] += 1
        else:
            dict[x] = 1
    return dict
count_dict = get_counts(time_zones)
count_dict_top = sorted([(v,k) for k,v in count_dict.items()],reverse = True)[:10]
print(count_dict_top)

from collections import Counter
count_dict1 = Counter(time_zones)
print(count_dict1.most_common(10))

from pandas import DataFrame, Series
import pandas as pd
import numpy as np
frame = DataFrame(records)
tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_count_clean = clean_tz.value_counts()
print(tz_count_clean[:10])

import matplotlib
import pylab
tz_count_clean[:10].plot(kind = 'barh')
#pylab.show()


results = Series([x.split()[0] for x in frame.a.dropna()])

cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe.a.str.contains('Windows'),'Windows','Not windows')
print(operating_system[:5])
by_tz_os = cframe.groupby(['tz',operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
indexer = agg_counts.sum(1).argsort()
print(indexer)
count_subset = agg_counts.take(indexer)[-10:]
count_subset.plot(kind='barh',stacked = True)
normed_subset = count_subset.div(count_subset.sum(1),axis = 0)
normed_subset.plot(kind= 'barh',stacked = True)
#pylab.show()

import pandas as pd
"""
unames = ['user_id','gender','age','occupation','zip']
users = pd.read_table('./movielens/users.dat',sep='::',header=None,names=unames,engine='python')
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('./movielens/ratings.dat',sep='::',header=None,names=rnames,engine='python')
mnames = ['movie_id','title','genres']
movies = pd.read_table('./movielens/movies.dat',sep='::',header=None,names=mnames,engine='python')


print(users[:5])
data = pd.merge(pd.merge(ratings,users),movies)
print(data.ix[0])

mean_ratings = data.pivot_table(values= 'rating',index = 'title',columns = 'gender',aggfunc = 'mean')

print(mean_ratings.head())

rating_by_title = data.groupby('title').size()
active_titles = rating_by_title.index[rating_by_title > 250]
mean_ratings = mean_ratings.ix[active_titles]
top_female_rating = mean_ratings.sort_values(by='F',ascending = False)
print(top_female_rating)
"""
pieces = []
years = range(1880,2011)
for year in years:
    path = 'names/yob%d.txt' % year
    frame = pd.read_csv(path,names=['name','sex','births'])
    frame = frame.assign(year = Series(year,index=frame.index))
    pieces.append(frame)
names = pd.concat(pieces,ignore_index=True)

total_births = names.pivot_table('births',index='year',columns='sex',aggfunc='sum')
total_births1 = names.groupby(['year','sex']).births.sum().unstack()

print(total_births.head())
print(total_births1.head())

total_births.plot(title='Total births by sex and year')
#pylab.show()

def get_prop(frame, col='births'):
    frame['prop'] = frame[col]/frame[col].sum()
    return frame

names1 = names.groupby(['year','sex']).apply(get_prop)

print(names1.head())

print(np.allclose(names1.prop.sum(),1))
print(np.allclose(names1.groupby(['year','sex']).prop.sum(),1))

def get_top(group, n=1000):
    return(group.sort_values(by='births',ascending=False)[:n])

#top1000 = names1.groupby(['year','sex']).apply(get_top)
grouped = names1.groupby(['year','sex'])
top1000 = grouped.apply(get_top)

print(top1000.head())
pieces = []
for year, group in names1.groupby(['year','sex']):
    pieces.append(group.sort_values(by='births',ascending=False)[:1000])
top1000a = pd.concat(pieces)
print(top1000a.head())

boys =top1000[top1000.sex=='M']
girls = top1000a[top1000a.sex=='F']

total_birth = top1000.pivot_table('births',index='year',columns='name',aggfunc = 'sum')
print(total_birth.head())
total_birth1 = top1000a.pivot_table('births',index='year',columns='name',aggfunc = 'sum')
print(total_birth1.head())

subset = total_birth[['John','Harry','Marilyn','Mary']]
subset.plot(subplots=True)
#pylab.show()

table = top1000a.pivot_table('prop',index='year',columns='sex',aggfunc=sum)
table.plot(title= 'sum')
#pylab.show()

def get_quantile_count(group,q=0.5,col='prop'):
    return group.sort_values(col,ascending=False).prop.cumsum().searchsorted(q)+1

diversity= top1000.groupby(['year','sex']).apply(get_quantile_count).unstack('sex')
print(diversity.head())
diversity.astype(float).plot()
#pylab.show()

print(names1.head())

last_letters = names1.name.map(lambda x: x[-1])
names1 = names1.assign(last_letter=Series(last_letters,index=names1.index))
print(names1.head())

table1 = names1.pivot_table('births',index='last_letter',columns=['sex','year'],aggfunc='sum')
subtable = table1.reindex(columns=[1910,1960,2010],level='year')

print(subtable)

letter_prop = subtable/subtable.sum().astype(float)
print(letter_prop)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2,1)
letter_prop['M'].plot(kind='bar',ax=axes[0])
letter_prop['F'].plot(kind='bar',ax=axes[1])
plt.show()

letter_prop1 = table1/table1.sum().astype(float)
dny_ts = letter_prop.ix[['d','n','y'],'M'].T
print(dny_ts.head())

dny_ts.plot()
plt.show()

all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
print(mask)
lesley_like = all_names[mask]
print(lesley_like)
print(all_names[['lesl' in x.lower() for x in all_names]])

filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered)

filtered.groupby('name').births.sum()

table2 = filtered.pivot_table('births',index='year',columns='sex',aggfunc='sum')
table3 = table2.div(table2.sum(1),axis=0)
print(table3)
print(table2)
table3.plot()
pylab.show()


