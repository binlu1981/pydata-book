#-*- coding: utf-8 -*-
path = './usagov_bitly_data2012-03-16-1331923249.txt'
line = open(path).readline()
import json
records = [json.loads(line) for line in open(path)]
print(records[0].get('tz'))
time_zones = [record.get('tz') for record in records]
def get_counts(tz_list):
    counts = {}
    for tz in tz_list:
        if tz in counts:
            counts[tz] += 1
        else:
            counts[tz] = 1
    return counts
    
counts = get_counts(time_zones)
print(counts['Asia/Seoul'])

def top_counts(tz_count_dict,n=10):
    v_k_pair = [(v,k) for k,v in tz_count_dict.items()]
    v_k_pair.sort()
    return v_k_pair[-n:]
print(sorted([(v,k) for k,v in counts.items()])[-10:])

from collections import Counter
counts = Counter(time_zones)
print(counts.most_common(10))

from pandas import DataFrame,Series
import pandas as pd
import numpy as np

frame = DataFrame(records)
print(frame['tz'][0])
print(frame[:3])
tz_counts = frame['tz'].value_counts()
print(tz_counts[:3])
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz ==''] = 'Unknown'
print(clean_tz[:3])
tz_counts = clean_tz.value_counts()
print(tz_counts[:3])
tz_counts[:3].plot(kind='barh',rot=0)
import pylab
pylab.show()

results = [x.split()[0] for x in frame.a.dropna()]
print(results[:5])
results = Series([x.split()[0] for x in frame.a.dropna()])
print(results[:5])
print(results.value_counts()[:5])
print(frame.a[:20].notnull())
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe.a.str.contains('Windows'),'Windows','Not Windows')
print(Series(operating_system[:3]))
by_tz_os = cframe.groupby(['tz',operating_system])
print(by_tz_os)
print(by_tz_os.size()[:10])
agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts[:5])
indexer = agg_counts.sum(1).argsort()
print(indexer[:10])
count_subset = agg_counts.take(indexer)[-10:]
count_subset.plot(kind='barh',stacked=True)
normed_subset = count_subset.div(count_subset.sum(1),axis=0)
normed_subset.plot(kind='barh',stacked=True)
pylab.show()

import pandas as pd
unames = ['user_id','gender','age','occupation','zip']
users = pd.read_table('./movielens/users.dat',sep = '::',header = None,names=unames)
print(users[:5])
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('./movielens/ratings.dat',sep='::',header=None,names=rnames)
mnames=['movie_id','title','genres']
movies=pd.read_table('./movielens/movies.dat',sep='::',header=None,names=mnames)
print(users[:5])
data = pd.merge(movies,pd.merge(ratings,users))
print(data.ix[0])
mean_ratings = data.pivot_table(values = 'rating',index = 'title',columns = 'gender',aggfunc='mean')
print(mean_ratings[:5])
rating_by_title = data.groupby('title').size()
active_titles = rating_by_title.index[rating_by_title >= 250]
mean_ratings = mean_ratings.ix[active_titles]
top_female_ratings = mean_ratings.sort_index(by='F',ascending =False)
print(top_female_ratings[:10])
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
print(sorted_by_diff[:5])
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.ix[active_titles]
print(rating_std_by_title.order(ascending = False)[:8])

names1880 = pd.read_csv('names/yob1880.txt',names = ['name','sex','birth'])
print(names1880[:10])
print(names1880.groupby('name').birth.sum())

pieces = []
for year in range(1880,2011):
    path = 'names/yob%d.txt' % year
    frame = pd.read_csv(path,names = ['name','sex','birth'])
    frame['year'] = year
    pieces.append(frame)
    
names = pd.concat(pieces,ignore_index=True)
print(names[:10])

total_births = names.pivot_table('birth',index = 'year',columns = 'sex',aggfunc=sum)
print(total_births.tail())
total_births.plot(title = 'Total births by sex and year')
pylab.show()

def add_prop(group):
    birth = group.birth.astype(float)
    group['prop'] = birth/birth.sum()
    return group
    
names = names.groupby(['year','sex']).apply(add_prop)
print(names[:10])
print(np.allclose(names.groupby(['year','sex']).prop.sum(),1))   

print(names.sort_index(by='birth',ascending=False)[:10])
#print(names.groupby(['year','sex']).sort_index(by='birth',ascend=False)[:10])
#AttributeError: Cannot access callable attribute 'sort_index' of 'DataFrameGroupBy' objects, try using the 'apply' method
def top10(group):
    return group.sort_index(by='birth',ascending=False)[:10]
top10=names.groupby(['year','sex']).apply(top10)
print(top10)

print(names.groupby(['year','sex']))

total_births = top10.pivot_table('births',index='year',column='name',aggfucn='sum')
