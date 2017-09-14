import json
path = './usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]
time_zone = [x.get('tz') for x in records]

def get_counts(lst):
    dict = {}
    for x in lst:
        if x in dict:
            dict[x] += 1
        else:
            dict[x] = 1
    return dict

from collections import Counter
count10 = Counter(time_zone).most_common(10)


from pandas import DataFrame, Series
import pandas as pd
import numpy as np

frame = DataFrame(records)
tz_counts = frame.tz.value_counts()

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts1 = clean_tz.value_counts()

resuts = Series([x.split()[0] for x in frame.a.dropna()])
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Window'),'Windows','Not Windows')
by_tz_os = cframe.groupby(['tz',operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts[:10])

indexer = agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer)[-10:]
normed_subset = count_subset.div(count_subset.sum(1),axis=0)
print(normed_subset.head())

pieces = []
years = range(1880,2011)
for year in years:
    path = 'names/yob%d.txt' % year
    frame = pd.read_csv(path,names=['name','sex','births'])
    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces,ignore_index=True)
total_births = names.pivot_table('births',index = 'year',columns = 'sex',aggfunc='sum')
def get_prop(df,col='births'):
    df['prop'] = df[col]/df[col].sum(0)
    return df

names1 = names.groupby(['year','sex']).apply(get_prop)
print(names1)
print(np.allclose(names1.groupby(['year','sex']).prop.sum(),1))

def get_top(df,n = 1000):
    return df.sort_values('births',ascending=False)[:n]

top1000 = names1.groupby(['year','sex']).apply(get_top)

print(top1000.head())
pieces = []
for year, df in names1.groupby(['year','sex']):
    pieces.append(df.sort_values(by='births',ascending=False)[:1000])
top1000a = pd.concat(pieces,ignore_index=True)
print(top1000a.head())

total_births1 = top1000.pivot_table('births',index='year',columns='name',aggfunc='sum')
subset = total_births1[['John','Harry']]

def get_50(df,p=0.5):
    return df.sort_values('prop',ascending = False).prop.cumsum().searchsorted(p)+1

diversity = top1000.groupby(['year','sex']).apply(get_50).unstack('sex')
print(diversity)

last_letter = names1.name.map(lambda x: x[-1])
last_letter.name = 'last_letter'
table = names1.piovt('births',index=last_letter,columns=['sex','year'],aggfunc='sum')