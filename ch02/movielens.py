# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
unames = ['user_id','gender','age','occupation','zip']
users = pd.read_table('./movielens/users.dat',sep='::',header=None,names=unames)
#print(users.head())
"""
   user_id gender  age  occupation    zip
0        1      F    1          10  48067
1        2      M   56          16  70072
2        3      M   25          15  55117
3        4      M   45           7  02460
4        5      M   25          20  55455
"""
rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_table('./movielens/ratings.dat',sep='::',header=None,names=rnames)
#print(ratings.head())
"""
   user_id  movie_id  rating  timestamp
0        1      1193       5  978300760
1        1       661       3  978302109
2        1       914       3  978301968
3        1      3408       4  978300275
4        1      2355       5  978824291
"""
mnames = ['movie_id','title','genres']
movies = pd.read_table('./movielens/movies.dat',sep='::',header=None,names=mnames)
#print(movies[:5])
"""
   movie_id                               title                        genres
0         1                    Toy Story (1995)   Animation|Children's|Comedy
1         2                      Jumanji (1995)  Adventure|Children's|Fantasy
2         3             Grumpier Old Men (1995)                Comedy|Romance
3         4            Waiting to Exhale (1995)                  Comedy|Drama
4         5  Father of the Bride Part II (1995)                        Comedy
"""
#根据性别和年龄计算某部电影的平均得分
#合并所有数据
data = pd.merge(pd.merge(ratings, users),movies)
print(data.head())
#第一行
print(data.ix[0])
#根据任意个用户或电影属性对评分数据进行聚合操作
#这里按性别计算，使用pivot_table
mean_ratings = data.pivot_table(values='rating',index=['title'],columns=['gender'],aggfunc=np.mean)
print(mean_ratings[:5])
"""
gender                            F     M
title                                    
$1,000,000 Duck (1971)        3.375 2.762
'Night Mother (1986)          3.389 3.353
'Til There Was You (1997)     2.676 2.733
'burbs, The (1989)            2.793 2.962
...And Justice for All (1979) 3.829 3.689
"""
#新的DataFrame, 内容为电影平均分，行标为电影名称，列标为性别
#下一步，过滤掉评分数据不足250条的电影，需要先对title进行分组，然后利用size()得到一个含有个电影分组大小的Series对象
ratings_by_title = data.groupby('title').size()
print(ratings_by_title[:5])
active_titles = ratings_by_title[ratings_by_title >= 250]
print(active_titles[0:5])
"""
title
'burbs, The (1989)                   303
10 Things I Hate About You (1999)    700
101 Dalmatians (1961)                565
101 Dalmatians (1996)                364
12 Angry Men (1957)                  616
"""
active_titles_index = ratings_by_title.index[ratings_by_title >= 250]
#mean_ratings = mean_ratings.ix[active_titles]
"""
active_titles_index[:10]
Out[208]: 
Index([u''burbs, The (1989)', u'10 Things I Hate About You (1999)',
       u'101 Dalmatians (1961)', u'101 Dalmatians (1996)',
       u'12 Angry Men (1957)', u'13th Warrior, The (1999)',
       u'2 Days in the Valley (1996)', u'20,000 Leagues Under the Sea (1954)',
       u'2001: A Space Odyssey (1968)', u'2010 (1984)'],
      dtype='object', name=u'title')
"""
#得到一个列表
print(active_titles.index)
mean_ratings = mean_ratings.ix[active_titles.index]
print(mean_ratings[0:5])
"""
gender                                F     M
title                                        
'burbs, The (1989)                2.793 2.962
10 Things I Hate About You (1999) 3.647 3.312
101 Dalmatians (1961)             3.791 3.500
101 Dalmatians (1996)             3.240 2.911
12 Angry Men (1957)               4.184 4.328
"""
#女性观众最喜欢的电影，降序排列
top_female_ratings = mean_ratings.sort_values(by = 'F', ascending = False)
print(top_female_ratings[:5])
"""
gender                                                 F     M
title                                                         
Close Shave, A (1995)                              4.644 4.474
Wrong Trousers, The (1993)                         4.588 4.478
Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)      4.573 4.465
Wallace & Gromit: The Best of Aardman Animation... 4.563 4.385
Schindler's List (1993)                            4.563 4.491
"""

#计算评分分歧
#办法1加一列存放平均分差，然后排序,分歧最大且女性观众更喜欢的电影
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
print(sorted_by_diff[::-1][:5])

#只找出分歧最大的电影（不考虑性别），可计算得分数据的方差或标准差
#根据电影分组的得分数据标准差
rating_std_by_title = data.groupby('title')['rating'].std()
"""
Zeus and Roxanne (1997)                           1.122884
eXistenZ (1999)                                   1.178568
Name: rating, dtype: float64
"""
#根据active_titles 进行过滤
rating_std_by_title = rating_std_by_title.ix[active_titles]
#根据值对Series进行降序排列
print(rating_std_by_title.order(ascending=False)[:10])
"""
Black Sunday (La Maschera Del Demonio) (1960)               1.892969
Ballad of Narayama, The (Narayama Bushiko) (1958)           1.767767
"""

