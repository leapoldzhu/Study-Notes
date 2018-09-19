#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-15 22:07:49
# @Author  : Leo Zhu (leapold_zhu@yeah.net)
# @Link    : ${link}
# @Version : $Id$

import pandas as pd
import numpy as np

# Class 1
# Basic attributes
s = pd.Series([1,3,4,np.nan,44,1])
print(s)

dates = pd.date_range('20180915', periods = 6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = ['a', 'b', 'c', 'd'])
print(df)

df = pd.DataFrame(np.random.randn(12).reshape((3,4)))
print(df)

df2 = pd.DataFrame({'A' : 1.,
                       'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4,dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'foo'})
print(df2)
print(df2.dtypes)
# ouput all label for rows
print(df2.index)
# output all label for columns
print(df2.columns)
# output all values
print(df2.values)
print(df2.describe())
print(df2.T)
print(df2.sort_index(axis = 1, ascending = False))
print(df2.sort_values(by = 'E'))

# Class 2
# Choose data
print("====================")
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['A', 'B', 'C', 'D'])

print(df['A'], '\n', df.A)
print('\n', df[0:3], '\n', df['20130102':'20130104'])

# select by label: loc
print('\n', df.loc['20130102'])
print('\n', df.loc[:,['A','B']])
print('\n', df.loc['20130102', ['A','B']])

# select by position: iloc
print('\n', df.iloc[3])
print('\n', df.iloc[3, 1])
print('\n', df.iloc[3:5,0:2])
print('\n', df.iloc[[1,2,4],[0,2]])

# mixed selection: ix
print(df.ix[:3, ['A', 'C']])
# Boolean indexing
print(df[df.A > 0])

# Class 3
# Set Value
print("====================")
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['A', 'B', 'C', 'D'])

df.iloc[2,2] = 1111
df.loc['2013-01-03', 'D'] = 2222
df[df.A>0] = 0
df.A[df.A>0] = 0
df['F'] = np.nan
# index should idencle with table
df['G']  = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130101', periods=6))
print(df)

# Class 4
# Deal with Nan value
print("====================")
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A', 'B', 'C', 'D'])

df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df.dropna(axis=0, how='any'))   # how={'any', 'all'}
print(df.fillna(value=0))
print(pd.isnull(df))

# Class 5
# Read/Export data
print("====================")
data = pd.read_csv('student.csv')
print(data)

# save to
data.to_pickle('student.pickle')

# Class 6
# Concat different data
# concatenating
# ignore index
print("====================")
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
# ignore_index, relabel data
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# join, ('inner', 'outer')
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d', 'e'], index=[2,3,4])
res = pd.concat([df1, df2], axis=1, join='outer')
res = pd.concat([df1, df2], axis=1, join='inner')

# join_axes
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])

# append
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d', 'e'], index=[2,3,4])
res = df1.append(df2, ignore_index=True)
res = df1.append([df2, df3])

s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
res = df1.append(s1, ignore_index=True)

print(res)

# Class 7
# merging two df by key/keys. (may be used in database)
# simple example
print("====================")
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                                  'A': ['A0', 'A1', 'A2', 'A3'],
                                  'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                                    'C': ['C0', 'C1', 'C2', 'C3'],
                                    'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)
res = pd.merge(left, right, on='key')
print(res)

# consider two keys
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                             'key2': ['K0', 'K1', 'K0', 'K1'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                              'key2': ['K0', 'K0', 'K0', 'K0'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)
res = pd.merge(left, right, on=['key1', 'key2'], how='inner')  # default for how='inner'
# how = ['left', 'right', 'outer', 'inner']
res = pd.merge(left, right, on=['key1', 'key2'], how='left')
print(res)

# indicator
df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
print(df1)
print(df2)
res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
# give the indicator a custom name
res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')


# merged by index
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                                  'B': ['B0', 'B1', 'B2']},
                                  index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                                     'D': ['D0', 'D2', 'D3']},
                                      index=['K0', 'K2', 'K3'])
print(left)
print(right)
# left_index and right_index
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
res = pd.merge(left, right, left_index=True, right_index=True, how='inner')

# handle overlapping
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
res = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
print(res)

# join function in pandas is similar with merge. If know merge, you will understand join

# Class 8
# Visualization
print("====================")
import matplotlib.pyplot as plt

# plot data

# Series
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()
##data.plot()

# DataFrame
data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))
data = data.cumsum()
# plot methods:
# 'bar', 'hist', 'box', 'kde', 'area', scatter', hexbin', 'pie'
ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label="Class 1")
data.plot.scatter(x='A', y='C', color='LightGreen', label='Class 2', ax=ax)

plt.show()
