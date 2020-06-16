# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 08:56:49 2020
@author: FanYangyi
"""
import pandas as pd
#对汽车质量数据进行统计
result = pd.read_csv('car_complain.csv')
#drop('problem', 1) -> 删除列
#join -> 合并列
#result.problem.str.get_dummies(',') -> 将'problem'列作为str，通过','区分，进行one-hot编码（离散特征有多少取值，就用多少维来表示这个特征）
result = result.drop('problem', 1).join(result.problem.str.get_dummies(','))
tags = result.columns[7:]#问题种类标签
#groupby后括号内标签将作为列，非括号内标签作为下一级汇总条件
df= result.groupby(['brand'])['id'].agg(['count'])#对不同品牌的问题数量进行计数汇总
df2= result.groupby(['brand'])[tags].agg(['sum'])#对不同品牌的各类问题数量进行加和汇总
#将两个dataframe进行合并，按照left侧的brand信息合并
df2 = df.merge(df2, left_index=True, right_index=True, how='left')
#通过reset_index将DataFrameGroupBy => DataFrame
df2.reset_index(inplace=True)
#按照count列的值，非升序（即降序）排列
df2= df2.sort_values('count', ascending=False)
df2.to_csv('temp.csv', index=False)#不需要index保存为csv
query = ('A12', 'sum')#按照此问题进行排序
df3 = df2.sort_values(query, ascending=False)