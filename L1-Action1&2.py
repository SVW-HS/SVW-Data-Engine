# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:31:18 2020
@author: FanYangyi
"""
#求2+4+6+8+...+100的求和
sum1 = 0
for n in range(2,102,2):
    sum1 = sum1 + n
print(sum1)

#统计全班的成绩
import pandas as pd
#通过字典导入DataFrame
data = {'姓名':['张飞','关羽','刘备','典韦','许褚'],
        '语文':[68, 95, 98, 90, 80],
        '数学':[65, 76, 86, 88, 90],
        '英语':[30, 98, 88, 77, 90]}
df = pd.DataFrame(data)
#2-4列进行汇总计算
mean1 = df.iloc[0:,1:4].mean(axis=0)
min1 = df.iloc[0:,1:4].min(axis=0)
max1 = df.iloc[0:,1:4].max(axis=0)
var1 = df.iloc[0:,1:4].var(axis=0)
std1 = df.iloc[0:,1:4].std(axis=0)
#拼接结果（横向），增加名称
pd.concat([mean1,min1,max1,var1,std1],axis=1,keys=['mean','min','max','var','std'])
#增加总分列，横向加和
df['总分'] = df.sum(axis=1)
#按总分列降序排列
df.sort_values(ascending = False,by=['总分'])
#也可以用df.sort('总分',ascending = False)