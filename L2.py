# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:28:11 2020

@author: CaoYang2
"""


import requests
from lxml import etree
import pandas as pd
#固定使用pandas起手式

headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36",}
#具体查网页的user-agent

file_lists = []
for i in range(1,5,1):
    url = "http://www.12365auto.com/zlts/0-0-0-0-0-0_0-0-0-0-0-0-0-{}.shtml".format(i)
    #定义i是变量，其中具体情况参照具体网页分析，format函数，i为自定义
    file_lists.append(url)
    #共有四个url，最后都加在lists里
    
data_all = []   
#建立总的数据
for page_url in file_lists:
    response = requests.get(page_url,headers=headers)
    content = response.content.decode("utf8")
    #utf8具体从审查网页中的类型
    html = etree.HTML(content)
    #将数据树形化
    
    trs = html.xpath('//tr')[1:]
    #xpath语句将tr下第2个作为第一个数据
    
    for tr in trs:
        #循环每一行然后内容放进data里
        data = {}
        #从61行循环至此变为原空字典
        number = tr.xpath('.//td/text()')[0]
        brand = tr.xpath('.//td/text()')[1]
        chexi = tr.xpath('.//td/text()')[2]
        chexing = tr.xpath('.//td/text()')[3]
        problem = tr.xpath('.//td/a/text()')[0]
        question = tr.xpath('.//td[@class="tsgztj"]//text()')
        #老师批的时候还请您告知一下为什么直接用td【6】或者'//td/span/text()'[0]都不行
        times = tr.xpath('.//td/text()')[-1]
        #老师麻烦再问一下 为什么-1可以4不可以-0-
        states = tr.xpath('.//td/em/text()')[0]
        data = {       
                "编号": number,
                "投诉品牌": brand,
                "投诉车系": chexi,
                "投诉车型": chexing,
                "问题简述": problem,
                "典型问题": question,
                "投诉时间": times,
                "投诉状态": states
                #最后一个没有逗号
                }
        data_all.append(data)
        #把data放入data——all然后清空
 
df = pd.DataFrame(data_all)
df.to_csv('l2.csv')       