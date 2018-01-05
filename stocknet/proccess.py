import quandl
import math
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
#import csv
dabhi = 0

quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
style.use('ggplot')
'''
    stock = input('ENTER STOCK')
    print(stock)
    time1 = input('from date:yyyy-mm-dd')
    print(time1)
    time2 = input('to date:yyyy-mm-dd')
    print(time2)
    quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
    data = quandl.get("WIKI/"+stock, start_date=time1, end_date=time2)
'''
print("mkay")
time1 = '2004-09-16'
time2 = '2018-01-01'

#time1 = time1.replace("-", "")
#time1 = time1.replace("-", "")

time1 = dt.datetime.strptime(time1, "%Y-%m-%d")
time2 = dt.datetime.strptime(time2, "%Y-%m-%d")

time2s1 = time2 + dt.timedelta(days=1)
time2s2 = time2 + dt.timedelta(days=2)
time2s3 = time2 + dt.timedelta(days=3)
time2s4 = time2 + dt.timedelta(days=4)
time2s5 = time2 + dt.timedelta(days=5)
time2s6 = time2 + dt.timedelta(days=6)
time2s7 = time2 + dt.timedelta(days=7)
time2s8 = time2 + dt.timedelta(days=8)
time2s9 = time2 + dt.timedelta(days=9)
time2s10 = time2 + dt.timedelta(days=10)
i = 0
'''
while time1 not in [time2,time2s1,time2s2,time2s3,time2s4,time2s5,time2s6,time2s7,time2s9,time2s10]:
    time1 = time1 + relativedelta(months=1)
    for index, row in data.iterrows():
        print (row[2], row[3])
print("done")
'''
data = quandl.get("WIKI/GOOGL", start_date='2001-01-01', end_date='2018-01-01')
print (data)
data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100
data = data[['Adj. Close','PCT_change','Adj. Volume']]
bigarray = np.array([])
smallarray=np.array([])
def changetoform(df1):
    df1 = df1.date()
    df1.strftime('%Y-%m-%d')
    df1 = str(df1)
    df1 = df1.replace("-", "")
    return(df1)

def ifsmall(df1):
    smallarray=np.array([])
    if df1 < -0.5:
        a = np.array([0,0,1])
        smallarray = np.append(smallarray,a)
        print(smallarray)

    elif df1 -0.5 < df1 < 0.5:
        b = np.array([0,1,0])
        smallarray = np.append(smallarray,b)
        print(smallarray)

    elif 0.5 < df1:
        c = np.array([1,0,0])
        smallarray = np.append(smallarray,c)
        print(smallarray)
    print("###########################################")
    strr = str(smallarray)
    with open('y.csv','a') as file:
        file.write(strr)
        file.write(',')
        file.close()



while time1 not in [time2,time2s1,time2s2,time2s3,time2s4,time2s5,time2s6,time2s7,time2s9,time2s10]:
    time1 = time1 + dt.timedelta(days=1) #dt.timedelta(days=1)#df1 = changetoform(df1)df1=data.loc[[df1]]print(df1)
    df1=time1
    dabhi = dabhi+1
    df1 = changetoform(df1)
    if dabhi == 10:
        strr = str(bigarray)
        with open('x.csv','a') as file:
            file.write(strr)
            file.write(',')
            file.close()
        bigarray = np.array([])
        time1 = time1 + dt.timedelta(days=1) #dt.timedelta(days=1)#df1 = changetoform(df1)df1=data.loc[[df1]]print(df1)
        df1=time1
        dabhi = 0
        df1 = changetoform(df1)
        try:
            df1=data.loc[df1,'PCT_change']
            print(df1)
            print(time1)
            ifsmall(df1)
        except:
            dabhi = dabhi-1
            print('no lol')
            continue


    else:



        try:
            df1=data.loc[df1,'PCT_change']
            print(df1)
            print(time1)


            if df1 < -0.5:
                a = np.array([0,0,1])
                bigarray = np.append(bigarray,a)
                print(bigarray)
            elif df1 -0.5 < df1 < 0.5:
                b = np.array([0,1,0])
                bigarray = np.append(bigarray,b)
                print(bigarray)
            elif 0.5 < df1:
                c = np.array([1,0,0])
                bigarray = np.append(bigarray,c)
                print(bigarray)

        except:
            print("skipped")
            dabhi = dabhi-1
        continue


print("done")
'''
#forecast_col = 'Adj. Close'
#forecast_out = int(math.ceil(0.01*len(df)))
#print("length of time set--->", ftlday)
#dp = input('how many days in advanced to predict:')
#dp = int(dp)
#forday = (1/len(data))
#print("value of forcasted days forward as a decimal of the lenghth of time in dataset ----> ", end="")
#print(forday)
'''
