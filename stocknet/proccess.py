import quandl
import math
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from dateutil.relativedelta import relativedelta

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
time1 = '2001-01-01'
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

def changetoform(yeet):
    yeet = yeet.date()
    yeet.strftime('%m/%d/%Y')
    print (yeet)


while time1 not in [time2,time2s1,time2s2,time2s3,time2s4,time2s5,time2s6,time2s7,time2s9,time2s10]:
    time1 = time1 + dt.timedelta(days=1)#relativedelta(day=1)
    df1=data.loc[time1]
    print(df1)
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
