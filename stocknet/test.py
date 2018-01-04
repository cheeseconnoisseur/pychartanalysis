import quandl
import math
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from dateutil.relativedelta import relativedelta
import numpy as np
'''
data = quandl.get("WIKI/GOOGL", start_date='2001-01-01', end_date='2018-01-01')
print (data)
data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100
data = data[['Adj. Close','PCT_change','Adj. Volume']]

df1=data.loc["20040917",'PCT_change']
df2=data.loc["20040920",'PCT_change']
df3=data.loc["20040921",'PCT_change']
print("done")
print(df1,df2,df3)
'''
'''
time1 = '2001-01-01'
time2 = '2018-01-01'

#time1 = time1.replace("-", "")
#time1 = time1.replace("-", "")

time1 = dt.datetime.strptime(time1, "%Y-%m-%d")
time2 = dt.datetime.strptime(time2, "%Y-%m-%d")
'''
'''
yeet = time1
yeet = yeet.date()
yeet.strftime('%Y-%m-%d')
yeet = str(yeet)
yeet = yeet.replace("-", "")
print(yeet)
'''


a = np.array([0,1,0])
b = np.array([])
c = np.append(a,b)
print(c)
