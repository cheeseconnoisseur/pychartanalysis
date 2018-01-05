import quandl
import math
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from dateutil.relativedelta import relativedelta
import numpy as np
import tensorflow as tf
from numpy import genfromtxt
import csv
import re
import types
from numpy import array
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

'''
a = np.array([0,1,0])
b = np.array([])
c = np.append(a,b)
print(c)
'''
'''
mk1 = np.array([0,1,0,0,0,1,1,0,0])
strr = str(mk1)
with open('cc.csv','a') as file:
    file.write(strr)
    file.write(',')
    file.close()


##text=List of strings to be written to file
def openn(df1):
    with open('csvfile.csv','a') as file:
        file.write(df1)
        file.write(',')
        file.close()

mk = np.array([0,1,0,0,0,1,1,0,0])
strr = str(mk)
df1 = strr
openn(df1)
'''
'''
train_x = genfromtxt('x.csv', delimiter=',',usecols=np.arange(0,0))
train_y = genfromtxt('y.csv', delimiter=',')
test_x = genfromtxt('tx.csv', delimiter=',',usecols=np.arange(0,1))
test_y = genfromtxt('ty.csv', delimiter=',')
'''

with open('tx.txt') as csvfile:
    f = csvfile.read()
    csvfile.close()
f=f.replace(" ","")
f=f.replace(",","")
f=f.replace(" ","")
f=f.replace(".",",")
f=f.replace("' ","")
f=f.replace("\n","")
f=f.replace("]","] ")
f=f.replace("' ","")
f=f.replace(",]","]")
f = f.split(" ")
print(f)

a = np.array([])
a = array( f )
a = a[:-1]
print(a)
b = np.array([[0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,0,1,2]])
np.concatenate([a, [0,0,1,0,0,1,1,0,0,0,0,1,0,0,1,0,0,1]])

print(a)


