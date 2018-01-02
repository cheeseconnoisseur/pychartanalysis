import pandas as pd
import quandl
import math
import datetime as dt
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

pik = 0
data = []
quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'


##def process(n):
##    dp = input('how many days in advanced to predict:')
##    print(dp)
##    fday = n.ix[0]['Date']
##    fday = pd.Series(n)
##    
##    
##    print(n.head())
##    print(' ')
##    print(len(n))
##    print(fday)


def getstock():
    #stock = input('ENTER STOCK')
    #print(stock)
    #time1 = input('from date:yyyy-mm-dd')
    #print(time1)
    #time2 = input('to date:yyyy-mm-dd')
    #print(time2)
    quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
    data = quandl.get("WIKI/GOOGL", start_date="2001-12-31", end_date="2005-12-31")
    print(data)
    print(data.head())
    dp = input('how many days in advanced to predict:')
    print(dp)

    data.to_csv('data.csv')
    datacv = pd.read_csv('data.csv')
    fday = datacv.iloc[-1,0]


    
    print(data.head())
    print(' ')
    print(len(data))
    print(' ')
    print(' ')
    print(' ')

    print(fday)
    #iloc (pandas) finds something by index or position
    last_unix = pd.to_datetime('13000101', format='%Y%m%d', errors='ignore',infer_datetime_format=True)
    oneday = 60*60*24 #(86400)
    next_unix = last_unix + dt.timedelta(minutes=1440)
    print(next_unix)




def getforex():
    forr = input('ENTER FOREX')
    print(forr)
    time1 = input('from date:yyyy-mm-dd')
    print(time1)
    time2 = input('to date:yyyy-mm-dd')
    print(time2)
    try:

        data = quandl.get("FRED/"+forr, start_date="2001-12-31", end_date="2005-12-31")
        print(data)
    except:
        print('no')

def first():
   # try:
 #       pickle_in = open('linearregression.pickle','rb')
  #  except:
    pik = 1
    choice = input('stock or forex')
    if choice == 'stock' or 's':
        getstock()
    elif choice == 'f' or 'forex':
        getforex()
    process(data)


first()
