import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

pik = 0
data = []



def getstock():
    stock = input('ENTER STOCK')
    print(stock)
    time1 = input('from date:yyyy-mm-dd')
    print(time1)
    time2 = input('to date:yyyy-mm-dd')
    print(time2)
    try:
        quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
        data = quandl.get("WIKI/"+stock, start_date="2001-12-31", end_date="2005-12-31")
        print(data)
    except:
        print('no')

def getforex():
    forr = input('ENTER FOREX')
    print(forr)
    time1 = input('from date:yyyy-mm-dd')
    print(time1)
    time2 = input('to date:yyyy-mm-dd')
    print(time2)
    try:
        quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
        data = quandl.get("FRED/"+forr, start_date="2001-12-31", end_date="2005-12-31")
        print(data)
    except:
        print('no')

def first():
    try:
        pickle_in = open('linearregression.pickle','rb')
    except:
        pik = 1
        choice = input('stock or forex')
        if choice == 'stock' or 's':
            getstock()
        else:
            getforex()
    print('lol')
first()
