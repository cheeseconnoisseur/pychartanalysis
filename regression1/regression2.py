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
data = {}
pik = 0
quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
style.use('ggplot')
pd.options.mode.chained_assignment = None


def forecast(forday, data, lday):
    olddata = data

    data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100
    data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100
    forecast_out = int(forday*len(data))
    print(forecast_out)
    data = data[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
    forecast_col = 'Adj. Close'
    data.fillna(value=-99999, inplace=True)
    data['label'] = data[forecast_col].shift(-forecast_out)
    x = np.array(data.drop(['label'], 1))
##    x = preprocessing.scale(x)
    x_recent = x[-forecast_out:]
    dp = input(x_recent)
    x = x[:-forecast_out]
    data.dropna(inplace=True)
    y = np.array(data['label'])
    #this shuffles up the table
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
    #making the classifier
    clas = LinearRegression()
    #'fit' means train
    #change pik to 1 to use previously saved model
    pik = 0
    if pik == 0 :
        clas.fit(x_train,y_train)
        with open('linearregression.pickle','wb') as f:
            pickle.dump(clas, f)
    else:
        pickle_in = open('linearregression.pickle','rb')
        clas = pickle.load(pickle_in)
    #score means test
    accuracy = clas.score(x_test,y_test)
    print(forecast_set)

    forecast_set = clas.predict(x_recent)


    print('forecast -> ',forecast_set)
    print('accuracy-> ', accuracy)
    print('forecasted ', forecast_out, ' days out')
    data['forecast'] = np.nan

##    last_date = lday
##    last_unix = last_date
##    last_unix = int(last_unix)
##    oneday = 60*60*24 #(86400)
##    next_unix = last_unix + oneday

    last_date = data.index[-1]
    last_unix = last_date.timestamp()
    oneday = 60*60*24 #(86400)
    next_unix = last_unix + oneday

    for i in forecast_set:
        #for loops iterated throught the forcast sets the future numbers in the df to nan
        next_date = dt.datetime.fromtimestamp(next_unix)
        next_unix += oneday
        #because the values are indexed by date it finds the date and if there isnt one
        #makes it it also makes all the other collums nan and adds the i
        #which is the forecast makingit just time and forcast.
        data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] + [i]
    plt.figure(figsize=(11,5.5))
    plt.subplot(121)
    plt.title('what happened')
    olddata['Adj. Close'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')



    plt.subplot(122)
    plt.title('what regression.py thought would happen')
    data['Adj. Close'].plot()
    data['forecast'].plot()
    #loc is location 4 is bottom right
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
################################################################################################################
    ##########################################################################
    ###################################################################
def futureforecast(forday, data):
    olddata = data
    data.to_csv('data.csv')
    datacv = pd.read_csv('data.csv')
    data["X_DATE"] = datacv["Date"] + dt.timedelta(days=forday*len(data))
    print(data.head())
    data.set_index('X_date', inplace=True)
    print(data.head())
     

        
    data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100
    data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100
    forecast_out = int(forday*len(data))
    print(forecast_out)
    data = data[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
    forecast_col = 'Adj. Close'
    data.fillna(value=-99999, inplace=True)
    data['label'] = data[forecast_col].shift(-forecast_out)
    x = np.array(data.drop(['label'], 1))
##    x = preprocessing.scale(x)
    x_recent = x[-forecast_out:]
    dp = input(x_recent)
    x = x[:-forecast_out]
    data.dropna(inplace=True)
    y = np.array(data['label'])
    #this shuffles up the table
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
    #making the classifier
    clas = LinearRegression()
    #'fit' means train
    #change pik to 1 to use previously saved model
    pik = 0
    if pik == 0 :
        clas.fit(x_train,y_train)
        with open('linearregression.pickle','wb') as f:
            pickle.dump(clas, f)
    else:
        pickle_in = open('linearregression.pickle','rb')
        clas = pickle.load(pickle_in)
    #score means test
    accuracy = clas.score(x_test,y_test)
    print(forecast_set)

    forecast_set = clas.predict(x_recent)


    print('forecast -> ',forecast_set)
    print('accuracy-> ', accuracy)
    print('forecasted ', forecast_out, ' days out')
    data['forecast'] = np.nan

##    last_date = lday
##    last_unix = last_date
##    last_unix = int(last_unix)
##    oneday = 60*60*24 #(86400)
##    next_unix = last_unix + oneday

    last_date = data.index[-1]
    last_unix = last_date.timestamp()
    oneday = 60*60*24 #(86400)
    next_unix = last_unix + oneday

    for i in forecast_set:
        #for loops iterated throught the forcast sets the future numbers in the df to nan
        next_date = dt.datetime.fromtimestamp(next_unix)
        next_unix += oneday
        #because the values are indexed by date it finds the date and if there isnt one
        #makes it it also makes all the other collums nan and adds the i
        #which is the forecast makingit just time and forcast.
        data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] + [i]
    plt.figure(figsize=(11,5.5))
    plt.subplot(121)
    plt.title('what happened')
    olddata['Adj. Close'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')



    plt.subplot(122)
    plt.title('what regression.py thought would happen')
    data['Adj. Close'].plot()
    data['forecast'].plot()
    #loc is location 4 is bottom right
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def process(data):
    data.to_csv('data.csv')
    datacv = pd.read_csv('data.csv')
    fday = datacv.iloc[-1,0]
    lday = datacv.iloc[0,0]
    print("LENGTH OF DATA --> ", len(data))
    fday = str(fday)
    lday = str(lday)
    fday = fday.replace("-", "")
    lday = lday.replace("-", "")
    fday = dt.datetime.strptime(fday,'%Y%m%d')
    lday = dt.datetime.strptime(lday,'%Y%m%d')
    #iloc (pandas) finds something by index or position
    oneday = 60*60*24 #(86400)
    ftlday = fday - lday
    ftlday = str(ftlday)
    ftlday = ftlday.replace(" days, 0:00:00", "")
    ftlday = int(ftlday)
    print("length of time set--->", ftlday)
    dp = input('how many days in advanced to predict:')
    dp = int(dp)
    forday = (dp/len(data))
    print("value of forcasted days forward as a decimal of the lenghth of time in dataset ----> ", end="")
    print(forday)
    forrday = forday*100
    forecast(forday, data, lday)

def processwithopik():
    data1 = quandl.get("WIKI/GOOGL", start_date='2001-01-01')
    print(data1)
    dp = input('how many days in advanced to predict:')
    dp = int(dp)
    forday = (dp/len(data1))
    print("value of forcasted days forward as a decimal of the lenghth of time in dataset ----> ", end="")
    print(forday)
    forrday = forday*100
    futureforecast(forday, data1)


def getstock():
    stock = input('ENTER STOCK')
    print(stock)
    time1 = input('from date:yyyy-mm-dd')
    print(time1)
    time2 = input('to date:yyyy-mm-dd')
    print(time2)
    quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
    data = quandl.get("WIKI/"+stock, start_date=time1, end_date=time2)
    print(data.head())
    process(data)

def getstockd():
    forp = input('do you want to predict using past data or predict future(pp,pf)')
    if forp == 'pp':
        stock = input('ENTER STOCK')
        print(stock)
        time1 = input('from date:yyyy-mm-dd')
        print(time1)
        time2 = input('to date:yyyy-mm-dd')
        print(time2)
        quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
        data = quandl.get("WIKI/"+stock, start_date=time1, end_date=time2)
        print(data.head())
        process(data)
    else:
        try:
            pickle_in = open('linearregression.pickle','rb')
            print('###model pickle has been found###')
            processwithopik()
##            _ = input('do you want to use model pickle(yes,no)')
##            processwithopik()
##            #if _ == 'yes':
            

        except:
            processwithopik()
            print("no")




def getforex():
    forr = input('ENTER FOREX')
    print(forr)
    time1 = input('from date:yyyy-mm-dd')
    print(time1)
    time2 = input('to date:yyyy-mm-dd')
    print(time2)
    quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
    data = quandl.get("FRED/"+forr, start_date="2001-12-31", end_date="2016-12-31")
    print(data.head())
    process(data)


def first():
    choice = input('stock or forex(in beta)')
    if choice in ('s', 'stock', 'stocks'):
        getstock()
    elif choice in ('sd'):
        print("devmode")
        getstockd()
    else:
        print("i said its in damn beta why are you here it barely works go to stocks its just better trust me thank you np")
        getforex()
first()

'''
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
data = {}
pik = 0
quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
style.use('ggplot')
pd.options.mode.chained_assignment = None

def forecast(forday, data):
    olddata = data

    data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100
    data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100
    forecast_out = int(forday*len(data))
    print(forecast_out)
    data = data[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
    forecast_col = 'Adj. Close'
    data.fillna(value=-99999, inplace=True)
    data['label'] = data[forecast_col].shift(-forecast_out)
    x = np.array(data.drop(['label'], 1))
##    x = preprocessing.scale(x)
    x_recent = x[-forecast_out:]
    dp = input(x_recent)
    x = x[:-forecast_out]
    data.dropna(inplace=True)
    y = np.array(data['label'])
    #this shuffles up the table
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
    #making the classifier
    clas = LinearRegression()
    #'fit' means train
    #change pik to 1 to use previously saved model
    pik = 0
    if pik == 0 :
        clas.fit(x_train,y_train)
        with open('linearregression.pickle','wb') as f:
            pickle.dump(clas, f)
    else:
        pickle_in = open('linearregression.pickle','rb')
        clas = pickle.load(pickle_in)
    #score means test
    accuracy = clas.score(x_test,y_test)

    forecast_set = clas.predict(x_recent)


    print('forecast -> ',forecast_set)
    print('accuracy-> ', accuracy)
    print('forecasted ', forecast_out, ' days out')
    data['forecast'] = np.nan

##    last_date = lday
##    last_unix = last_date
##    last_unix = int(last_unix)
##    oneday = 60*60*24 #(86400)
##    next_unix = last_unix + oneday

    last_date = data.index[-1]
    last_unix = last_date.timestamp()
    oneday = 60*60*24 #(86400)
    next_unix = last_unix + oneday

    for i in forecast_set:
        #for loops iterated throught the forcast sets the future numbers in the df to nan
        next_date = dt.datetime.fromtimestamp(next_unix)
        next_unix += oneday
        #because the values are indexed by date it finds the date and if there isnt one
        #makes it it also makes all the other collums nan and adds the i
        #which is the forecast makingit just time and forcast.
        data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] + [i]
    plt.figure(figsize=(11,5.5))
    plt.subplot(121)
    plt.title('what happened')
    olddata['Adj. Close'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')



    plt.subplot(122)
    plt.title('what regression.py thought would happen')
    data['Adj. Close'].plot()
    data['forecast'].plot()
    #loc is location 4 is bottom right
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def forecastnew(forday, data, dp):
    olddata = data

    data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100
    data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100
    forecast_out = int(forday*len(data))
    print(forecast_out)
    data = data[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
    forecast_col = 'Adj. Close'
    data.fillna(value=-99999, inplace=True)
    data['label'] = data[forecast_col].shift(-forecast_out)
    x = np.array(data.drop(['label'], 1))
##    x = preprocessing.scale(x)
    x_recent = olddata.tail(dp)
    last_date = data.index[-1]
    last_unix = last_date.timestamp()
    oneday = 60*60*24 #(86400)
    next_unix = last_unix + oneday
    x_recent = olddata[['PCT_change']]
    for i in x_recent:
        #for loops iterated throught the forcast sets the future numbers in the df to nan
        next_date = dt.datetime.fromtimestamp(next_unix)
        next_unix += oneday
        #because the values are indexed by date it finds the date and if there isnt one
        #makes it it also makes all the other collums nan and adds the i
        #which is the forecast makingit just time and forcast.
        olddata.loc[next_date] = [np.nan for _ in range(len(olddata.columns)-1)] + [i]
    print(x_recent)

    x = x[:-forecast_out]
    x = x[:]
    data.dropna(inplace=True)
    y = np.array(data['label'])
    #this shuffles up the table
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
    #making the classifier
    clas = LinearRegression()
    #'fit' means train
    #change pik to 1 to use previously saved model
    pik = 0
    if pik == 0 :
        clas.fit(x_train,y_train)
        with open('linearregression.pickle','wb') as f:
            pickle.dump(clas, f)
    else:
        pickle_in = open('linearregression.pickle','rb')
        clas = pickle.load(pickle_in)
    #score means test
    accuracy = clas.score(x_test,y_test)
    print(x_recent)

    forecast_set = clas.predict(x_recent)


    print('forecast -> ',forecast_set)
    print('accuracy-> ', accuracy)
    print('forecasted ', forecast_out, ' days out')
    data['forecast'] = np.nan

##    last_date = lday
##    last_unix = last_date
##    last_unix = int(last_unix)
##    oneday = 60*60*24 #(86400)
##    next_unix = last_unix + oneday

    last_date = data.index[-1]
    last_unix = last_date.timestamp()
    oneday = 60*60*24 #(86400)
    next_unix = last_unix + oneday

    for i in forecast_set:
        #for loops iterated throught the forcast sets the future numbers in the df to nan
        next_date = dt.datetime.fromtimestamp(next_unix)
        next_unix += oneday
        #because the values are indexed by date it finds the date and if there isnt one
        #makes it it also makes all the other collums nan and adds the i
        #which is the forecast makingit just time and forcast.
        data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] + [i]
    plt.figure(figsize=(11,5.5))
    plt.subplot(121)
    plt.title('what happened')
    olddata['Adj. Close'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')



    plt.subplot(122)
    plt.title('what regression.py thought would happen')
    data['Adj. Close'].plot()
    data['forecast'].plot()
    #loc is location 4 is bottom right
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def process(data):
    data.to_csv('data.csv')
    datacv = pd.read_csv('data.csv')
    fday = datacv.iloc[-1,0]
    lday = datacv.iloc[0,0]
    print("LENGTH OF DATA --> ", len(data))
    fday = str(fday)
    lday = str(lday)
    fday = fday.replace("-", "")
    lday = lday.replace("-", "")
    fday = dt.datetime.strptime(fday,'%Y%m%d')
    lday = dt.datetime.strptime(lday,'%Y%m%d')
    #iloc (pandas) finds something by index or position
    oneday = 60*60*24 #(86400)
    ftlday = fday - lday
    ftlday = str(ftlday)
    ftlday = ftlday.replace(" days, 0:00:00", "")
    ftlday = int(ftlday)
    print("length of time set--->", ftlday)
    dp = input('how many days in advanced to predict:')
    dp = int(dp)
    forday = (dp/len(data))
    print("value of forcasted days forward as a decimal of the lenghth of time in dataset ----> ", end="")
    print(forday)
    forrday = forday*100
    forecast(forday, data, lday)

def processwithoutpik():
    data1 = quandl.get("WIKI/GOOGL", start_date='2001-01-01')
    print(data1)
    dp = input('how many days in advanced to predict:')
    dp = int(dp)
    forday = (dp/len(data1))
    forecastnew(forday, data1, dp)



def getstock():
    stock = input('ENTER STOCK')
    print(stock)
    time1 = input('from date:yyyy-mm-dd')
    print(time1)
    time2 = input('to date:yyyy-mm-dd')
    print(time2)
    quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
    data = quandl.get("WIKI/"+stock, start_date=time1, end_date=time2)
    print(data.head())
    process(data)

def getstockd():
    forp = input('do you want to predict using past data or predict future(pp,pf)')
    if forp == 'pp':
        stock = input('ENTER STOCK')
        print(stock)
        time1 = input('from date:yyyy-mm-dd')
        print(time1)
        time2 = input('to date:yyyy-mm-dd')
        print(time2)
        quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
        data = quandl.get("WIKI/"+stock, start_date=time1, end_date=time2)
        print(data.head())
        process(data)
    else:
        try:
            pickle_in = open('linearregression.pickle','rb')
            print('###model pickle has been found###')
            processwithoutpik(data)
            _ = input('do you want to use model pickle(yes,no)')
            if _ == 'yes':
                processwithoutpik(data)
            else:
                processwithoutpik(data)

        except:
            processwithoutpik()
            print("no")




def getforex():
    forr = input('ENTER FOREX')
    print(forr)
    time1 = input('from date:yyyy-mm-dd')
    print(time1)
    time2 = input('to date:yyyy-mm-dd')
    print(time2)
    quandl.ApiConfig.api_key = '2PDeyG166PJWMeZ1fZqi'
    data = quandl.get("FRED/"+forr, start_date="2001-12-31", end_date="2016-12-31")
    print(data.head())
    process(data)


def first():
    choice = input('stock or forex(in beta)')
    if choice in ('s', 'stock', 'stocks'):
        getstock()
    elif choice in ('sd'):
        print("devmode")
        getstockd()
    else:
        print("i said its in damn beta why are you here it barely works go to stocks its just better trust me thank you np")
        getforex()
first()
'''
