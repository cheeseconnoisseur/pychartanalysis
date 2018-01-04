import quandl
import math
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from dateutil.relativedelta import relativedelta

data = quandl.get("WIKI/GOOGL", start_date='2001-01-01', end_date='2018-01-01')
print (data)
data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100
data = data[['Adj. Close','PCT_change','Adj. Volume']]

df1=data.loc["20040917"]
print("done")
print(df1)
