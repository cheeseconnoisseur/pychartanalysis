
import tensorflow as tf
import pickle
import numpy as np
from numpy import genfromtxt
from numpy import array
import quandl
import math
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
from dateutil.relativedelta import relativedelta
import pandas as pd

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
time1 = '2009-09-20'
time2 = '2011-06-19'

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
bbigarray = np.array([])
bsmallarray=np.array([])
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
    bsmallarray = np.append(bsmallarray,smallarray)




while time1 not in [time2,time2s1,time2s2,time2s3,time2s4,time2s5,time2s6,time2s7,time2s9,time2s10]:
    time1 = time1 + dt.timedelta(days=1) #dt.timedelta(days=1)#df1 = changetoform(df1)df1=data.loc[[df1]]print(df1)
    df1=time1
    dabhi = dabhi+1
    print(dabhi)
    df1 = changetoform(df1)
    if np.size(bigarray) == 18:
        bbigarray = np.append(bbigarray,bigarray)
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
            print(dabhi)
            dabhi = dabhi-1
            print('no lol')
            continue


    else:



        try:
            df1=data.loc[df1,'PCT_change']
            print(df1)
            print(time1)
            print(dabhi)


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
            print(dabhi)
            dabhi = dabhi-1
        continue

'''
train_y = genfromtxt('y.csv', delimiter=',')
test_x = genfromtxt('tx.csv', delimiter=',')
test_y = genfromtxt('ty.csv', delimiter=',')
'''
print(bbigarray)
train_x = bbigarray
train_y = bsmallarray
print(train_x)

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 3
batch_size = 100
hm_epochs = 10

x = tf.placeholder(tf.float32, shape=(1, 18))
y = tf.placeholder(tf.float32)

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([18, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Nothing changes
def neural_network_model(data):
    #tf.cast()
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
				epoch_loss += c
				i+=batch_size

			print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


train_neural_network(x)
