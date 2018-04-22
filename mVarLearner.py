import pandas as pd
import csv
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

"""
datafunctions
"""
def series_to_sup(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(),list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		if(i==0):
			names+=[('var%d(t)' % (j+1))for j in range(n_vars)]
		else:
			names+=[('var%d(t+%d)' % (j+1, i))for j in range(n_vars)]
		agg = pd.concat(cols,axis=1)
		agg.columns = names
		if dropnan:
			agg.dropna(inplace=True)
		return agg
def create_model(): 
	model = Sequential()
	model.add(Dens(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
seed = 7
numpy.random.seed(seed)

minMkap = 100000
minVol = 10000
dfs = []
path = "/home/me/Documents/mnCoins/"
for coin in os.listdir(path):
	if("TOBA" not in coin):
		continue
	coin = path+coin
	curr = pd.read_csv(coin)
	end = len(curr)-1
	curr_vol = curr['vol'].iloc[end]
	curr_mkap = curr['mkap'].iloc[end]
	curr_roi = curr['roi'].iloc[end]
#filter the coins
	if(curr_roi<3000 and curr_vol<=minVol and curr_mkap<=minMkap):
		continue

	#create params
	vm = curr['vol'].astype(float)/curr['mkap'].astype(float)
	mv = curr['mkap'].astype(float)/curr['vol'].astype(float)
	vm_n = (curr['vol'].astype(float)/curr['mkap'].astype(float))/curr['ncount']
	mv_nr = (curr['mkap'].astype(float)/curr['vol'].astype(float))*(curr['ncount']/curr['roi'])
	v_np = curr['vol'].astype(float)/(curr['ncount']*curr['Prices'])
	nv = (curr['ncount'])/curr['vol'].astype(float)
	coins = (curr['mkap']/curr['Prices'])
	ncoins =coins*curr['ncount']
	rv = curr['roi']*curr['vol']
	#add to curr
	curr['rv'] = rv
	curr['vm'] = vm
	curr['nv'] = nv
	curr = curr.filter(['Prices',"vol","mkap","ncount","roi","nv"])
	vals = curr.values
	groups = [0,1,2,3,4,5]
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled = scaler.fit_transform(vals)
	reframed = series_to_sup(scaled, 1, 1)
	#print(reframed.head())
	# split into train and test sets
	values = reframed.values
	n_train_hours = int(len(values)*.85)
	train = values[:n_train_hours, :]
	test = values[n_train_hours:, :]
	# split into input and outputs
	train_X, train_y = train[:, :-1], train[:, -1]
	test_X, test_y = test[:, :-1], test[:, -1]
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
	# design network
	model = Sequential()
	model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')
	# fit network
	history = model.fit(train_X, train_y, epochs=500, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
	# plot history
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.show()


	

	#graphing
	"""
	i=1
	plt.figure()
	for group in groups:
		plt.subplot(len(groups),1,i)
		plt.plot(vals[:,group])
		plt.title(curr.columns[group], y=.5, loc='right')
		i+=1
	print(coin)
	print(curr.head(2))
	print(curr.corr())
	plt.show()
	"""
	#dfs.append(curr.corr())

	break
print(len(dfs))
