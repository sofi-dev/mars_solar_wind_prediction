import math
import numpy as np
from numpy import array
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure, show, rc
import pandas as pd
import glob
import csv
import sklearn as sk
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
import warnings
warnings.filterwarnings("ignore")

# multivariate output 1d cnn example
from numpy import array
from numpy import hstack
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model

# define useful constants
r_m = 3389
dtor = math.pi/180.
x_0 = 0.64
ecc = 1.02
L = 2.06
k=0


# split a multivariate sequence into samples
def remove_magnetosphere_measurements(array):
	sw_array = array.copy()
	for i in range(array.shape[0]):
		theta = np.arctan2(np.sqrt(array[i,1]*array[i,1]+array[i,2]*array[i,2]),array[i,0]-x_0)

		if (array[i,0]-x_0) < np.cos(theta)*(L/(1+ecc*np.cos(theta))):
			if np.sin(theta)*(np.sqrt(array[i,1]*array[i,1]+array[i,2]*array[i,2])) < (L/(1+ecc*np.cos(theta))):
				sw_array[i,4:] = np.NaN
				sw_array[i,4:] = np.NaN
				sw_array[i,4:] = np.NaN
	return sw_array


def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		#seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, :-1]
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1:out_end_ix, :]
		#print(seq_x.shape,seq_y.shape)
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def standardise(X_train,X_test,y_train,y_test):
	X_mean = np.mean(X_train,axis=(0,1))
	print(X_mean.shape)
	X_sigma = np.std(X_train,axis=(0,1))
	X_train_standardised = X_train.copy()
	X_test_standardised = X_test.copy()
	y_train_standardised = y_train.copy()
	y_test_standardised = y_test.copy()

	for i in range(X_train.shape[2]):
		for n_train in range(X_train.shape[0]):
			for y_steps in range(y_train.shape[1]):
				y_train_standardised[n_train][y_steps][i] = (y_train[n_train][y_steps][i] - X_mean[i])/X_sigma[i]
			for X_steps in range(X_train.shape[1]):
				X_train_standardised[n_train][X_steps][i] = (X_train[n_train][X_steps][i] - X_mean[i])/X_sigma[i]

	for i in range(X_train.shape[2]):
		for n_test in range(X_test.shape[0]):
			for y_steps in range(y_train.shape[1]):
				y_test_standardised[n_test][y_steps][i] = (y_test[n_test][y_steps][i] - X_mean[i])/X_sigma[i]
			for X_steps in range(X_train.shape[1]):
				X_test_standardised[n_test][X_steps][i] = (X_test[n_test][X_steps][i] - X_mean[i])/X_sigma[i]

	return X_train_standardised,X_test_standardised,y_train_standardised,y_test_standardised

def destandardise(y_predict,X_train):
	X_mean = np.mean(X_train,axis=(0,1))
	print(X_mean.shape)
	X_sigma = np.std(X_train,axis=(0,1))
	y_predict_destandardised = y_predict.copy()

	for i in range(y_predict.shape[1]):
		for n_test in range(y_predict.shape[0]):
			for n_steps in range(y_predict.shape[1]):
				y_predict_destandardised[n_test][n_steps][i] = (y_predict[n_test][n_steps][i]*X_sigma[i]) + X_mean[i]
	return y_predict_destandardised

def rearrange_axes(yhat_array):
	yhat_rearranged = np.zeros((yhat_array.shape[1],yhat_array.shape[2],yhat_array.shape[0]))
	for i in range(yhat_array.shape[0]):
		for j in range(yhat_array.shape[1]):
			for k in range(yhat_array.shape[2]):
				yhat_rearranged[j][k][i] = yhat_array[i][j][k]
	return yhat_rearranged

make_data = False

training_set_percentages = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]

if make_data == True:
	# choose a number of time steps
	n_steps_in_before, n_steps_out, n_steps_in_after = 3,4,2
	n_features = 9

	y_train_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_y_train.txt')
	y_test_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_y_test.txt')
	X_train_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_x_train.txt')
	X_test_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_x_test.txt')

	y_test_raw = y_test_raw_reshaped.reshape(
	    y_test_raw_reshaped.shape[0], y_test_raw_reshaped.shape[1] // n_features, n_features)
	y_train_raw = y_train_raw_reshaped.reshape(
	    y_train_raw_reshaped.shape[0], y_train_raw_reshaped.shape[1] // n_features, n_features)
	X_test_raw = X_test_raw_reshaped.reshape(
	    X_test_raw_reshaped.shape[0], X_test_raw_reshaped.shape[1] // n_features, n_features)
	X_train_raw = X_train_raw_reshaped.reshape(
	    X_train_raw_reshaped.shape[0], X_train_raw_reshaped.shape[1] // n_features, n_features)

	X_no_nans = np.vstack((X_train_raw,X_test_raw))
	y_no_nans = np.vstack((y_train_raw,y_test_raw))
	#training_set_percentages = [0.99,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50]

	test_accuracies = np.zeros(10)
	train_accuracies = np.zeros(10)
	for i in range(len(training_set_percentages)):
		print('begin training size='+str(training_set_percentages[i]))
		test_size = training_set_percentages[i]

		X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_no_nans,y_no_nans,test_size=test_size)

		X_train, X_test, y_train, y_test = standardise(X_train_raw,X_test_raw,y_train_raw,y_test_raw)

		# separate output
		y1_train = y_train[:, :, 0]
		y2_train = y_train[:, :, 1]
		y3_train = y_train[:, :, 2]
		y4_train = y_train[:, :, 3]
		y5_train = y_train[:, :, 4]
		y6_train = y_train[:, :, 5]
		y7_train = y_train[:, :, 6]
		y8_train = y_train[:, :, 7]
		y9_train = y_train[:, :, 8]

		y1_test = y_test[:, :, 0]
		y2_test = y_test[:, :, 1]
		y3_test = y_test[:, :, 2]
		y4_test = y_test[:, :, 3]
		y5_test = y_test[:, :, 4]
		y6_test = y_test[:, :, 5]
		y7_test = y_test[:, :, 6]
		y8_test = y_test[:, :, 7]
		y9_test = y_test[:, :, 8]

		# define model
		visible = Input(shape=(n_steps_in_before+n_steps_in_after, n_features))
		cnn = Conv1D(filters=64, kernel_size=2, activation='relu')(visible)
		cnn = MaxPooling1D(pool_size=2)(cnn)

		cnn = Conv1D(filters=32, kernel_size=2, activation='relu')(cnn)
		cnn = MaxPooling1D(pool_size=2, padding='same')(cnn)

		cnn = Flatten()(cnn)
		cnn = Dense(50, activation='relu')(cnn)

		# define output 1
		output1 = Dense(4)(cnn)
		# define output 2
		output2 = Dense(4)(cnn)
		# define output 3
		output3 = Dense(4)(cnn)
		# define output 1
		output4 = Dense(4)(cnn)
		# define output 2
		output5 = Dense(4)(cnn)
		# define output 3
		output6 = Dense(4)(cnn)
		# define output 1
		output7 = Dense(4)(cnn)
		# define output 2
		output8 = Dense(4)(cnn)
		# define output 3
		output9 = Dense(4)(cnn)

		# tie together
		model = Model(inputs=visible, outputs=[output1, output2, output3, output4, output5, output6, output7, output8, output9])
		model.compile(optimizer='adam', loss='mse')
		#print(model.summary())
		# fit model
		model.fit(X_train, [y1_train,y2_train,y3_train,y4_train,y5_train,y6_train,y7_train,y8_train,y9_train], epochs=200, verbose=False)
		test_results = model.evaluate(X_test,[y1_test,y2_test,y3_test,y4_test,y5_test,y6_test,y7_test,y8_test,y9_test])
		train_results = model.evaluate(X_train,[y1_train,y2_train,y3_train,y4_train,y5_train,y6_train,y7_train,y8_train,y9_train])
		print('training set mse: '+str(train_results[0])+', test set mse: '+str(test_results[0]))

		train_accuracies=np.vstack((train_accuracies,train_results))
		test_accuracies=np.vstack((test_accuracies,test_results))
		print('done')
	np.savetxt(r'D:\forecasting_study\training_size_test_accuracies.txt',test_accuracies)
	np.savetxt(r'D:\forecasting_study\training_size_train_accuracies.txt',train_accuracies)
elif make_data == False:
	test_accuracies = np.loadtxt(r'D:\forecasting_study\training_size_test_accuracies.txt')
	train_accuracies = np.loadtxt(r'D:\forecasting_study\training_size_train_accuracies.txt')

print(train_accuracies.shape,test_accuracies.shape)
train_accuracies = np.delete(train_accuracies,[0],0)
test_accuracies = np.delete(test_accuracies,[0],0)

x=[50,55,60,65,70,75,80,85,90,95]
print(len(x))

fig,axs = plt.subplots(6,tight_layout=True,sharex=True)
axs[0].plot(x,train_accuracies[0:,0],label='train',color=cm.Greys(140))
axs[0].plot(x,test_accuracies[0:,0],label='test',linestyle='dashed',color=cm.Greys(140))
axs[0].set_ylabel('Total loss')

axs[1].plot(x,train_accuracies[0:,1],label='train',color=cm.plasma(0))
axs[1].plot(x,test_accuracies[0:,1],label='test',linestyle='dashed',color=cm.plasma(0),alpha=0.75)

axs[1].plot(x,train_accuracies[0:,2],label='train',color=cm.plasma(125))
axs[1].plot(x,test_accuracies[0:,2],label='test',linestyle='dashed',color=cm.plasma(125),alpha=0.75)

axs[1].plot(x,train_accuracies[0:,3],label='train',color=cm.plasma(250))
axs[1].plot(x,test_accuracies[0:,3],label='test',linestyle='dashed',color=cm.plasma(250),alpha=0.75)
axs[1].set_ylabel('B loss')

axs[2].plot(x,train_accuracies[0:,4],label='train',color=cm.plasma(0))
axs[2].plot(x,test_accuracies[0:,4],label='test',linestyle='dashed',color=cm.plasma(0),alpha=0.75)

axs[2].plot(x,train_accuracies[0:,5],label='train',color=cm.plasma(125))
axs[2].plot(x,test_accuracies[0:,5],label='test',linestyle='dashed',color=cm.plasma(125),alpha=0.75)

axs[2].plot(x,train_accuracies[0:,6],label='train',color=cm.plasma(250))
axs[2].plot(x,test_accuracies[0:,6],label='test',linestyle='dashed',color=cm.plasma(250),alpha=0.75)
axs[2].set_ylabel('v loss')

axs[3].plot(x,train_accuracies[0:,7],label='train',color=cm.Greys(140))
axs[3].plot(x,test_accuracies[0:,7],label='test',linestyle='dashed',color=cm.Greys(140))
axs[3].set_ylabel('n_p loss')

axs[4].plot(x,train_accuracies[0:,8],label='train',color=cm.Greys(140))
axs[4].plot(x,test_accuracies[0:,8],label='test',linestyle='dashed',color=cm.Greys(140))
axs[4].set_ylabel('T_p loss')

axs[5].plot(x,train_accuracies[0:,9],label='train',color=cm.Greys(140))
axs[5].plot(x,test_accuracies[0:,9],label='test',linestyle='dashed',color=cm.Greys(140))
axs[5].set_xlabel('Training set size (% of measurements)')
axs[5].set_xlim(xmin=50,xmax=95)
axs[5].set_ylabel('P_dyn loss')

plt.show()

fig, axs = plt.subplots(1)
axs.plot(x[0:4],train_accuracies[0:4,0],color=cm.Greys(140),label='training set')
axs.plot(x[0:4],test_accuracies[0:4,0],linestyle='dashed',color=cm.Greys(140),label='test set')
axs.legend()
plt.show()
