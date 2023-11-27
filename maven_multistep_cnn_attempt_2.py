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
import datetime
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
rtod = 180./math.pi
x_0 = 0.64
ecc = 1.03
L = 2.04
k=0

def remove_magnetosphere_measurements(array):
	sw_array = array.copy()
	#print('array shape before magnetosphere removements '+str(sw_array.shape))
	for i in range(array.shape[0]):
		theta = np.arctan2((array[i,0]-x_0),np.sqrt(array[i,1]*array[i,1]+array[i,2]*array[i,2]))
		if (np.sqrt((array[i,0]-x_0)*(array[i,0]-x_0)+array[i,2]*array[i,2]+array[i,1]*array[i,1])) < (L/(1+ecc*np.cos(theta))):
			sw_array[i,3:] = np.NaN
	return sw_array


def split_sequences_throughcast_with_times(sequences, times, n_steps_in_before, n_steps_out, n_steps_in_after):
	print('times shape: '+str(times.shape))
	X, y, X_times, y_times  = np.zeros((int(n_steps_in_before+n_steps_in_after),sequences.shape[1])), np.zeros((int(n_steps_out),sequences.shape[1])), np.zeros((int(n_steps_in_before+n_steps_in_after))), np.zeros((int(n_steps_out)))
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in_before
		out_end_ix = end_ix + n_steps_out
		after_out_end_ix = out_end_ix + n_steps_in_after
		# check if we are beyond the dataset
		if after_out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		#seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, :-1]
		seq_x, seq_y, seq_x2= sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :], sequences[out_end_ix:after_out_end_ix, :]
		tim_x, tim_y, tim_x2= times[i:end_ix], times[end_ix:out_end_ix], times[out_end_ix:after_out_end_ix]
		print(seq_x.shape,seq_y.shape, seq_x2.shape)
		combined_x = np.vstack((seq_x,seq_x2))
		combined_x_times = np.concatenate((tim_x,tim_x2),axis=None)
		print(X.shape,combined_x.shape)
		print(y.shape,seq_y.shape)
		X = np.dstack((X,combined_x))
		#print(X)
		y = np.dstack((y,seq_y))
		#print(y)
		X_times = np.vstack((X_times,combined_x_times))
		print(y_times.shape,tim_y.shape)
		y_times = np.vstack((y_times,tim_y))
	return X, y, X_times, y_times

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
	#print(X_mean.shape)
	X_sigma = np.std(X_train,axis=(0,1))
	y_predict_destandardised = y_predict.copy()

	for i in range(y_predict.shape[1]):
		for n_test in range(y_predict.shape[0]):
			for n_steps in range(y_predict.shape[1]):
				y_predict_destandardised[n_test][n_steps][i] = (y_predict[n_test][n_steps][i]*X_sigma[i]) + X_mean[i]
	return y_predict_destandardised

for name in glob.glob(r'D:\maven\data\sci\kp\insitu\data\20*\*\*.tab'):
	with open(name,) as csvfile:
		file = csv.reader(csvfile)
		i= 0
		for row in file:
			if i == 8:
				begin_row = row[0]
				begin_row = int(begin_row[6:9])-1
				#print(begin_row)
			elif i == 9:
				n_rows = row[0]
				n_rows = int(n_rows[4:9])
				#print(n_rows)
			elif i > 9:
				break
			i+=1

	df = pd.read_table(name,sep='\s+',names=['Datetime','Proton Density (n/cc)','Vx Velocity (km/s)','Vy Velocity (km/s)','Vz Velocity (km/s)','Proton Temperature (eV)','Dynamic Pressure (nPa)','BX (nT GSE/GSM)','BY (nT GSE)','BZ (nT GSE)','Spacecraft MSO X (km)','Spacecraft MSO Y (km)','Spacecraft MSO Z (km)'],index_col=False,usecols=[0,40,42,44,46,48,50,127,129,131,189,190,191],na_values=[999.99,9999.99,99999.9,99999.99,9999999.],skiprows=begin_row,nrows=n_rows)[['Datetime','Spacecraft MSO X (km)','Spacecraft MSO Y (km)','Spacecraft MSO Z (km)','BX (nT GSE/GSM)','BY (nT GSE)','BZ (nT GSE)','Vx Velocity (km/s)','Vy Velocity (km/s)','Vz Velocity (km/s)','Proton Density (n/cc)','Proton Temperature (eV)','Dynamic Pressure (nPa)']]

	df['Spacecraft MSO X (km)'] = df['Spacecraft MSO X (km)']/r_m
	df['Spacecraft MSO Y (km)'] = df['Spacecraft MSO Y (km)']/r_m
	df['Spacecraft MSO Z (km)'] = df['Spacecraft MSO Z (km)']/r_m

	df.set_index('Datetime',inplace=True,drop=True)
	#print(df.index)
	df.index= pd.to_datetime(df.index,format='%Y-%m-%dT%H:%M:%S')
	df = df.resample('30T').mean()

	day_times = np.array(df.index, dtype=str)
	day_dataset = df.to_numpy(dtype=np.float32)
	print(day_dataset)
	#year_dataset = year_dataset[4:,0:]
	#year_dataset=year_dataset[:-24,4:]


	if dataset_input == 0:
		times = day_times
		dataset = day_dataset
		dataset_input = 1
	else:
		times = np.concatenate([times,day_times],axis=0)
		dataset = np.concatenate([dataset,day_dataset],axis=0)
	print(dataset.shape)

sw_dataset = remove_magnetosphere_measurements(dataset)

# choose a number of time steps
n_steps_in_before, n_steps_out, n_steps_in_after = 3,4,2
n_features = 9

unshaped_X, unshaped_y, X_times, y_times = split_sequences_throughcast_with_times(maven_data, maven_times, n_steps_in_before, n_steps_out, n_steps_in_after)
