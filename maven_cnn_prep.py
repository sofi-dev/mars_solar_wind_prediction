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

# define useful constants
r_m = 3389
dtor = math.pi/180.
x_0 = 0.64
ecc = 1.02
L = 2.06
k=0

def convert_to_spherical(x,y,z):
	magnitude = np.zeros(x.shape[0])
	clock = np.zeros(x.shape[0])
	cone = np.zeros(x.shape[0])
	for i in range(x.shape[0]):
		magnitude[i] = np.sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i])
		clock[i] = rad_to_deg*np.arctan2(z[i],y[i])
		cone[i] = rad_to_deg*np.arccos(x[i]/magnitude[i])
	print(min(clock),max(clock),min(cone),max(cone))
	return magnitude, clock, cone

# split a multivariate sequence into samples
def remove_magnetosphere_measurements(DataFrame):
	array = DataFrame.to_numpy()
	sw_array = array.copy()
	sw_array_times = DataFrame.index.to_numpy()
	#print(sw_array_times)
	for i in range(array.shape[0]):
		theta = np.arctan2(np.sqrt(array[i,1]*array[i,1]+array[i,2]*array[i,2]),array[i,0]-x_0)

		if (array[i,0]-x_0) < np.cos(theta)*(L/(1+ecc*np.cos(theta))):
			if np.sin(theta)*(np.sqrt(array[i,1]*array[i,1]+array[i,2]*array[i,2])) < (L/(1+ecc*np.cos(theta))):
				sw_array[i,4:] = np.NaN
				print('removed')
	return sw_array,sw_array_times


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

# import and format data
dataset_input = 0

for name in glob.glob(r'D:\maven\data\sci\kp\insitu\data\*\*\*.tab'):
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

	#time_index_df =  pd.DataFrame(df['Year']*10000000 + df['Day']*10000 + df['Hour']*100 + df['Minute'])
	#print(time_index_df)
	#df.insert(4,'Datetime',time_index_df,)
	df['Spacecraft MSO X (km)'] = df['Spacecraft MSO X (km)']/r_m
	df['Spacecraft MSO Y (km)'] = df['Spacecraft MSO Y (km)']/r_m
	df['Spacecraft MSO Z (km)'] = df['Spacecraft MSO Z (km)']/r_m

	df.set_index('Datetime',inplace=True,drop=True)

	df.index= pd.to_datetime(df.index,format='%Y-%m-%dT%H:%M:%S')
	df = df.resample('30T').mean()
	#print(df.columns)
	#print(df)
	#year_dataset = df.to_numpy(dtype=np.float32)
	#year_dataset=year_dataset[:-24,4:]

	if dataset_input == 0:
		dataset = df.copy()
		dataset_input = 1
	else:
		dataset = pd.concat([dataset,df],axis=0)
		print(dataset.shape)
		#print(dataset)


sw_dataset,sw_dataset_times = remove_magnetosphere_measurements(dataset)


print(sw_dataset_times)
np.savetxt(r'D:\scripts_for_thesis\forecasting_study\maven_30_minute_averaged_sw_measurements_.txt',sw_dataset)
np.savetxt(r'D:\scripts_for_thesis\forecasting_study\maven_30_minute_averaged_sw_times_spherical.txt',sw_dataset_times,fmt="%s")
plt.plot(sw_dataset_times,sw_dataset[0:,0])
plt.show()
