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

# import and format data
# dataset_input = 0
#
# for name in glob.glob(r'D:\maven\data\sci\kp\insitu\data\2016\*\*.tab'):
# 	with open(name,) as csvfile:
# 		file = csv.reader(csvfile)
# 		i= 0
# 		for row in file:
# 			if i == 8:
# 				begin_row = row[0]
# 				begin_row = int(begin_row[6:9])-1
# 				#print(begin_row)
# 			elif i == 9:
# 				n_rows = row[0]
# 				n_rows = int(n_rows[4:9])
# 				#print(n_rows)
# 			elif i > 9:
# 				break
# 			i+=1
#
# 	df = pd.read_table(name,sep='\s+',names=['Datetime','Proton Density (n/cc)','Vx Velocity (km/s)','Vy Velocity (km/s)','Vz Velocity (km/s)','BX (nT GSE/GSM)','BY (nT GSE)','BZ (nT GSE)','Spacecraft MSO X (km)','Spacecraft MSO Y (km)','Spacecraft MSO Z (km)'],index_col=False,usecols=[0,40,42,44,46,127,129,131,189,190,191],na_values=[999.99,9999.99,99999.9,99999.99,9999999.],skiprows=begin_row,nrows=n_rows)[['Datetime','Spacecraft MSO X (km)','Spacecraft MSO Y (km)','Spacecraft MSO Z (km)','BX (nT GSE/GSM)','BY (nT GSE)','BZ (nT GSE)','Vx Velocity (km/s)','Vy Velocity (km/s)','Vz Velocity (km/s)','Proton Density (n/cc)']]
#
# 	#time_index_df =  pd.DataFrame(df['Year']*10000000 + df['Day']*10000 + df['Hour']*100 + df['Minute'])
# 	#print(time_index_df)
# 	#df.insert(4,'Datetime',time_index_df,)
# 	df['Spacecraft MSO X (km)'] = df['Spacecraft MSO X (km)']/r_m
# 	df['Spacecraft MSO Y (km)'] = df['Spacecraft MSO Y (km)']/r_m
# 	df['Spacecraft MSO Z (km)'] = df['Spacecraft MSO Z (km)']/r_m
#
# 	df.set_index('Datetime',inplace=True,drop=True)
# 	#print(df.index)
# 	df.index= pd.to_datetime(df.index,format='%Y-%m-%dT%H:%M:%S')
# 	df = df.resample('H').mean()
# 	print(df)
# 	year_dataset = df.to_numpy(dtype=np.float32)
# 	#year_dataset=year_dataset[:-24,4:]
#
#
# 	if dataset_input == 0:
# 		dataset = year_dataset
# 		dataset_input = 1
# 	else:
# 		dataset = np.concatenate([dataset,year_dataset],axis=0)
# 	print(dataset.shape)
#
# sw_dataset = remove_magnetosphere_measurements(dataset)
# np.savetxt(r'D:\forecasting_study\maven_hour_averaed_input.txt',sw_dataset)


# dataset = np.loadtxt(r'D:\forecasting_study\maven_hour_averaged_sw_measurements.txt')
# choose a number of time steps
n_steps_in_before, n_steps_out, n_steps_in_after = 3,4,2
n_features = 9
# dataset = dataset[:,3:]
# print(dataset.shape)
# # covert into input/output
# X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# #y = np.reshape(y,(y.shape[0],y.shape[2]))
y_train_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_train.txt')
y_test_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_test.txt')
X_train_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_train.txt')
X_test_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_test.txt')

y_test_raw = y_test_raw_reshaped.reshape(
    y_test_raw_reshaped.shape[0], y_test_raw_reshaped.shape[1] // n_features, n_features)
y_train_raw = y_train_raw_reshaped.reshape(
    y_train_raw_reshaped.shape[0], y_train_raw_reshaped.shape[1] // n_features, n_features)
X_test_raw = X_test_raw_reshaped.reshape(
    X_test_raw_reshaped.shape[0], X_test_raw_reshaped.shape[1] // n_features, n_features)
X_train_raw = X_train_raw_reshaped.reshape(
    X_train_raw_reshaped.shape[0], X_train_raw_reshaped.shape[1] // n_features, n_features)
#
# X_no_nans = np.vstack((X_train_raw,X_test_raw))
# y_no_nans = np.vstack((y_train_raw,y_test_raw))
#
# X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_no_nans,y_no_nans,test_size=0.35)

X_train, X_test, y_train, y_test = standardise(X_train_raw,X_test_raw,y_train_raw,y_test_raw)
print(X_test.shape, y_test.shape)

#print(X_test)
# # remove rows containing nans from dataset
# nan_rows = np.zeros(X.shape[0],dtype=bool)
#
# for i in range(X.shape[0]):
# 	for j in range(X.shape[1]):
# 		for k in range(X.shape[2]):
# 			if X[i,j,k] != X[i,j,k]:
# 				nan_rows[i] = True
# 	for j_2 in range(y.shape[1]):
# 		for k in range(X.shape[2]):
# 			if y[i,j_2,k] != y[i,j_2,k]:
# 				nan_rows[i] = True
# not_nan_rows = ~nan_rows
#
# X = X[not_nan_rows,:,:]
# y = y[not_nan_rows,:]
# print(X, y)
#
# # the dataset knows the number of features, e.g. 2
# n_features = X.shape[2]
#
# # split dataset into train and test subsets
# X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X,y,test_size=0.33)
# print(X_train_raw.shape,y_train_raw.shape)
# X_train, X_test, y_train, y_test = standardise(X_train_raw,X_test_raw,y_train_raw,y_test_raw)
# print(y_train)
#np.savetxt(r'D:\forecasting_study\omni_cnn_X_train.txt',X_train)
#np.savetxt(r'D:\forecasting_study\omni_cnn_y_train.txt',y_train)

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
# y10_train = y_train[:, :, 9]
# y11_train = y_train[:, :, 10]
# y12_train = y_train[:, :, 11]
# y13_train = y_train[:, :, 12]
# y14_train = y_train[:, :, 13]
# y15_train = y_train[:, 14].reshape((y_train.shape[0], 1))
# y16_train = y_train[:, 15].reshape((y_train.shape[0], 1))
# y17_train = y_train[:, 16].reshape((y_train.shape[0], 1))

y1_test = y_test[:, :, 0]
y2_test = y_test[:, :, 1]
y3_test = y_test[:, :, 2]
y4_test = y_test[:, :, 3]
y5_test = y_test[:, :, 4]
y6_test = y_test[:, :, 5]
y7_test = y_test[:, :, 6]
y8_test = y_test[:, :, 7]
y9_test = y_test[:, :, 8]
# y10_test = y_test[:, :, 9]
# y11_test = y_test[:, :, 10]
# y12_test = y_test[:, :, 11]
# y13_test = y_test[:, :, 12]
# y14_test = y_test[:, :, 13]
# y15_test = y_test[:, 14].reshape((y_test.shape[0], 1))
# y16_test = y_test[:, 15].reshape((y_test.shape[0], 1))
# y17_test = y_test[:, 16].reshape((y_test.shape[0], 1))


# define model
visible = Input(shape=(n_steps_in_before+n_steps_in_after, n_features))
cnn = Conv1D(filters=64, kernel_size=2, activation='sigmoid')(visible)
cnn = MaxPooling1D(pool_size=2)(cnn)

cnn = Conv1D(filters=32, kernel_size=2, activation='sigmoid')(cnn)
cnn = MaxPooling1D(pool_size=2, padding='same')(cnn)

#cnn = Conv1D(filters=16, kernel_size=2, activation='relu')(cnn)
#cnn = MaxPooling1D(pool_size=2, padding='same')(cnn)

cnn = Flatten()(cnn)
cnn = Dense(50, activation='sigmoid')(cnn)
#cnn = Dense(50, activation='relu')(cnn)
#cnn = Dense(50, activation='relu')(cnn)
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
# define output 1
#output10 = Dense(4)(cnn)
# define output 2
#output11 = Dense(4)(cnn)
# define output 3
# output12 = Dense(1)(cnn)
# define output 1
# output13 = Dense(1)(cnn)
# define output 2
# output14 = Dense(1)(cnn)
# define output 3
#output15 = Dense(1)(cnn)
# define output 1
#output16 = Dense(1)(cnn)
# define output 2
#output17 = Dense(1)(cnn)

# tie together
model = Model(inputs=visible, outputs=[output1, output2, output3, output4, output5, output6, output7, output8, output9])
model.compile(optimizer='adam', loss='mse')
print(model.summary())
# fit model
model.fit(X_train, [y1_train,y2_train,y3_train,y4_train,y5_train,y6_train,y7_train,y8_train,y9_train], epochs=200, verbose=True)
# demonstrate prediction
# model.save(r'D:\forecasting_study\maven_multistep_3_hour_input_before_2_hour_input_after_cnn_trained')
# model.save_weights(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_trained_weights')
# json_string = model.to_json()
# with open(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_trained_architecture','w') as f:
# 	f.write(json_string)


yhat = model.predict(X_test, verbose=0)
yhat_array = np.array(yhat)
print('yhat_array.shape: '+str(yhat_array.shape))
yhat_array = rearrange_axes(yhat_array)
print('yhat_array.shape: '+str(yhat_array.shape))
yhat_array_units = destandardise(yhat_array,X_train_raw)
print(yhat_array_units)
print('yhat_array.shape, y_test.shape: '+str(yhat_array.shape)+' , '+str(y_test.shape))
yhat_array_units_reshaped = yhat_array_units.reshape(yhat_array_units.shape[0], -1)
#y_test_raw_reshaped = y_test_raw.reshape(y_test_raw.shape[0], -1)
y_test_reshaped = y_test.reshape(y_test_raw.shape[0], -1)
np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_yhat_sigmoid.txt',yhat_array_units_reshaped)
