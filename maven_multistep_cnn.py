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
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# define useful constants
r_m = 3389
dtor = math.pi/180.
x_0 = 0.64
ecc = 1.03
L = 2.04
k=0

def copyModel2Model(model_source,model_target,certain_layer=""):
    for l_tg,l_sr in zip(model_target.layers,model_source.layers):
        wk0=l_sr.get_weights()
        l_tg.set_weights(wk0)
        if l_tg.name==certain_layer:
            break
            print("model source was copied into model target")


# split a multivariate sequence into samples
def remove_magnetosphere_measurements(array):
    sw_array = array.copy()
    #print('array shape before magnetosphere removements '+str(sw_array.shape))
    for i in range(array.shape[0]):
        theta = np.arctan2((array[i,0]-x_0),np.sqrt(array[i,1]*array[i,1]+array[i,2]*array[i,2]))
        if (np.sqrt((array[i,0]-x_0)*(array[i,0]-x_0)+array[i,2]*array[i,2]+array[i,1]*array[i,1])) < (L/(1+ecc*np.cos(theta))):
            sw_array[i,3:] = np.NaN
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

def split_sequences_with_times(sequences, times, n_steps_in, n_steps_out):
    X, y, X_times, y_times = list(), list(), list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
            # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1:out_end_ix, :]
        tim_x, tim_y = times[i:end_ix], times[end_ix-1:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
        X_times.append(tim_x)
        y_times.append(tim_y)
    return array(X), array(y), array(X_times), array(y_times)

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

def rearrange_axes(yhat_array):
    yhat_rearranged = np.zeros((yhat_array.shape[1],yhat_array.shape[2],yhat_array.shape[0]))
    for i in range(yhat_array.shape[0]):
        for j in range(yhat_array.shape[1]):
            for k in range(yhat_array.shape[2]):
                yhat_rearranged[j][k][i] = yhat_array[i][j][k]
    return yhat_rearranged

def convert_to_datetime(array):
    datetime_array = np.empty(array.shape[0],dtype=datetime)
    for i in range(array.shape[0]):
        datetime_array[i] = datetime.strptime(array[i],'%Y-%m-%dT%H:%M:%S')
        #print(datetime_array[i])
    return datetime_array

# import and format data
dataset_input = 0
# choose a number of time steps
n_steps_in_before, n_steps_out, n_steps_in_after = 9,12,6
n_features = 9
n_positions = 3

make_dataset = True

if make_dataset == True:
    for name in glob.glob(r'D:\maven\data\sci\kp\insitu\data\20*\*\*.tab'):
        with open(name,) as csvfile:
            #print(name)
            file = csv.reader(csvfile)
            i= 0
            for row in file:
                #print('row = '+str(i))
                if i == 8:
                    begin_row = row[0]
                    begin_row = int(begin_row[6:9])-1
                    #print('begin row: '+str(begin_row))
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
            print(df)
            day_times = np.array(df.index, dtype=str)
            day_dataset = df.to_numpy(dtype=np.float32)

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
    #np.savetxt(r'D:\forecasting_study\maven__averaged_input.txt',sw_dataset)
    maven_data = sw_dataset[1:,0:]
    maven_times = times[1:]

    # choose a number of time steps
    n_steps_in_before, n_steps_out, n_steps_in_after = 3,4,2
    n_features = 9
    #maven_data = maven_data[:,3:]
    print(maven_data.shape)
    # covert into input/output
    unshaped_X, unshaped_y, X_times, y_times = split_sequences_throughcast_with_times(maven_data, maven_times, n_steps_in_before, n_steps_out, n_steps_in_after)
    #y = np.reshape(y,(y.shape[0],y.shape[2]))
    print(unshaped_X.shape, unshaped_y.shape, X_times.shape, y_times.shape)

    # reshape arrays for model input and output
    X = np.zeros((unshaped_X.shape[2],unshaped_X.shape[0],unshaped_X.shape[1]))
    y = np.zeros((unshaped_y.shape[2],unshaped_y.shape[0],unshaped_y.shape[1]))

    for k in range(unshaped_X.shape[2]):
        for j in range(unshaped_X.shape[1]):
            for i in range(unshaped_X.shape[0]):
                X[k,i,j] = unshaped_X[i,j,k]
            for i_2 in range(unshaped_y.shape[0]):
                y[k,i_2,j] = unshaped_y[i_2,j,k]

    print('X shape: '+str(X.shape)+', y shape: '+str(y.shape)+', X times shape: '+str(X_times.shape)+', y times shape: '+str(y_times.shape))
    X_spacecraft_positions, y_spacecraft_positions = X[0:,0:,0:3], y[0:,0:,0:3]
    X, y = X[0:,0:,3:], y[0:,0:,3:]

    # remove rows containing nans from dataset
    nan_rows = np.zeros(y.shape[0],dtype=bool)

    for i in range(X.shape[0]):
        for k in range(X.shape[2]):
            for j in range(X.shape[1]):
                if X[i,j,k] != X[i,j,k]:
                    nan_rows[i] = True
                    #print('NaN value: '+str(X[i,j,k]))
            for j_2 in range (y.shape[1]):
                if y[i,j_2,k] != y[i,j_2,k]:
                    nan_rows[i] = True
    print(X.shape,y.shape)

    X_no_nans = X[~nan_rows,:,:]
    y_no_nans = y[~nan_rows,:,:]
    X_positions_no_nans = X_spacecraft_positions[~nan_rows,:,:]
    y_positions_no_nans = y_spacecraft_positions[~nan_rows,:,:]
    print(X_times.shape,nan_rows.shape)
    X_no_nans_times = X_times[~nan_rows]
    y_no_nans_times = y_times[~nan_rows]
    print(X_no_nans_times.shape, y_no_nans_times.shape)

#     # the dataset knows the number of features, e.g. 2
#     X_no_nans_reshaped = X_no_nans.reshape(X_no_nans.shape[0],-1)
#     y_no_nans_reshaped = y_no_nans.reshape(y_no_nans.shape[0],-1)
#     X_positions_no_nans_reshaped = X_positions_no_nans.reshape(X_positions_no_nans.shape[0],-1)
#     y_positions_no_nans_reshaped = y_positions_no_nans.reshape(y_positions_no_nans.shape[0],-1)
#     np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_no_nans.txt',X_no_nans_reshaped)
#     np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_no_nans.txt',y_no_nans_reshaped)
#     np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_positions_no_nans.txt',X_positions_no_nans_reshaped)
#     np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_positions_no_nans.txt',y_positions_no_nans_reshaped)
#     np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_no_nans_times.txt',X_no_nans_times,fmt='%s')
#     np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_no_nans_times.txt',y_no_nans_times,fmt='%s')
# if make_dataset == False:
#     X_no_nans_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_no_nans.txt')
#     y_no_nans_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_no_nans.txt')
#     X_positions_no_nans_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_positions_no_nans.txt')
#     y_positions_no_nans_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_positions_no_nans.txt')
#     X_no_nans_times = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_no_nans_times.txt',dtype='U')
#     y_no_nans_times = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_no_nans_times.txt',dtype='U')
#
#     X_no_nans = X_no_nans_reshaped.reshape(
#         X_no_nans_reshaped.shape[0], X_no_nans_reshaped.shape[1] // n_features, n_features)
#     y_no_nans = y_no_nans_reshaped.reshape(
#         y_no_nans_reshaped.shape[0], y_no_nans_reshaped.shape[1] // n_features, n_features)
#     X_positions_no_nans = X_positions_no_nans_reshaped.reshape(
#         X_positions_no_nans_reshaped.shape[0], X_positions_no_nans_reshaped.shape[1] // n_positions, n_positions)
#     y_positions_no_nans = y_positions_no_nans_reshaped.reshape(
#         y_positions_no_nans_reshaped.shape[0], y_positions_no_nans_reshaped.shape[1] // n_positions, n_positions)
#
#
# # split dataset into train and test subsets
# X_train_raw, X_test_raw, y_train_raw, y_test_raw, X_train_times, X_test_times, y_train_times, y_test_times, X_train_positions, X_test_positions, y_train_positions, y_test_positions = train_test_split(X_no_nans,y_no_nans,X_no_nans_times,y_no_nans_times,X_positions_no_nans,y_positions_no_nans,test_size=0.35)
# print(X_train_raw.shape,y_train_raw.shape,X_test_raw.shape,y_test_raw.shape)
#
# X_train_reshaped = X_train_raw.reshape(X_train_raw.shape[0],-1)
# y_train_reshaped = y_train_raw.reshape(y_train_raw.shape[0],-1)
# X_test_reshaped = X_test_raw.reshape(X_test_raw.shape[0],-1)
# y_test_reshaped = y_test_raw.reshape(y_test_raw.shape[0],-1)
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_train.txt',X_train_reshaped)
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_train.txt',y_train_reshaped)
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_test.txt',X_test_reshaped)
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_test.txt',y_test_reshaped)
#
# X_train_positions_reshaped = X_train_positions.reshape(X_train_positions.shape[0],-1)
# y_train_positions_reshaped = y_train_positions.reshape(y_train_positions.shape[0],-1)
# X_test_positions_reshaped = X_test_positions.reshape(X_test_positions.shape[0],-1)
# y_test_positions_reshaped = y_test_positions.reshape(y_test_positions.shape[0],-1)
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_train_positions.txt',X_train_positions_reshaped)
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_train_positions.txt',y_train_positions_reshaped)
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_test_positions.txt',X_test_positions_reshaped)
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_test_positions.txt',y_test_positions_reshaped)
#
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_train_times.txt',X_train_times,fmt='%s')
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_test_times.txt',X_test_times,fmt='%s')
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_train_times.txt',y_train_times,fmt='%s')
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_test_times.txt',y_test_times,fmt='%s')
# # if make_dataset == False:
# #     print('no dataset to make')

print('Done')
# if make_dataset == False:
# 	y_train_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_y_train.txt')
# 	y_test_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_y_test.txt')
# 	X_train_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_x_train.txt')
# 	X_test_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_x_test.txt')
# 	y_test_raw = y_test_raw_reshaped.reshape(
# 	    y_test_raw_reshaped.shape[0], y_test_raw_reshaped.shape[1] // n_features, n_features)
# 	y_train_raw = y_train_raw_reshaped.reshape(
# 	    y_train_raw_reshaped.shape[0], y_train_raw_reshaped.shape[1] // n_features, n_features)
# 	X_test_raw = X_test_raw_reshaped.reshape(
# 	    X_test_raw_reshaped.shape[0], X_test_raw_reshaped.shape[1] // n_features, n_features)
# 	X_train_raw = X_train_raw_reshaped.reshape(
# 	    X_train_raw_reshaped.shape[0], X_train_raw_reshaped.shape[1] // n_features, n_features)
# 	X_train, X_test, y_train, y_test = standardise(X_train_raw,X_test_raw,y_train_raw,y_test_raw)
# 	print(X_test.shape, y_test.shape)
#
# # separate output
# y1_train = y_train[:, :, 0]
# y2_train = y_train[:, :, 1]
# y3_train = y_train[:, :, 2]
# y4_train = y_train[:, :, 3]
# y5_train = y_train[:, :, 4]
# y6_train = y_train[:, :, 5]
# y7_train = y_train[:, :, 6]
# y8_train = y_train[:, :, 7]
# y9_train = y_train[:, :, 8]
# # y10_train = y_train[:, :, 9]
# # y11_train = y_train[:, :, 10]
# # y12_train = y_train[:, :, 11]
# # y13_train = y_train[:, :, 12]
# # y14_train = y_train[:, :, 13]
# # y15_train = y_train[:, 14].reshape((y_train.shape[0], 1))
# # y16_train = y_train[:, 15].reshape((y_train.shape[0], 1))
# # y17_train = y_train[:, 16].reshape((y_train.shape[0], 1))
#
# y1_test = y_test[:, :, 0]
# y2_test = y_test[:, :, 1]
# y3_test = y_test[:, :, 2]
# y4_test = y_test[:, :, 3]
# y5_test = y_test[:, :, 4]
# y6_test = y_test[:, :, 5]
# y7_test = y_test[:, :, 6]
# y8_test = y_test[:, :, 7]
# y9_test = y_test[:, :, 8]
# # y10_test = y_test[:, :, 9]
# # y11_test = y_test[:, :, 10]
# # y12_test = y_test[:, :, 11]
# # y13_test = y_test[:, :, 12]
# # y14_test = y_test[:, :, 13]
# # y15_test = y_test[:, 14].reshape((y_test.shape[0], 1))
# # y16_test = y_test[:, 15].reshape((y_test.shape[0], 1))
# # y17_test = y_test[:, 16].reshape((y_test.shape[0], 1))
#
# # import pre-made model trained on omni data
# omni_trained_cnn = load_model(r'D:\forecasting_study\omni_multistep_10_hour__3_hour_input_before_2_hour_input_after_trained_9_features',compile=False)
#
# # trained_model = clone_model(omni_trained_cnn)
# # trained_model.set_weights(omni_trained_cnn.get_weights())
# # trained_model.trainable = False
# # for layer in trained_model.layers:
# # 	layer._name = layer._name + str('layer_0')
# #
# # # add a few extra trainable layers
# # cnn = Dense(50, activation='relu')(trained_model.layers[-3].output)
# # cnn = Dense(50, activation='softmax')(cnn)
# #
# # output_1 = Dense(4)(cnn)
# # # define output 2
# # output_2 = Dense(4)(cnn)
# # # define output 3
# # output_3 = Dense(4)(cnn)
# # # define output 1
# # output_4 = Dense(4)(cnn)
# # # define output 2
# # output_5 = Dense(4)(cnn)
# # # define output 3
# # output_6 = Dense(4)(cnn)
# # # define output 1
# # output_7 = Dense(4)(cnn)
# # # define output 2
# # output_8 = Dense(4)(cnn)
# # # define output 3
# # output_9 = Dense(4)(cnn)
# #
# # model = Model(inputs=trained_model.input, outputs=[output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9])
# # model.compile(optimizer='adam', loss='mse')
# #
# # model.fit(X_train, [y1_train,y2_train,y3_train,y4_train,y5_train,y6_train,y7_train,y8_train,y9_train], epochs=500, verbose=0)
#
# yhat = omni_trained_cnn.predict(X_test, verbose=True)
# yhat_array = np.array(yhat)
# print('yhat_array.shape: '+str(yhat_array.shape))
# yhat_array = rearrange_axes(yhat_array)
# print('yhat_array.shape: '+str(yhat_array.shape))
# yhat_array_units = destandardise(yhat_array,X_train_raw)
# print('yhat_array.shape, y_test.shape: '+str(yhat_array.shape)+' , '+str(y_test.shape))
# yhat_array_units_reshaped = yhat_array_units.reshape(yhat_array_units.shape[0], -1)
# y_test_raw_reshaped = y_test_raw.reshape(y_test_raw.shape[0],36)
# y_train_raw_reshaped = y_train_raw.reshape(y_train_raw.shape[0],36)
# X_test_raw_reshaped = X_test_raw.reshape(X_test_raw.shape[0],45)
# X_train_raw_reshaped = X_train_raw.reshape(X_train_raw.shape[0],45)
#
# print(X_train_raw_reshaped.shape,y_train_raw_reshaped.shape,X_test_raw_reshaped.shape,y_test_raw_reshaped.shape)
#
#
#
# np.savetxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_yhat.txt',yhat_array_units_reshaped)
