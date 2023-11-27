import math
import numpy as np
from numpy import array
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure, show, rc
from matplotlib.dates import DateFormatter
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
from datetime import datetime
import datetime as dt

def remove_magnetosphere_measurements(array):
    sw_array = array.copy()
    #print('array shape before magnetosphere removements '+str(sw_array.shape))
    for i in range(array.shape[0]):
        theta = np.arctan2((array[i,0]-x_0),np.sqrt(array[i,1]*array[i,1]+array[i,2]*array[i,2]))
        if (np.sqrt((array[i,0]-x_0)*(array[i,0]-x_0)+array[i,2]*array[i,2]+array[i,1]*array[i,1])) < (L/(1+ecc*np.cos(theta))):
            sw_array[0:] = np.NaN
    return sw_array


r_m = 3389
dtor = math.pi/180.
x_0 = 0.64
ecc = 1.02
L = 2.06
k=0

n_features = 9
n_positions = 3

y_train_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_train_positions.txt')
y_test_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_test_positions.txt')
X_train_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_train_positions.txt')
X_test_raw_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_test_positions.txt')
y_test_raw = y_test_raw_reshaped.reshape(
    y_test_raw_reshaped.shape[0], y_test_raw_reshaped.shape[1] // n_positions, n_positions)
y_train_raw = y_train_raw_reshaped.reshape(
    y_train_raw_reshaped.shape[0], y_train_raw_reshaped.shape[1] // n_positions, n_positions)
X_test_raw = X_test_raw_reshaped.reshape(
    X_test_raw_reshaped.shape[0], X_test_raw_reshaped.shape[1] // n_positions, n_positions)
X_train_raw = X_train_raw_reshaped.reshape(
    X_train_raw_reshaped.shape[0], X_train_raw_reshaped.shape[1] // n_positions, n_positions)
print(X_train_raw.shape,y_test_raw.shape)


train_positions = np.concatenate((X_train_raw,y_train_raw),axis=1)
test_positions = np.concatenate((X_test_raw,y_test_raw),axis=1)



x_train_positions = train_positions[0:,0:,0]
rho_train_positions = np.zeros((train_positions.shape[0],train_positions.shape[1],1))
x_test_positions = test_positions[0:,0:,0]
rho_test_positions = np.zeros((test_positions.shape[0],test_positions.shape[1],1))
for j in range(rho_train_positions.shape[1]):
    for i in range(rho_train_positions.shape[0]):
        rho_train_positions[i][j][0] = np.sqrt((train_positions[i][j][1]*train_positions[i][j][1]+train_positions[i][j][2]*train_positions[i][j][2]))
    for i in range(rho_test_positions.shape[0]):
        rho_test_positions[i][j][0] = np.sqrt((test_positions[i][j][1]*test_positions[i][j][1]+test_positions[i][j][2]*test_positions[i][j][2]))
# x_train_positions = x_train_positions.reshape(x_train_positions.shape[0]*x_train_positions.shape[1])
# rho_train_positions = rho_train_positions.reshape(rho_train_positions.shape[0]*rho_train_positions.shape[1])
# x_test_positions = x_test_positions.reshape(x_test_positions.shape[0]*x_test_positions.shape[1])
# rho_test_positions = rho_test_positions.reshape(rho_test_positions.shape[0]*rho_test_positions.shape[1])
# print(x_train_positions.shape,x_test_positions.shape)
# train_nan_rows = np.zeros(x_train_positions.shape[0],dtype=bool)
# test_nan_rows = np.zeros(x_test_positions.shape[0],dtype=bool)
#
# for i in range(train_nan_rows.shape[0]):
#     theta = np.arctan2((x_train_positions[i]-x_0),rho_train_positions[i])
#     if (np.sqrt((x_train_positions[i]-x_0)*(x_train_positions[i]-x_0)+rho_train_positions[i]*rho_train_positions[i])) < (L/(1+ecc*np.cos(theta))):
#         train_nan_rows[i] = True
#
# for i in range(test_nan_rows.shape[0]):
#     theta = np.arctan2((x_test_positions[i]-x_0),rho_test_positions[i])
#     if (np.sqrt((x_test_positions[i]-x_0)*(x_test_positions[i]-x_0)+rho_test_positions[i]*rho_test_positions[i])) < (L/(1+ecc*np.cos(theta))):
#         train_nan_rows[i] = True
#
# x_train_positions = x_train_positions[~train_nan_rows,]
# x_test_positions = x_test_positions[~test_nan_rows,]
# rho_train_positions = rho_train_positions[~train_nan_rows,]
# rho_test_positions = rho_test_positions[~test_nan_rows,]
# print(x_train_positions.shape,x_test_positions.shape)

angles = np.arange(360)*dtor*0.5
mars_x = np.zeros(len(angles))
mars_rho = np.zeros(len(angles))
bowshock_x = np.zeros(len(angles))
bowshock_rho = np.zeros(len(angles))

for i in range(len(angles)):
    mars_x[i] = np.cos(angles[i])
    mars_rho[i] = np.sin(angles[i])
    r = L/(1+ecc*np.cos(angles[i]))
    bowshock_x[i] = x_0+r*np.cos(angles[i])
    bowshock_rho[i] = r*np.sin(angles[i])


print(x_train_positions.shape,rho_test_positions.shape)

fig, axs = plt.subplots(2,1,tight_layout=True)
train = axs[0].hist2d(x_train_positions[0:,0],rho_train_positions[0:,0],density=True,cmap='RdPu')
fig.colorbar(train[3],ax=axs[0])
axs[0].plot(mars_x,mars_rho,c=cm.Greys(254))
axs[0].plot(bowshock_x,bowshock_rho,c=cm.Greys(254))
axs[0].set_xlabel('X (R_M)')
axs[0].set_ylabel('Rho (R_M)')

test = axs[1].hist2d(x_test_positions[0:,0],rho_test_positions[0:,0],density=True,cmap='RdPu')
fig.colorbar(test[3],ax=axs[1])
axs[1].plot(mars_x,mars_rho,c=cm.Greys(254))
axs[1].plot(bowshock_x,bowshock_rho,c=cm.Greys(254))
axs[1].set_xlabel('X (R_M)')
axs[1].set_ylabel('Rho (R_M)')
plt.show()
