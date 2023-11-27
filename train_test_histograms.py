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

r_m = 3389
dtor = math.pi/180.
x_0 = 0.64
ecc = 1.02
L = 2.06
k=0

n_features = 9

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

train = np.concatenate((X_train_raw,y_train_raw),axis=1)
test = np.concatenate((X_test_raw,y_test_raw),axis=1)

fig, axs = plt.subplots(9,2,tight_layout=True)
axs[0][0].hist(train[0:,0:,0].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[0][0].set_xlabel('B_x (nT)')
axs[1][0].hist(train[0:,0:,1].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[1][0].set_xlabel('B_y (nT)')
axs[2][0].hist(train[0:,0:,2].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[2][0].set_xlabel('B_z (nT)')
axs[3][0].hist(train[0:,0:,3].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-600,30])
axs[3][0].set_xlabel('v_x (km/s)')
axs[4][0].hist(train[0:,0:,4].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-100,110])
axs[4][0].set_xlabel('v_y (km/s)')
axs[5][0].hist(train[0:,0:,5].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-100,110])
axs[5][0].set_xlabel('v_z (km/s)')
axs[6][0].hist(train[0:,0:,6].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,12])
axs[6][0].set_xlabel('n_p (cm^-3)')
axs[7][0].hist(train[0:,0:,7].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,350])
axs[7][0].set_xlabel('T_p (eV)')
axs[8][0].hist(train[0:,0:,8].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,2])
axs[8][0].set_xlabel('P_dyn (nPa)')

axs[0][1].hist(test[0:,0:,0].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[0][1].set_xlabel('B_x (nT)')
axs[1][1].hist(test[0:,0:,1].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[1][1].set_xlabel('B_y (nT)')
axs[2][1].hist(test[0:,0:,2].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[2][1].set_xlabel('B_z (nT)')
axs[3][1].hist(test[0:,0:,3].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-600,30])
axs[3][1].set_xlabel('v_x (km/s)')
axs[4][1].hist(test[0:,0:,4].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-100,110])
axs[4][1].set_xlabel('v_y (km/s)')
axs[5][1].hist(test[0:,0:,5].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-100,110])
axs[5][1].set_xlabel('v_z (km/s)')
axs[6][1].hist(test[0:,0:,6].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,12])
axs[6][1].set_xlabel('n_p (cm^-3)')
axs[7][1].hist(test[0:,0:,7].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,350])
axs[7][1].set_xlabel('T_p (eV)')
axs[8][1].hist(test[0:,0:,8].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,2])
axs[8][1].set_xlabel('P_dyn (nPa)')
plt.show()

fig, axs = plt.subplots(3,2,tight_layout=True)
axs[0][0].hist(train[0:,0:,0].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[0][0].set_xlabel('B_x (nT)')
axs[1][0].hist(train[0:,0:,1].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[1][0].set_xlabel('B_y (nT)')
axs[1][0].set_ylabel('Frequency density')
axs[2][0].hist(train[0:,0:,2].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[2][0].set_xlabel('B_z (nT)')
axs[0][1].hist(test[0:,0:,0].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[0][1].set_xlabel('B_x (nT)')
axs[1][1].hist(test[0:,0:,1].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[1][1].set_xlabel('B_y (nT)')
axs[2][1].hist(test[0:,0:,2].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-10,10])
axs[2][1].set_xlabel('B_z (nT)')
plt.show()

fig, axs = plt.subplots(3,2,tight_layout=True)
axs[0][0].hist(train[0:,0:,3].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-600,30])
axs[0][0].set_xlabel('v_x (km/s)')
axs[1][0].hist(train[0:,0:,4].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-100,110])
axs[1][0].set_xlabel('v_y (km/s)')
axs[1][0].set_ylabel('Frequency density')
axs[2][0].hist(train[0:,0:,5].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-100,110])
axs[2][0].set_xlabel('v_z (km/s)')

axs[0][1].hist(test[0:,0:,3].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-600,30])
axs[0][1].set_xlabel('v_x (km/s)')
axs[1][1].hist(test[0:,0:,4].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-100,110])
axs[1][1].set_xlabel('v_y (km/s)')
axs[2][1].hist(test[0:,0:,5].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[-100,110])
axs[2][1].set_xlabel('v_z (km/s)')
plt.show()

fig, axs = plt.subplots(3,2,tight_layout=True)
axs[0][0].hist(train[0:,0:,6].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,12])
axs[0][0].set_xlabel('n_p (cm^-3)')
axs[1][0].hist(train[0:,0:,7].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,350])
axs[1][0].set_xlabel('T_p (eV)')
axs[1][0].set_ylabel('Frequency density')
axs[2][0].hist(train[0:,0:,8].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,2])
axs[2][0].set_xlabel('P_dyn (nPa)')

axs[0][1].hist(test[0:,0:,6].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,12])
axs[0][1].set_xlabel('n_p (cm^-3)')
axs[1][1].hist(test[0:,0:,7].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,350])
axs[1][1].set_xlabel('T_p (eV)')
axs[2][1].hist(test[0:,0:,8].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.Greys(130),range=[0,2])
axs[2][1].set_xlabel('P_dyn (nPa)')
plt.show()

fig, axs = plt.subplots(3,3,tight_layout=True)
axs[0][0].hist(train[0:,0:,0].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.plasma(0),alpha=0.7,range=[-10,10])
axs[0][0].hist(test[0:,0:,0].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.plasma(100),alpha=0.7,range=[-10,10])
axs[0][0].set_xlabel('B_x (nT)')
axs[0][1].hist(train[0:,0:,1].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.plasma(0),alpha=0.7,range=[-10,10])
axs[0][1].hist(test[0:,0:,1].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.plasma(100),alpha=0.7,range=[-10,10])
axs[0][1].set_xlabel('B_y (nT)')
axs[0][2].hist(train[0:,0:,2].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.plasma(0),alpha=0.7,range=[-10,10])
axs[0][2].hist(test[0:,0:,2].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.plasma(100),alpha=0.7,range=[-10,10])
axs[0][2].set_xlabel('B_z (nT)')
axs[1][0].hist(train[0:,0:,3].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.plasma(0),alpha=0.7,range=[-600,30])
axs[1][0].hist(test[0:,0:,3].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.plasma(100),alpha=0.7,range=[-600,30])
axs[1][0].set_xlabel('v_x (km/s)')
axs[1][1].hist(train[0:,0:,4].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.plasma(0),alpha=0.7,range=[-100,110])
axs[1][1].hist(test[0:,0:,4].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.plasma(100),alpha=0.7,range=[-100,110])
axs[1][1].set_xlabel('v_y (km/s)')
axs[1][2].hist(train[0:,0:,5].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.plasma(0),alpha=0.7,range=[-100,110])
axs[1][2].hist(test[0:,0:,5].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.plasma(100),alpha=0.7,range=[-100,110])
axs[1][2].set_xlabel('v_z (km/s)')
axs[2][0].hist(train[0:,0:,6].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.plasma(0),alpha=0.7,range=[0,12])
axs[2][0].hist(test[0:,0:,6].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.plasma(100),alpha=0.7,range=[0,12])
axs[2][0].set_xlabel('n_p (cm^-3)')
axs[2][1].hist(train[0:,0:,7].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.plasma(0),alpha=0.7,range=[0,350])
axs[2][1].hist(test[0:,0:,7].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.plasma(100),alpha=0.7,range=[0,350])
axs[2][1].set_xlabel('T_p (eV)')
axs[2][2].hist(train[0:,0:,8].reshape(9*train.shape[0]),bins=50,density=True,fc=cm.plasma(0),alpha=0.7,range=[0,2])
axs[2][2].hist(test[0:,0:,8].reshape(9*test.shape[0]),bins=50,density=True,fc=cm.plasma(100),alpha=0.7,range=[0,2])
axs[2][2].set_xlabel('P_dyn (nPa)')

plt.show()
