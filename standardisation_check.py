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

def standardise(X_train,X_test,y_train,y_test):
	X_mean = np.mean(X_train,axis=(0,1))
	print(X_mean)
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

n_features=9

yhat_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_yhat.txt')
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

X_mean = np.mean(X_train_raw,axis=(0,1))
print(X_mean)

X_train, X_test, y_train, y_test = standardise(X_train_raw, X_test_raw, y_train_raw, y_test_raw)
