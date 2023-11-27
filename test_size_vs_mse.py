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
import warnings
warnings.filterwarnings("ignore")

# define useful functions

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

n_features = 9

names =  names = ['BX (nT GSE/GSM)','BY (nT GSE)','BZ (nT GSE)','Vx Velocity (km/s)','Vy Velocity (km/s)','Vz Velocity (km/s)','Proton Density (n/cc)''Proton Temperature (eV)','Flow Pressure (nPa)']

maven_yhat_0_05_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_05.txt')
y_test_0_05_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_05.txt')
maven_yhat_0_10_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_10.txt')
y_test_0_10_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_10.txt')

maven_yhat_0_15_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_15.txt')
y_test_0_15_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_15.txt')
maven_yhat_0_20_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_20.txt')
y_test_0_20_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_20.txt')

maven_yhat_0_25_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_25.txt')
y_test_0_25_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_25.txt')
maven_yhat_0_30_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_30.txt')
y_test_0_30_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_30.txt')

maven_yhat_0_35_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_35.txt')
y_test_0_35_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_35.txt')
maven_yhat_0_40_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_40.txt')
y_test_0_40_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_40.txt')

maven_yhat_0_45_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_45.txt')
y_test_0_45_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_45.txt')
maven_yhat_0_50_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_yhat_test_size_0_50.txt')
y_test_0_50_reshaped = np.loadtxt(r'D:\forecasting_study\maven_only_multistep_cnn_3_hour_input_before_2_hour_input_after_y_test_test_size_0_50.txt')

maven_yhat_0_05 = maven_yhat_0_05_reshaped.reshape(maven_yhat_0_05_reshaped.shape[0], maven_yhat_0_05_reshaped.shape[1] // n_features, n_features)
y_test_0_05 = y_test_0_05_reshaped.reshape(
    y_test_0_05_reshaped.shape[0], y_test_0_05_reshaped.shape[1] // n_features, n_features)

maven_yhat_0_10 = maven_yhat_0_10_reshaped.reshape(maven_yhat_0_10_reshaped.shape[0], maven_yhat_0_10_reshaped.shape[1] // n_features, n_features)
y_test_0_10 = y_test_0_10_reshaped.reshape(
    y_test_0_10_reshaped.shape[0], y_test_0_10_reshaped.shape[1] // n_features, n_features)

maven_yhat_0_15 = maven_yhat_0_15_reshaped.reshape(maven_yhat_0_15_reshaped.shape[0], maven_yhat_0_15_reshaped.shape[1] // n_features, n_features)
y_test_0_15 = y_test_0_15_reshaped.reshape(
    y_test_0_15_reshaped.shape[0], y_test_0_15_reshaped.shape[1] // n_features, n_features)

maven_yhat_0_20 = maven_yhat_0_20_reshaped.reshape(maven_yhat_0_20_reshaped.shape[0], maven_yhat_0_20_reshaped.shape[1] // n_features, n_features)
y_test_0_20 = y_test_0_20_reshaped.reshape(
    y_test_0_20_reshaped.shape[0], y_test_0_20_reshaped.shape[1] // n_features, n_features)

maven_yhat_0_25 = maven_yhat_0_25_reshaped.reshape(maven_yhat_0_25_reshaped.shape[0], maven_yhat_0_25_reshaped.shape[1] // n_features, n_features)
y_test_0_25 = y_test_0_25_reshaped.reshape(
    y_test_0_25_reshaped.shape[0], y_test_0_25_reshaped.shape[1] // n_features, n_features)

maven_yhat_0_30 = maven_yhat_0_30_reshaped.reshape(maven_yhat_0_30_reshaped.shape[0], maven_yhat_0_30_reshaped.shape[1] // n_features, n_features)
y_test_0_30 = y_test_0_30_reshaped.reshape(
    y_test_0_30_reshaped.shape[0], y_test_0_30_reshaped.shape[1] // n_features, n_features)

maven_yhat_0_35 = maven_yhat_0_35_reshaped.reshape(maven_yhat_0_35_reshaped.shape[0], maven_yhat_0_35_reshaped.shape[1] // n_features, n_features)
y_test_0_35 = y_test_0_35_reshaped.reshape(
    y_test_0_35_reshaped.shape[0], y_test_0_35_reshaped.shape[1] // n_features, n_features)

maven_yhat_0_40 = maven_yhat_0_40_reshaped.reshape(maven_yhat_0_40_reshaped.shape[0], maven_yhat_0_40_reshaped.shape[1] // n_features, n_features)
y_test_0_40 = y_test_0_40_reshaped.reshape(
    y_test_0_40_reshaped.shape[0], y_test_0_40_reshaped.shape[1] // n_features, n_features)

maven_yhat_0_45 = maven_yhat_0_45_reshaped.reshape(maven_yhat_0_45_reshaped.shape[0], maven_yhat_0_45_reshaped.shape[1] // n_features, n_features)
y_test_0_45 = y_test_0_45_reshaped.reshape(
    y_test_0_45_reshaped.shape[0], y_test_0_45_reshaped.shape[1] // n_features, n_features)

maven_yhat_0_50 = maven_yhat_0_50_reshaped.reshape(maven_yhat_0_50_reshaped.shape[0], maven_yhat_0_50_reshaped.shape[1] // n_features, n_features)
y_test_0_50 = y_test_0_50_reshaped.reshape(
    y_test_0_50_reshaped.shape[0], y_test_0_50_reshaped.shape[1] // n_features, n_features)

maven_diff_0_05 = y_test_0_05-maven_yhat_0_05
maven_mse_0_05 = np.mean((maven_diff_0_05)*(maven_diff_0_05))

maven_diff_0_10 = y_test_0_10-maven_yhat_0_10
maven_mse_0_10 = np.mean((maven_diff_0_10)*(maven_diff_0_10))

maven_diff_0_15 = y_test_0_15-maven_yhat_0_15
maven_mse_0_15 = np.mean((maven_diff_0_15)*(maven_diff_0_15))

maven_diff_0_20 = y_test_0_20-maven_yhat_0_20
maven_mse_0_20 = np.mean((maven_diff_0_20)*(maven_diff_0_20))

maven_diff_0_25 = y_test_0_25-maven_yhat_0_25
maven_mse_0_25 = np.mean((maven_diff_0_25)*(maven_diff_0_25))

maven_diff_0_30 = y_test_0_30-maven_yhat_0_30
maven_mse_0_30 = np.mean((maven_diff_0_30)*(maven_diff_0_30))

maven_diff_0_35 = y_test_0_35-maven_yhat_0_35
maven_mse_0_35 = np.mean((maven_diff_0_35)*(maven_diff_0_35))

maven_diff_0_40 = y_test_0_40-maven_yhat_0_40
maven_mse_0_40 = np.mean((maven_diff_0_40)*(maven_diff_0_40))

maven_diff_0_45 = y_test_0_45-maven_yhat_0_45
maven_mse_0_45 = np.mean((maven_diff_0_45)*(maven_diff_0_45))

maven_diff_0_50 = y_test_0_50-maven_yhat_0_50
maven_mse_0_50 = np.mean((maven_diff_0_50)*(maven_diff_0_50))

x = (np.arange(10)+1)*5
y = [maven_mse_0_05,maven_mse_0_10,maven_mse_0_15,maven_mse_0_20,maven_mse_0_25,maven_mse_0_30,maven_mse_0_35,maven_mse_0_40,maven_mse_0_45,maven_mse_0_50]

fig, axs = plt.subplots(1,tight_layout=True)
axs.plot(x,y,color=cm.Greys(140))
axs.set_xlim(xmin=5,xmax=50)
axs.set_xlabel('Test set size (% of total data)')
axs.set_ylabel('MSE')
plt.show()
