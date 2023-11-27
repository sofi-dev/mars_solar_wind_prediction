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

def convert_to_datetime(array):
    datetime_array = np.empty((array.shape[0],array.shape[1]),dtype=datetime)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            print(array[i][j])
            datetime_array[i][j] = datetime.strptime(array[i][j],'%Y-%m-%dT%H:%M:%S.000000000')
    return datetime_array

n_features = 9
i=58
#i=92

names =  names = ['BX (nT GSE/GSM)','BY (nT GSE)','BZ (nT GSE)','Vx Velocity (km/s)','Vy Velocity (km/s)','Vz Velocity (km/s)','Proton Density (n/cc)''Proton Temperature (eV)','Flow Pressure (nPa)']

#
yhat_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_yhat.txt')
y_test_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_test.txt')

X_train_times = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_train_times.txt',dtype='U')
X_test_times = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_test_times.txt',dtype='U')
y_train_times = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_train_times.txt',dtype='U')
y_test_times = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_test_times.txt',dtype='U')
y_train_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_y_train.txt')
X_train_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_train.txt')
X_test_reshaped = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_x_test.txt')
mae = np.loadtxt(r'D:\forecasting_study\maven_multistep_cnn_3_hour_input_before_2_hour_input_after_9_features_30_min_mse.txt')

yhat = yhat_reshaped.reshape(yhat_reshaped.shape[0], yhat_reshaped.shape[1] // n_features, n_features)
y_test = y_test_reshaped.reshape(
    y_test_reshaped.shape[0], y_test_reshaped.shape[1] // n_features, n_features)
y_train = y_train_reshaped.reshape(
    y_train_reshaped.shape[0], y_train_reshaped.shape[1] // n_features, n_features)
X_test = X_test_reshaped.reshape(
    X_test_reshaped.shape[0], X_test_reshaped.shape[1] // n_features, n_features)
X_train = X_train_reshaped.reshape(
    X_train_reshaped.shape[0], X_train_reshaped.shape[1] // n_features, n_features)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_train_times.shape,y_train_times.shape,X_test_times.shape,y_test_times.shape)
X_train,y_train,X_test,y_test,X_train_times,y_train_times,X_test_times,y_test_times = X_train[250:500],y_train[250:500],X_test[250:500],y_test[250:500],X_train_times[250:500],y_train_times[250:500],X_test_times[250:500],y_test_times[250:500]
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape,X_train_times.shape,y_train_times.shape,X_test_times.shape,y_test_times.shape)
X_train_times = convert_to_datetime(X_train_times)
X_test_times = convert_to_datetime(X_test_times)
y_train_times = convert_to_datetime(y_train_times)
y_test_times = convert_to_datetime(y_test_times)

name = r'D:\maven\data\sci\kp\insitu\data\2019\02\mvn_kp_insitu_20190221_v15_r02.tab'
with open(name,) as csvfile:
    #print(name)
    file = csv.reader(csvfile)
    j= 0
    for row in file:
        #print('row = '+str(i))
        if j == 8:
            begin_row = row[0]
            begin_row = int(begin_row[6:9])-1
            #print('begin row: '+str(begin_row))
        elif j == 9:
            n_rows = row[0]
            n_rows = int(n_rows[4:9])
            #print(n_rows)
        elif j > 9:
            break
        j+=1

df = pd.read_table(name,sep='\s+',names=['Datetime','Proton Density (n/cc)','Vx Velocity (km/s)','Vy Velocity (km/s)','Vz Velocity (km/s)','Proton Temperature (eV)','Dynamic Pressure (nPa)','BX (nT GSE/GSM)','BY (nT GSE)','BZ (nT GSE)','Spacecraft MSO X (km)','Spacecraft MSO Y (km)','Spacecraft MSO Z (km)'],index_col=False,usecols=[0,40,42,44,46,48,50,127,129,131,189,190,191],na_values=[999.99,9999.99,99999.9,99999.99,9999999.],skiprows=begin_row,nrows=n_rows)[['Datetime','Spacecraft MSO X (km)','Spacecraft MSO Y (km)','Spacecraft MSO Z (km)','BX (nT GSE/GSM)','BY (nT GSE)','BZ (nT GSE)','Vx Velocity (km/s)','Vy Velocity (km/s)','Vz Velocity (km/s)','Proton Density (n/cc)','Proton Temperature (eV)','Dynamic Pressure (nPa)']]

df['Spacecraft MSO X (km)'] = df['Spacecraft MSO X (km)']/r_m
df['Spacecraft MSO Y (km)'] = df['Spacecraft MSO Y (km)']/r_m
df['Spacecraft MSO Z (km)'] = df['Spacecraft MSO Z (km)']/r_m

df.set_index('Datetime',inplace=True,drop=True)
#print(df.index)
df.index= pd.to_datetime(df.index,format='%Y-%m-%dT%H:%M:%S')
#    df = df.resample('30T').mean()
print(df)
date_format = DateFormatter('%d-%m-%y %H:%M')

fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9) = plt.subplots(9,1,tight_layout=True,sharex=True)
ax1.plot(df.index,df['BX (nT GSE/GSM)'],c=cm.Greys(130))
ax1.errorbar(y_test_times[i],yhat[i,0:,0],yerr=mae[0],c=cm.plasma(130),label='model predictions')
ax1.set_ylabel('B_x',rotation='horizontal',labelpad=15.0)

ax2.plot(df.index,df['BY (nT GSE)'],c=cm.Greys(130))
ax2.errorbar(y_test_times[i],yhat[i,0:,1],yerr=mae[1],c=cm.plasma(130))
ax2.set_ylabel('B_y',rotation='horizontal',labelpad=15.0)

ax3.plot(df.index,df['BZ (nT GSE)'],c=cm.Greys(130))
ax3.errorbar(y_test_times[i],yhat[i,0:,2],yerr=mae[2],c=cm.plasma(130))
ax3.set_ylabel('B_z',rotation='horizontal',labelpad=15.0)

ax4.plot(df.index,df['Vx Velocity (km/s)'],c=cm.Greys(130))
ax4.errorbar(y_test_times[i],yhat[i,0:,3],yerr=mae[3],c=cm.plasma(130))
ax4.set_ylabel('v_x',rotation='horizontal',labelpad=15.0)

ax5.plot(df.index,df['Vy Velocity (km/s)'],c=cm.Greys(130))
ax5.errorbar(y_test_times[i],yhat[i,0:,4],yerr=mae[4],c=cm.plasma(130))
ax5.set_ylabel('v_y',rotation='horizontal',labelpad=15.3)

ax6.plot(df.index,df['Vz Velocity (km/s)'],c=cm.Greys(130))
ax6.errorbar(y_test_times[i],yhat[i,0:,5],yerr=mae[5],c=cm.plasma(130))
ax6.set_ylabel('v_z',rotation='horizontal',labelpad=15.0)

ax7.plot(df.index,df['Proton Density (n/cc)'],c=cm.Greys(130))
ax7.errorbar(y_test_times[i],yhat[i,0:,6],yerr=mae[6],c=cm.plasma(130))
ax7.set_ylabel('n_p',rotation='horizontal',labelpad=15.0)

ax8.plot(df.index,df['Proton Temperature (eV)'],c=cm.Greys(130))
ax8.errorbar(y_test_times[i],yhat[i,0:,7],yerr=mae[7],c=cm.plasma(130))
ax8.set_ylabel('T_p',rotation='horizontal',labelpad=15.0)

ax9.plot(df.index,df['Dynamic Pressure (nPa)'],c=cm.Greys(130))
ax9.errorbar(y_test_times[i],yhat[i,0:,8],yerr=mae[8],c=cm.plasma(130))
ax9.set_ylabel('P_sw',rotation='horizontal',labelpad=15.0)
ax9.set_xlabel('Datetime')
ax9.xaxis.set_major_formatter(date_format)
ax9.set_xlim(xmin=datetime(2019,2,21,6,0,0),xmax=datetime(2019,2,21,10,0,0))
fig.autofmt_xdate()
plt.show()
