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
print('hi')
name = r'D:\tao\output_mars_2018.txt'
print(name)
df = pd.read_table(name,sep='\s+',names=['Datetime','BX (nT GSE/GSM)','Tangential B (nT GSE)','Vx Velocity (km/s)','Tangential Velocity (km/s)','Proton Density (n/cc)','Proton Temperature (eV)','Dynamic Pressure (nPa)'],index_col=False,usecols=[0,1,2,3,4,5,6,7],skiprows=184)[['Datetime','BX (nT GSE/GSM)','Tangential B (nT GSE)','Vx Velocity (km/s)','Tangential Velocity (km/s)','Proton Density (n/cc)','Proton Temperature (eV)','Dynamic Pressure (nPa)']]
print(df)
df.set_index('Datetime',inplace=True,drop=True)

df.index= pd.to_datetime(df.index,format='%Y-%m-%dT%H:%M:%S.000')
#    df = df.resample('30T').mean()
print(df)
date_format = DateFormatter('%d-%m-%y %H:%M')

fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7,1,tight_layout=True,sharex=True)
ax1.plot(df.index,df['BX (nT GSE/GSM)'],c=cm.Greys(130))
ax1.set_ylabel('B_x',rotation='horizontal',labelpad=15.0)

ax2.plot(df.index,df['Tangential B (nT GSE)'],c=cm.Greys(130))
ax2.set_ylabel('B_T',rotation='horizontal',labelpad=15.0)

ax3.plot(df.index,df['Vx Velocity (km/s)'],c=cm.Greys(130))
ax3.set_ylabel('v_x',rotation='horizontal',labelpad=15.0)

ax4.plot(df.index,df['Tangential Velocity (km/s)'],c=cm.Greys(130))
ax4.set_ylabel('v_T',rotation='horizontal',labelpad=15.3)

ax5.plot(df.index,df['Proton Density (n/cc)'],c=cm.Greys(130))
ax5.set_ylabel('n_p',rotation='horizontal',labelpad=15.0)

ax6.plot(df.index,df['Proton Temperature (eV)'],c=cm.Greys(130))
ax6.set_ylabel('T_p',rotation='horizontal',labelpad=15.0)

ax7.plot(df.index,df['Dynamic Pressure (nPa)'],c=cm.Greys(130))
ax7.set_ylabel('P_dyn',rotation='horizontal',labelpad=15.0)
ax7.set_xlabel('Datetime')
ax7.xaxis.set_major_formatter(date_format)
ax7.set_xlim(xmin=datetime(2018,11,21,20,0,0),xmax=datetime(2018,11,22,0,0,0))
fig.autofmt_xdate()
plt.show()
