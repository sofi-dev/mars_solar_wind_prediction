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
ecc = 1.03
L = 2.04
e = 2.7182818284590452353602874713527
dataset_input = 0

for name in glob.glob(r'D:\maven\data\sci\kp\insitu\data\2016\06\*.tab'):
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
	print(df)
	df = df.resample('30T').mean()

	if dataset_input == 0:
		dataset = df.copy()
		dataset_input = 1
	else:
		dataset = pd.concat([dataset,df],axis=0)
		print(dataset.shape)
		#print(dataset)

for i in range(len(df['Spacecraft MSO X (km)'])):
    theta = np.arctan2(np.sqrt(df['Spacecraft MSO Y (km)'][i]*df['Spacecraft MSO Y (km)'][i]+df['Spacecraft MSO Z (km)'][i]*df['Spacecraft MSO Z (km)'][i]),df['Spacecraft MSO X (km)'][i]-x_0)

    if (np.sqrt((df['Spacecraft MSO X (km)'][i]-x_0)*(df['Spacecraft MSO X (km)'][i]-x_0)+df['Spacecraft MSO Y (km)'][i]*df['Spacecraft MSO Y (km)'][i]+df['Spacecraft MSO Z (km)'][i]*df['Spacecraft MSO Z (km)'][i])) < (L/(1+ecc*np.cos(theta))):
        df['BX (nT GSE/GSM)'][i] = np.NaN
        df['BY (nT GSE)'][i] = np.NaN
        df['BZ (nT GSE)'][i] = np.NaN
print(df['BY (nT GSE)'])
