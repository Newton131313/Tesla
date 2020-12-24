# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 20:29:31 2020

@author: Mike

"""

import numpy as np
import pandas as pd

# Data upload and dependable create variable array
df_train=pd.read_csv(r"D:\Udemy\Deep learning A-Z course\Part+3+-+Recurrent+Neural+Networks\Part 3 - Recurrent Neural Networks\Google_Stock_Price_Train.csv")
X_train=df_train.iloc[:,1:2].values # we're looking to predict the open stock value of Google market 

# Feature Scaling : Normalization is recommended & more relevant to be apply with RNN especialy when u have a sigmoid function as an activation function in the output layer
from sklearn.preprocessing import MinMaxScaler 
sc =MinMaxScaler(feature_range=(0,1))
X_scaled = sc.fit_transform(X_train)

# X_train = np.hstack(training_set_scaled[i:i-60 or None:1] for i in range(0,60))
# the previous line should be a replacement of the for loop below. need to check it. 
#Data Structure to build RNN (60 timesteps (based on experience or trial and error) and 1 output)
x_train =[]
y_train = []
for i in range (60 ,1258):
    x_train.append(X_scaled[i-60:i, 0])
    y_train.append(X_scaled[i, 0])
x_train, y_train=np.array(x_train), np.array(y_train)  

#Reshaping to add any new dimesion to ur function always use reshape function
x_train = np.reshape(x_train, (x_train.shape[0], x_train[1], 1)) # last 1 represent no. of indicators

from keras.model import Sequential 
from keras.layers import  Dense
from keras.layers import  LSTM 
from keras.layers import  Dropout

reg = Sequential()

# Adding first LSTM layer and some Dropout regularisation
reg.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
reg.add(Dropout(0.2))

# Adding second LSTM layer and some Dropout regularisation
reg.add(LSTM(units=50, return_sequences=True))
reg.add(Dropout(0.2))

# Adding third LSTM layer and some Dropout regularisation
reg.add(LSTM(units=50, return_sequences=True))
reg.add(Dropout(0.2))

# Adding fourth LSTM layer and some Dropout regularisation
reg.add(LSTM(units=50))
reg.add(Dropout(0.2))

# Adding output layer
reg.add(Dense(units=1))
