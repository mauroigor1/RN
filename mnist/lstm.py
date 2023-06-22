#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:23:36 2019

@author: mauri
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error







#link1 = '~/Escritorio/Carpetita/5to/redes neuronales/mnist/mnist_train.csv'


link1 = '~/Escritorio/Carpetita/5to/redes neuronales/Sydney.csv'

tb=pd.read_csv(link1)

v=tb['WindGustSpeed']
numero_filas=v.shape[0]
numero_columnas=4

matriz = [[], [], [], []]

for i in range(numero_columnas):
    matriz[i] = v[i: numero_filas - 3 + i]

matriz = np.array(matriz)  
matriz=matriz.T # trasponemos la matriz
delta = matriz.max()-matriz.min()
minimo = matriz.min()
matriz=(matriz-minimo)/delta  # normalizamos los datos


t = matriz[:,3] # a predecir
x = matriz[:,0:3]#
x =  x.reshape(x.shape[0],  x.shape[1],1)## impotante paso

x.shape

	
# fix random seed for reproducibility
np.random.seed(7)


#.values





	
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(9, input_shape=(3, 1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')



model.fit(x, t, epochs=200, batch_size=1, verbose=2)


	
# make predictions
trainPredict = model.predict(x)
testPredict = model.predict(t)








