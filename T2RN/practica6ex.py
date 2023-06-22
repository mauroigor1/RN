#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:27:21 2019

@author: mauri
"""


import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

####  Carga los datos
link1= '~/Escritorio/Carpetita/5to/redes neuronales/practica6/heartCP_train.csv'
link2='~/Escritorio/Carpetita/5to/redes neuronales/practica6/heartCP_test.csv'
dt=pd.read_csv(link1,",")
dt=(dt-dt.min())/(dt.max()-dt.min())	#Normalizar
dt1=pd.read_csv(link2,",")
#X=dt.loc[:,dt.columns!='chol'].values
#Y=dt.iloc[:,dt.columns].values

X=dt.iloc[:,1:13].as_matrix()
Y=dt.iloc[:,13:17].as_matrix()


Xtest=dt1.iloc[:,1:13].as_matrix()
Ytest=dt1.iloc[:,13:17].as_matrix()






n = X.shape[1]	# número de variables independientes
t = X.shape[0]	# cantidad de datos


# Crear las variables radiales

m = 5	# Cuántas neuronas radiales se quieren
mu = np.random.random((m,n))	#Los m centros


def gaussiana(x,c):
	dist = np.linalg.norm(x-c)
	return math.exp(-dist/2)

Xr = np.zeros((t,m))	#Las nuevas variables

for i in range(t):
	for j in range(m):
		Xr[i,j] = gaussiana(X[i,:],mu[j,:])

#### Prepara para aplicar redes neuronales
		
#X = Xr.copy() 	#Comentar esta línea para no hacer el cambio de variable
m = X.shape[1]

#### Redes neurnales
		
from keras.models import Sequential
from keras.layers import Dense

# Crear red
model = Sequential()
model.add(Dense(12, input_dim=m, activation='relu'))	#Probar con 22
model.add(Dense(5, activation='relu'))					#Probar con 18
model.add(Dense(4, activation='softmax'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])

# Entrenar red
history = model.fit(X, Y, validation_split=0.25, epochs=100, batch_size=10)



evaluacion = model.evaluate(Xtest, Ytest)   



# Graficar accuracy del entrenamiento y validicación
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Error cuadrático medio')
plt.ylabel('mean_squared_error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Graficar loss value del entrenamiento y validicación
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#Conocer su error
scores = model.evaluate(X, Y)
print(model.metrics_names[1] + ': ' + str(round(scores[1],8)) + '')


#Ver cómo resulta
Z=model.predict(X[:,:])	#Lo que predice la red

for i in range(10):
    print("La red entrega "+str(Z[i])+ " y debería dar " + str(Y[i]) )

# Comparar lo que predice la red con la realidad
plt.plot(Z[100:130])
plt.plot(Y[100:130])
plt.legend(['Red', 'Real'], loc='upper left')
