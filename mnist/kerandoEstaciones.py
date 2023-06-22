# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:45:58 2019

@author: Bruko
"""
## MInimizar el error linea 65
# Modificando la estructura de la red y funciones de activacion

# Fec 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


link = '~/Escritorio/Carpetita/5to/redes neuronales/Sydney.csv'


dt=pd.read_csv(link)



dt.info()


dt=dt.drop(labels=['Date'], axis=1)		#Eliminar fecha
dt=(dt-dt.min())/(dt.max()-dt.min())	#Normalizar
X=dt.loc[:,dt.columns!='Temp3pm'].values
Y=dt.loc[:,dt.columns=='Temp3pm'].values




dt['Mes']=0

pd.options.mode.chained_assignment = None  # para sacar un warning molesto

for i in range(dt.shape[0]):
    dt["Mes"][i]=int(dt["Date"][i].split('-')[1])

dt['Verano']=0
dt['Primavera']=0
dt['Otono']=0
dt['Invierno']=0

for i in range(dt.shape[0]):
    if dt["Mes"][i]==12 or dt["Mes"][i]==1 or dt["Mes"][i]==2:
        dt['Verano'][i]=1
    if dt["Mes"][i]==3 or dt["Mes"][i]==4 or dt["Mes"][i]==5:
        dt['Otono'][i]=1
    if dt["Mes"][i]==6 or dt["Mes"][i]==7 or dt["Mes"][i]==8:
        dt['Invierno'][i]=1
    if dt["Mes"][i]==9 or dt["Mes"][i]==10 or dt["Mes"][i]==11:
        dt['Primavera'][i]=1
 



#plt.hist(dt['Rainfall'])

from keras.models import Sequential
from keras.layers import Dense

# Crear modelo
model = Sequential()
model.add(Dense(12, input_dim=15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])

history = model.fit(X, Y, validation_split=0.25, epochs=100, batch_size=10)


# Graficar accuracy del entrenamiento y validicación
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Model accuracy')
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
Z=model.predict(X[:10,:])

for i in range(10):
    print("La red entrega "+str(Z[i])+ " y debería dar " + str(Y[i]) )
