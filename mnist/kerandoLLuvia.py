# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:45:58 2019

@author: Bruko
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dt=pd.read_csv("sydney.csv",",")
dt.info()
dt=dt.drop(labels=['Date'], axis=1)		#Eliminar fecha
dt=(dt-dt.min())/(dt.max()-dt.min())	#Normalizar
#X=dt.loc[:,dt.columns!='Temp3pm'].values
#Y=dt.loc[:,dt.columns=='Temp3pm'].values

X=dt.loc[:,dt.columns!='Rainfall'].values
Y=dt.loc[:,dt.columns=='Rainfall'].values

#plt.hist(dt['Rainfall'])

from keras.models import Sequential
from keras.layers import Dense

# Crear modelo
model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))
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



Z=model.predict(X)

scores = model.evaluate(X, Y)
print(model.metrics_names[1] + ': ' + str(round(scores[1],8)) + '')

#Evaluar otras cosas
c=np.array([[0,0,0,0,0,0,0,0,0,0.1]])
model.predict([c])

T=100
Z=model.predict(X[:T,:])

for i in range(T):
    print("La red entrega "+str(Z[i])+ " y debería dar " + str(Y[i]) )
