# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:45:58 2019

@author: Bruko
"""

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





#plt.hist(dt['Rainfall'])

from keras.models import Sequential
from keras.layers import Dense
## dense es full connected todos con todos
# Crear modelo
model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))##capa1
model.add(Dense(5, activation='relu'))##capa 2
model.add(Dense(1, activation='relu'))##capa3

print(model.summary())
#optimizer sgd adam
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['mean_squared_error'])

history = model.fit(X, Y, validation_split=0.3, epochs=50, batch_size=10)


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
