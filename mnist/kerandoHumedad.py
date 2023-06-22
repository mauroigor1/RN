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
X=dt.loc[:,dt.columns!='Humidity9am'].values
Y=dt.loc[:,dt.columns=='Humidity9am'].values
for i in range(Y.shape[0]):
    if Y[i]>0.6:
        Y[i]=1
    else:
        Y[i]=0





from keras.models import Sequential
from keras.layers import Dense

# Crear modelo
model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(X, Y, validation_split=0.25, epochs=100, batch_size=10)


# Graficar accuracy del entrenamiento y validicación
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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

Z=model.predict(X[:10,:])

for i in range(10):
    print("La red entrega "+str(Z[i])+ " y debería dar " + str(Y[i]) )
