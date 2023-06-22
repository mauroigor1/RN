#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:44:20 2019

@author: mauri
"""
import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout

link1 = '~/Escritorio/Carpetita/5to/redes neuronales/mnist/mnist_train.csv'
link2 = '~/Escritorio/Carpetita/5to/redes neuronales/mnist/mnist_test.csv'


# conjunto  de datos de entrenamiento y testeo. 
df1= pd.read_csv(link1)
df2= pd.read_csv(link2)

# separar para entrenamiento
xtrain=df1.iloc[:,1:]
ytrain=df1.iloc[:,:1].as_matrix()
# separar para testeo
xtest=df2.iloc[:,1:]
ytest=df2.iloc[:,:1].as_matrix()

# Se crea la variable a predecir ( num>=5 0 <5), para train y test:
for i in range(ytrain.shape[0]):
    if ytrain[i] >= 5:
        ytrain[i] = 1
    else:
        ytrain[i] = 0

for i in range(ytest.shape[0]):
    if ytest[i] >= 5:
        ytest[i] = 1
    else:
        ytest[i] = 0



# Normalizacion de X  y paso de Y a matriz binaria    
X= xtrain.astype('float32') / 255.# X contiene las imagenes normalizadas para entrenamiento
Y= to_categorical(ytrain.astype('float32'))## Y es la variable a predecir en entrenamiento 
Xt= xtest.astype('float32') / 255.# X contiene las imagenes normalizadas para testeo
Yt= to_categorical(ytest.astype('float32'))## Yt es la variable a predecir en testeo


### ENTRENAMIENTO
# Crear Modelo secuencial para entrenamiento 
model = Sequential()
model.add(Dense(80, input_dim=X.shape[1], activation='sigmoid'))
#model.add(Dense(8, activation='relu')) ## intentos
model.add(Dropout(0.5))##unids de entrada a desactivar al azar 
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Fit the model
history=model.fit(X, Y, batch_size=200, epochs=30,validation_split=0.3, verbose=1)

scores = model.evaluate(Xt, Yt)   
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))#precisión
output = K.function([model.layers[0].input], [model.layers[2].output])# obtener resultado
output = output([X])[0] # obtener resultado
output = np.argmax(output,axis=1)# de categorico a numerico
yy=np.argmax(Y,axis=1)## de categorico a numerico











#Plot the Loss Curves
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.xticks(range(0,35,5))
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)



#Plot the Accuracy Curves
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


















