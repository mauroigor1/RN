#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 23:04:32 2019

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
xtrain2=df1.iloc[:,1:]
ytrain2=df1.iloc[:,:1].as_matrix()
# separar para testeo
xtest2=df2.iloc[:,1:]
ytest2=df2.iloc[:,:1].as_matrix()


# Se crea la variable a predecir ( mod2=0, para train y test:
for i in range(ytrain2.shape[0]):
    if ytrain2[i]%2==0:
        ytrain2[i] = 1
    else:
        ytrain2[i] = 0

for i in range(ytest2.shape[0]):
    if ytest2[i]%2 ==0:
        ytest2[i] = 1
    else:
        ytest2[i] = 0



# Normalizacion de X  y paso de Y a matriz binaria    
X2= xtrain2.astype('float32') / 255.# X2 contiene las imagenes normalizadas para entrenamiento
Y2= to_categorical(ytrain2.astype('float32'))## Y2 es la variable a predecir en entrenamiento 
Xt2= xtest2.astype('float32') / 255.# Xt2 contiene las imagenes normalizadas para testeo
Yt2= to_categorical(ytest2.astype('float32'))## Yt2 es la variable a predecir en testeo


### ENTRENAMIENTO
# Crear Modelo secuencial para entrenamiento 
model = Sequential()
model.add(Dense(70, input_dim=X2.shape[1], activation='sigmoid'))
#model.add(Dense(8, activation='relu')) ## intentos
model.add(Dropout(0.5))## unids de entrada a desactivar al azar 
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Fit the model
history=model.fit(X2, Y2, batch_size=200, epochs=25,validation_split=0.3, verbose=1)

scores2 = model.evaluate(Xt2, Yt2)   
print("\n%s: %.2f%%" % (model.metrics_names[1], scores2[1]*100))#precisión
output2 = K.function([model.layers[0].input], [model.layers[2].output])# obtener resultado
output2 = output2([X2])[0]# obtener resultado
output2 = np.argmax(output2,axis=1)# de categorico a numerico
yy2=np.argmax(Y2,axis=1)# de categorico a numerico







#Plot the Loss Curves
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.xticks(range(0,31,5))
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





