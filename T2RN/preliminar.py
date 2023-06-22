#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:10:07 2019

@author: mauri
"""


import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense




#from utils import combine_images
#from PIL import Image
#from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

link = '~/Escritorio/Carpetita/5to/redes neuronales/mnist/mnist_train.csv'



# conjunto  de datos de entrenamiento y testeo. 
df= pd.read_csv(link)


# separar para entrenamiento
x=df.iloc[:,1:]
y=df.iloc[:,:1]



# Normalizacion de X  y paso de Y a matriz binaria    
x= x.astype('float32') / 255.# X contiene las imagenes normalizadas para entrenamiento
y= to_categorical(y.astype('float32'))## Y es la variable a predecir en entrenamiento 






# Crear Modelo secuencial 
model = Sequential()
model.add(Dense(100, input_dim=x.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Fit the model
history=model.fit(x, y, batch_size=350, epochs=20, validation_split=0.3, verbose=1)
scores = model.evaluate(x, y)   
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))#precisión


get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[2].output])
layer_output = get_3rd_layer_output([x])[0]
layer_output = np.argmax(layer_output,axis=1)
yyy=np.argmax(y,axis=1)




















#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)



#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)






