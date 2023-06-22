# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:30:12 2019

@author: Bruko
"""

import pandas as pd
import numpy as np

#Cargar datos
dt=pd.read_csv("datos.csv","\t")

#Número de neuronas de entrada
n=dt.shape[1]-1
m=dt.shape[0]
x=dt.iloc[:,:n].as_matrix() # pasar a matriz
t=dt.iloc[:,n].as_matrix()

#pesos
w = np.random.random(n)
b = np.random.random()  #bias

print('Input: ', x)
print('Output: ', t)

for i in range(100):
    error = 0
    for k in range(m):
        #Qué entrega
        z = np.sign(np.dot(x[k], w) - b)
        
        #Si no acierta
       
        if z !=t[k]:
           w = w+t[k]*x[k,:]
           b =  b-t[k]
           error = 1
           print("fallo con" + str(x[k]))
    if error==0:
        break

#probar
for k in range(m):
    #Qué entrega
    z = np.sign(np.dot(x[k], w) - b)
    print("Entrada: " + str(x[k]) + ", salida = " + str(z) + " y debería ser " + str(t[k]))