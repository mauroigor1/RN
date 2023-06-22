# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:30:12 2019

@author: Bruko
"""

import pandas as pd
import numpy as np

#Cargar datos

#dt=pd.read_csv("heart2.csv",",")
dt=pd.read_csv("ejemplo.csv",",")
dt=dt.fillna(0)
nu = 0.0000001    #tasa de aprendizaje

n=dt.shape[1]-1     #Número de neuronas de entrada
m=dt.shape[0]       #Número datos
x=dt.iloc[:,:n].as_matrix() #Variables independientes
t=dt.iloc[:,n].as_matrix()  #Variable dependiente
z=np.zeros(m)   #Salida de la red
#pesos
w = np.random.random(n) #Pesos de la red
b = np.random.random()  #Umbral de la neurona

def Error(y_mod,y_obs):
    e=0
    for i in range(y_mod.shape[0]):
        #e = e + 0.5*(t[k]-z[k])**2 ############################ Completar
        e = e + 0.5*(y_mod[i] - y_obs[i])**2    
    return e


print('Input: ', x)
print('Output: ', t)

for i in range(100):
    acerto = True
    for k in range(m):  #Por cada vector de entrada
        #Qué entrega
        z[k] = np.dot(x[k], w) - b
        
        #Si no acierta
        if z[k]!=t[k]:
            print("Dio "+str(z[k])+" y debería ser " + str(t[k]))
            ############### Completar
            w = w - nu* (t[k]-z[k])* x[k,:]
            b =  b + nu*(z[k]-t[k])
            acerto = False
    print("El error es " + str(Error(z,t)))
    if acerto is True:
        break

#probar
for k in range(m):
    #Qué entrega
    z = np.dot(x[k], w) - b
    print("Entrada: " + str(x[k]) + ", salida = " + str(z) + " y debería ser " + str(t[k]))
    
    
    
    