# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:30:12 2019

@author: Bruko
"""
####Aqui se predice target 
import pandas as pd
import numpy as np
import math

#Cargar datos

#dt=pd.read_csv("heart2.csv",",")
dt=pd.read_csv("heart0.csv",",")
dt=dt.fillna(0) #Eliminar NaN
#dt=(dt-dt.min())/(dt.max()-dt.min()) #Normalizar
nu = 0.01    #tasa de aprendizaje

n=dt.shape[1]-1     #Número de neuronas de entrada
m=dt.shape[0]       #Número datos
x=dt.iloc[:,:n].as_matrix() #Variables independientes
#f=dt.iloc[:,3].as_matrix()
#p=np.concatenate((x,f))



t=dt.iloc[:,n].as_matrix()  #Variable dependiente
z=np.zeros(m)   #Salida de la red

w = np.random.random(n)/10 #Pesos de la red
b = np.random.random()  #Umbral de la neurona
iteracion = 0
def Error(y_mod,y_obs):
    e=0
    for i in range(y_mod.shape[0]):
        e = e + 0.5*(y_mod[i]-y_obs[i])**2
    return e


def sigmoidal(x):
    return (1/(1+math.exp(-x)))

def derivadaSigmoidal(x):
    return (sigmoidal(x)*(1-sigmoidal(x)))

def relu(x):
    if x>0:
        return x
    else:
        return 0

def derivadaRelu(x):
    if x>0:
        return 1
    else:
        return 0

#Fracción de datos que correctamente clasifica
def aciertos(y_mod,y_obs):
    e=0
    for i in range(y_mod.shape[0]):
        if round(y_mod[i],0)==round(y_obs[i],0):
            e = e + 1
    return (e/y_mod.shape[0])


for i in range(100):
    error = 0
    iteracion = iteracion + 1
    for k in range(m):  #Por cada vector de entrada
        #Qué entrega
        z[k] = sigmoidal(np.dot(x[k], w) - b)
        
        #Si no acierta
        if z[k]!=t[k]:
            for j in range(x.shape[1]): #Modifica cada peso
                w[j] = w[j] - nu*(z[k]-t[k]) * x[k,j] * derivadaSigmoidal(x[k,j])
            b = b + nu*(z[k]-t[k]) * derivadaSigmoidal(1)
            error = error + 1
    #print(str(iteracion) + " El error es " + str(Error(z,t)))
    print(str(iteracion) + " Aciertos: " + str(aciertos(z,t)))
    if error==0:
        break




#probar
for k in range(m):
    #Qué entrega
    z[k] = sigmoidal(np.dot(x[k], w) - b)
    #print("Entrada: " + str(x[k]) + ", salida = " + str(z) + " y debería ser " + str(t[k]))
    print("salida = " + str(round(z[k],0)) + " y debería ser " + str(t[k]))
print(aciertos(z,t))