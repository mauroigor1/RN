#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:13:59 2019

@author: mauri
"""
#%%
import pandas as pd
import numpy as np
import math

#%%
link = '~/Escritorio/Carpetita/5to/redes neuronales/Sydney.csv'

tb=pd.read_csv(link)

v=tb['WindGustSpeed']
numero_filas=v.shape[0]
numero_columnas=4

matriz = [[], [], [], []]

for i in range(numero_columnas):
    matriz[i] = v[i: numero_filas - 3 + i]

matriz = np.array(matriz)  
matriz=matriz.T # trasponemos la matriz
delta = matriz.max()-matriz.min()
minimo = matriz.min()
matriz=(matriz-minimo)/delta  # normalizamos los datos

#%%

t = matriz[:,3]
x = matriz[:,0:3]

nu = 0.1  #tasa de aprendizaje
                                
n=x.shape[1]  #Número de neuronas de entrada
m=x.shape[0]       #Número datos

z=np.zeros(m)   #Salida de la red

w = np.zeros(n) #Pesos de la red
b = np.random.random() #Umbral de la neurona
iteracion = 0

#%%
def Error(y_mod,y_obs):
    e=0
    for i in range(y_mod.shape[0]):
        e = e + 0.5*(y_mod[i]-y_obs[i])**2
    return e

# utilizamos la funcion relu para abordar el problema

def funcion(x):
    #if x>0:
    #    return x
    #else:
    #   return 0
    return (1/(1+math.exp(-x)))
    #return math.tanh(x)

def derivada_funcion(x):
    #if x>0:
    #    return 1
    #else:
    #    return 0
    return (funcion(x)*(1-funcion(x)))

#Fracción de datos que correctamente clasifica
def aciertos(y_mod,y_obs):
    e=0
    for i in range(y_mod.shape[0]):
        if round(y_mod[i],0)==round(y_obs[i],0):
            e = e + 1
    return (float(e)/y_mod.shape[0])

#%%
for i in range(10):
    error = 0
    iteracion = iteracion + 1
    for k in range(m):  #Por cada vector de entrada
        #Qué entrega
        #z[k] = sigmoidal(np.dot(x[k], w) - b)
        z[k] = funcion(np.dot(x[k], w) - b)
        #Si no acierta
        if z[k]!=t[k]:
            for j in range(n): #Modifica cada peso
                #w[j] = w[j] - nu*(z[k]-t[k]) * x[k,j] * derivadaSigmoidal(x[k,j])
                 w[j]= w[j] - nu*(z[k]-t[k] )* x[k,j]*derivada_funcion(x[k,j])    
            b= b + nu*( z[k]-t[k])*derivada_funcion(1)                   
            #b = b + nu*(z[k]-t[k])*derivadaSigmoidal(1)
            error = error + 1
    #print(str(iteracion) + " El error es " + str(Error(z,t)))
    #print(str(iteracion) + " Aciertos: " + str(aciertos(z,t)))
    if error==0:
        break

print("porcentaje de aciertos: " + str(round(aciertos(z,t)*100,2)) + "%\n")
for k in range(m):
    print("salida = " + str(round(z[k],3)) + " y debería ser " + str(round(t[k],3)))

t = t*delta + minimo
z = z*delta + minimo

for k in range(m):
    print("salida = " + str(round(z[k],3)) + " y debería ser " + str(round(t[k],3)))

#%%
dia_predict = 130 # elegir el dia que se desea predecir, elegir entre 0 y 1756
print("\npara el dia " + str(dia_predict) + " tenemos \n")
print("prediccion segun modelo: " + str(round(z[dia_predict],2)) + "\ndato real: " + str(round(t[dia_predict],2)) )







