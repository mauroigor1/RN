# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:09:44 2019

@author: User
"""
#%%
import pandas as pd
import numpy as np
import math

#%%
link = '~/Escritorio/Carpetita/5to/redes neuronales/Sydney.csv'
df = pd.read_csv(link)
#x = df['MaxTemp']
del df['Date'] # Eliminamos la columna de las fechas porque no nos permite normalizar los datos

t = df['Humidity9am'].as_matrix()  #Variable dependiente

t_binario = np.zeros(t.shape[0]) # variable a predecir
for i in range(t.shape[0]):
    if t[i] >= 60:
        t_binario[i] = 1
    else:
        t_binario[i] = 0

df=(df-df.min())/(df.max()-df.min()) # normalizamos los datos
nu = 0.5
#%%

n=df.shape[1]-1     #Número de neuronas de entrada
m=df.shape[0]       #Número datos

del df['Humidity9am']
x = df.as_matrix()

z=np.zeros(m)   #Salida de la red

w = np.random.random(n)/20 #Pesos de la red
b = np.random.random()  #Umbral de la neurona
iteracion = 0

#%%

def Error(y_mod,y_obs):
    e=0
    for i in range(y_mod.shape[0]):
        e = e + 0.5*(y_mod[i]-y_obs[i])**2
    return e

def funcion(x):
    return math.tanh(x)

def derivada_funcion(x):
    return math.cosh(x)**(-2)

def aciertos(y_mod,y_obs):
    e=0
    for i in range(y_mod.shape[0]):
        if round(y_mod[i],0)==round(y_obs[i],0):
            e = e + 1
    return (float(e)/y_mod.shape[0])
#%%
for i in range(100):
    error = 0
    iteracion = iteracion + 1
    for k in range(m):  #Por cada vector de entrada
        #Qué entrega
        z[k] = funcion(np.dot(x[k], w) - b)
        
        #Si no acierta
        if z[k]!=t_binario[k]:
            for j in range(x.shape[1]): #Modifica cada peso
                w[j] = w[j] - nu*(z[k] - t_binario[k])*derivada_funcion(x[k,j])*x[k,j]
            b = b + nu*(z[k] - t_binario[k])*derivada_funcion(1)
            error = error + 1
    #print(str(iteracion) + " El error es " + str(Error(z,t)))
    #print(str(iteracion) + " Aciertos: " + str(aciertos(z,t)))

    if error == 0:  # si no hay error, entonces termina
        break

for i in range(z.shape[0]):
    if z[i] >= 0.6:
        z[i] = 1
    else:
        z[i] = 0
        
for k in range(m):
    print("salida = " + str(round(z[k],3)) + " y debería ser " + str(round(t_binario[k],3)))

print("\nPorcentaje de aciertos: " + str(aciertos(z,t_binario)*100) + "%")
#%%
        
dia_predict = 1547 #INGRESE EL NUMERO DEL DIA A PREDECIR ENTRE 0 Y 1756
print("\nsi obtiene el numero 1 como resultado significa que la humedad será mayor al 60% (según modelo), y recibirá -1 en caso contrario\n\nSalida: " + str(z[dia_predict]))
##
## en caso de querer cambiar el dia de la prediccion basta ejecutar solo este bloque.
##
