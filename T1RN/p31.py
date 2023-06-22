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
delta = df.max()-df.min()
minimo = df.min()
df=(df-minimo)/delta # normalizamos los datos

nu = 0.5
#%%
n=df.shape[1]-1     #Número de neuronas de entrada
m=df.shape[0]       #Número datos
xx = df
t=df['MaxTemp'].as_matrix()  #Variable dependiente (la que queremos predecir)
del xx['MaxTemp']
x = xx.as_matrix() # variables independientes (con las que intentaremos predecir la temperatura maxima)

z=np.zeros(m)   #Salida de la red

w = np.zeros(n) #Pesos de la red
b = np.random.random()  #Umbral de la neurona
iteracion = 0

#%%

def funcion(x):
    return math.tanh(x)

def derivada_funcion(x):
    return math.cosh(x)**(-2)

def aciertos(y_mod,y_obs):
    e=0
    for i in range(y_mod.shape[0]):
        # definimos una diferencia de 2.5 grados (o menor) como aceptable
        # para la prediccion y la consideramos correcta
        if abs(y_mod[i] - y_obs[i]) <= 2.5: 
            e = e + 1          
    return (float(e)/y_mod.shape[0])
        #if round(y_mod[i],1)==round(y_obs[i],1):
        

#%%
for i in range(300):
    error = 0
    iteracion = iteracion + 1
    for k in range(m):  #Por cada vector de entrada
        #Qué entrega
        z[k] = funcion(np.dot(x[k], w) - b)
        
        #Si no acierta
        if z[k]!=t[k]:
            for j in range(x.shape[1]): #Modifica cada peso
                w[j] = w[j] - nu*(z[k] - t[k])*derivada_funcion(x[k,j])*x[k,j]
            b = b + nu*(z[k] - t[k])*derivada_funcion(1)
            error = error + 1

    if error == 0:  # si no hay error, entonces termina
        break
# desnormalizamos los datos originales y los obtenidos por el modelo
z = z*delta['MaxTemp'] + minimo['MaxTemp']
t = t*delta['MaxTemp'] + minimo['MaxTemp']

for k in range(m):
    print("salida = " + str(round(z[k],3)) + " y debería ser " + str(round(t[k],3)))

# ver linea 44 para saber como estan definidos los aciertos
print("\nPorcentaje de aciertos " + str(round(aciertos(z,t)*100,2)) + "%")

#%%
dia_predict = 1756   #INGRESE EL DIA QUE SE DESEA PREDECIR, ENTRE 0 Y 1756

print("\nla predicción de la temperatura máxima para el dia " + str(dia_predict) + " es según:")
print("el modelo: " + str(round(z[dia_predict],1)) + "\nel resultado real es: " + str(round(t[dia_predict],1))+"\n\n")


#print('draw.io '+' pag para dibujar grafos')
