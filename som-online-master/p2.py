#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:20:42 2019

@author: mauri
"""


#El siguiente, es el codigo python que presenta los pasos seguidos para abordar en problema.

import os
os.chdir('/home/mauri/Escritorio/Carpetita/5to/redes neuronales/som-online-master')#3 fijar directorio de trabajo

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines
#link1 = '~/Escritorio/Carpetita/5to/redes neuronales/mnist/mnist_train.csv'

link = '~/Escritorio/Carpetita/5to/redes neuronales/T3RN/Parques_y_Jardines_Arbol_ZonaVerde.csv'

#Se buscan las 16 especies de árboles más frecuentes ¿Cuáles son?:

df = pd.read_csv(link)
nombre=df['Nombre'].value_counts()
nombres=nombre[:16]

#El siguiente es el filtrado de df para obtener las filas de las 16 especies.


s = [('Citrus aurantium'), ('Pinus pinea'),('Tipuana tipu'),('Melia azedarach'),('Celtis australis'),('Olea europaea'),('Jacaranda mimosifolia'),('Brachychiton populneus'),('Ulmus pumila'),('Ceratonia siliqua'),('Quercus ilex'),('Platanus x hybrida'),('Casuarina equisetifolia'),('Robinia pseudoacacia'),('Ulmus sp'),('Fraxinus excelsior')]
input_data = df[df.Nombre.isin(s)] # df filtrado
input_data=input_data.dropna() # quitar Nan de la tabla
input_data.iloc[:20,:]## ver un par de datos ​


#Se desea encontrar vector representativo respecto a la altura, posición y diámetro.


agri_data = input_data.iloc[np.random.permutation(len(input_data))]## random permutacion
trunc_data = agri_data[["xlo","ylo","Perimetro","Altura"]]## 
trunc_data.iloc[:20,:]

# Normalizarlos
max=trunc_data.max()
trunc_data = trunc_data / trunc_data.max()
trunc_data.iloc[:10,:]


#Se debe importar las funciones de online-som.py para crear el modelo:


from com.machinelearningnepal.som.online_som import SOM


#A continuación se crea el modelo (y se entrena) de 4 neuronas de entrada y salida y 4 caracteriscticas (x_lo,y_lo,perimetro,altura) base de las cuales se quiere agrupar los datos de input_data. Dicha cantidad de neuronas de entrada y salida se escogen con la idea de agrupar los datos en 16 grupos, intentando que dichos grupos coincidan con la cantidad de tipos de especies de arbol con las que se trabaja.


agri_som = SOM(4,4,4)
init_fig = plt.figure()
agri_som.show_plot(init_fig, 1, 0)
plt.show()

a#gri_som.train(trunc_data.values,num_epochs=100,init_learning_rate=0.01)​
agri_som.train(trunc_data.values,num_epochs=100,init_learning_rate=0.01)
#Se crea la fc predict para luego agrupar.

def predict(df):
    bmu, bmu_idx = agri_som.find_bmu(df.values)
    df['bmu'] = bmu         #unidad de mejor correspondencia 
    df['bmu_idx'] = bmu_idx  # el índice de la unidad de mejor correspondencia 
    return df

clustered_df = trunc_data.apply(predict, axis=1)## se clusteriza trunc_data
joined_df = agri_data.join(clustered_df, rsuffix="_norm")## se agregan las demas variables con las que nose agrupó
joined_df[0:20]#ver un par de datos


#¿Cuáles serán son los clusters? A continuación se separan los grupos obtenidos para encontrar el vector caracteristico de cada uno de ellos y para ver los grupos (obvio).


c00=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==0 and x[1]==0)]
c01=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==0 and x[1]==1)]
c02=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==0 and x[1]==2)]
c03=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==0 and x[1]==3)]



c10=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==1 and x[1]==0)]
c11=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==1 and x[1]==1)]
c12=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==1 and x[1]==2)]
c13=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==1 and x[1]==3)]

​
c20=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==2 and x[1]==0)]
c21=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==2 and x[1]==1)]
c22=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==2 and x[1]==2)]
c23=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==2 and x[1]==3)]
​
​
​
c30=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==3 and x[1]==0)]
c31=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==3 and x[1]==1)]
c32=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==3 and x[1]==2)]
c33=joined_df[joined_df['bmu_idx'].apply(lambda x: x[0]==3 and x[1]==3)]
​
​
#Cada tabla representante de cada grupo presenta una variable llamada bmu (best matching unit) que representa las coordenadas de la neurona ganadora para cada grupo o el vector caracteristico de cada grupo otenido a partir de las variable xlo,ylo,perimetro,altura. bmu_idx indica el indice (i,j) de la neurona ganadora.


c00['bmu']



#Mirando cada bmu de cada grupo se obtienen los vectores caracteristicos, que son los sgtes:


v00=[0.97475624, 0.9987395 , 0.05428342, 0.00562672]
v01=[0.97066911, 0.99870891, 0.07260804, 0.00626777]
v02= [0.9679143 , 0.99879698, 0.11640426, 0.00755761]
v03= [0.96894361, 0.99890476, 0.14307396, 0.00851467]
v10=[0.98045026, 0.9988977 , 0.07131754, 0.00598091]
v11=[0.97625025, 0.99886178, 0.09882555, 0.00693225]
v12=[0.97270433, 0.99889087, 0.15293757, 0.00895797]
v13=[0.96977359, 0.99892401, 0.18204585, 0.01002218]
v20=[0.98867452, 0.999126  , 0.11290045, 0.00722218]
v21=[0.98216196, 0.99897155, 0.15464209, 0.00926637]
v22=[0.97679108, 0.99893907, 0.20699875, 0.0112726 ]
v23=[0.97311787, 0.99890804, 0.25690427, 0.012881  ]
v30=[0.99015945, 0.99917659, 0.15105916, 0.00859597]
v31=[0.98718231, 0.99907023, 0.17889937, 0.00996519]
v32=[0.97987564, 0.99898161, 0.24555102, 0.01233951]
v33=[0.97258439, 0.99887763, 0.33366329, 0.01510758]
​
#La tabla de vectores caracteristicos será:


import numpy as np
max * np.array(v00)
​
​

M=[[v00,v01,v02,v03],[v10,v11,v12,v13],[v20,v21,v22,v23],[v30,v31,v32,v33]]
M#tabla de vectores caracts

#Se observa que los vectores son similares, y los mas parecidos entre si se encuentrar mas cercanos por ej v00 y v01 son mas cercanos entre si que v01 y v10, esto se puede apreciar simplemente mirando sus componentes.

#Para la construccion de la U matriz se define la sgte fc:


def matrixU(m):
    mu = np.zeros((m.shape[0]-2,m.shape[1]-2))
            
    for i in range(0,m.shape[0]-2):
        for j in range(0,m.shape[1]-2):
            mu[i,j] = np.linalg.norm(m[i,j]-m[i,j+1])+np.linalg.norm(m[i,j]-m[i,j-1])+np.linalg.norm(m[i,j]-m[i+1,j])+np.linalg.norm(m[i,j]-m[i-1,j])
    return mu

M = np.array(M)
u=matrixU(M)## la U matriz

#??