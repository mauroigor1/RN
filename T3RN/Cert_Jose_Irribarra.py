# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:25:41 2019

@author: User
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
""" Problema 3 """
# a)

link = 'C:/Users/josep/Documents/Python Scripts/Data Minning/Certamen I/Vino.csv'
df =  pd.read_csv(link, sep = ',', decimal = '.')
df.head()

#%%
print('La dimension de la tabla es de ' +  str(df.shape[0]) + 'x' + str(df.shape[1]) + '\n')
print('las variables que contiene son: '+ str(df.columns.tolist()))
print('\n')

media = []
for i in range(df.shape[1]):
    media.append(np.mean(df[df.columns[i]])) 
    print('la media para la variable  ' + df.columns[i] + '  es: ' + str(media[i]) )

print('\n')
# CALCULAMOS LA DESVIACION ESTANDAR PARA CADA VARIABLE
desviacion = []
for i in range(df.shape[1]):
    desviacion.append(np.std(df[df.columns[i]])) 
    print('la desviación estandar para la variable  ' + df.columns[i] + '  es: ' + str(desviacion[i]) )
#eliminamos la columna 'Unnamed: 0' que no aporta informacion
del df[df.columns[0]]


#%%
"""
    Problema 3.b
    Aplicamos PCA a la tabla de datos
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns

escala = MinMaxScaler()
df = escala.fit_transform(df) # estandarizamos los datos con valores entre 0 y 1

n_comp = df.shape[1]

pca = PCA(n_components = n_comp)
PC = pca.fit_transform(df)

# Grafico de la varianza explicada acumulada por cada componente
plt.figure(figsize = (9,6))
y = np.array([])
var_exp = np.array([])

#%%
"""

-------ESTE BLOQUE ES PARA GRAFICAR LA VARIANZA EXPLICADA DE PCA----------

"""


for j in range(n_comp):
    y = np.append(y, pca.explained_variance_ratio_[j])
    var_exp = np.append(var_exp, y.sum())
    
plt.plot(np.linspace(1,n_comp, n_comp), var_exp,'bo-')
plt.axhline(0.7, color = 'r')
plt.axvline(3, color = 'g')
plt.axvline(4, color = 'g')
plt.xticks(range(1,15))
plt.xlabel('Numero de Componentes Principales')
plt.ylabel('Varianza explicada Acumulada')
plt.grid()

"""
    Por lo visto en el grafico, cuando se tiene 3 componentes principales 
    se explica poco menos de un 70% de la informacion de los datos, pero con
    4 componentes se explica mas o menos un 75%.
    En particular, se explica un 75.7081847630228 % de la informacion con las
    4 componentes.

"""

#%%

pca2 = PCA(n_components = 4)
PC2 = pca2.fit_transform(df)
print("\nla varianza explicada con 4 componentes es de un " + str(pca2.explained_variance_ratio_.sum()*100) + "%" )



#%%
"""
    Problema 3.c
"""
from sklearn.cluster import KMeans as KM

"""
    Primero aplicaremos PCA con 2 componentes para poder graficar los datos en IR2 con un 60% 
    de la informacion aproxiadamente.
"""

pca = PCA(n_components = 2)
escala = MinMaxScaler()
df_escal = escala.fit_transform(df) # estandarizamos los datos con valores entre 0 y 1
PC = pca.fit_transform(df_escal)
dt_PC = pd.DataFrame( pca.fit_transform(df_escal) )
dt_PC.columns = ['PC1', 'PC2']

#%%

plt.figure(figsize = (9,6))
def codo(data, maxK, seed_centroids=None):
    sse = {}  # sse es un OBJETO
    for k in range(1, maxK):
        if seed_centroids is not None:
            seeds = seed_centroids.head(k)
            kmeans = KM(n_clusters=k, max_iter=500, n_init=100, random_state=0, init=np.reshape(seeds, (k,1))).fit(data)
            data["clusters"] = kmeans.labels_
        else:
            kmeans = KM(n_clusters=k, max_iter=300, n_init=100, random_state=0).fit(data)
            data["clusters"] = kmeans.labels_
        # Inertia: Sum of distances of samples to their closest cluster center
        sse[k] = kmeans.inertia_
    plt.figure(figsize = (9,6))
    plt.plot(list(sse.keys()), list(sse.values()),'o-')
    plt.title('Codo')
    plt.xlabel('Número de clasters')
    plt.ylabel('Inercia')
    plt.grid() , plt.show()
    return sse

inercias = codo(dt_PC, maxK = 15)

"""
    Vemos del grafico que entre 4 y 6 cluster aproximadamente es lo optimo para
    poder clasificar los datos.  Consideremos 5 clusters
"""


#%%
"""
    Implementamos Método de K-means con k = 5 clusters
"""
k = 5

km = KM(n_clusters = k, max_iter = 3000, tol = 0.001).fit(PC)
labels = km.predict(PC)
dt_PC['Cluster'] = km.labels_
centroids = km.cluster_centers_

#%%
colors = ['mo', 'ro', 'co', 'yo', 'bo', 'ko']
(fig, ax) = plt.subplots(figsize = (10,9))
plt.grid()
plt.scatter(centroids[:,0], centroids[:,1], marker = 'X', color = 'k', s = 400)
for i in range(PC.shape[0]):
    plt.plot(PC[i,0], PC[i,1], colors[labels[i]], markersize = 5)
#%%


