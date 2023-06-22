#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:40:01 2019

@author: mauri
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines

#link1 = '~/Escritorio/Carpetita/5to/redes neuronales/mnist/mnist_train.csv'

link = '~/Escritorio/Carpetita/5to/redes neuronales/T3RN/Parques_y_Jardines_Arbol_ZonaVerde.csv'





# Cargar datos
df = pd.read_csv(link)





nombre=df['Nombre'].value_counts()
nombres=nombre[:16]

s = [('Citrus aurantium'), ('Pinus pinea'),('Tipuana tipu'),('Melia azedarach'),('Celtis australis'),('Olea europaea'),('Jacaranda mimosifolia'),('Brachychiton populneus'),('Ulmus pumila'),('Ceratonia siliqua'),('Quercus ilex'),('Platanus x hybrida'),('Casuarina equisetifolia'),('Robinia pseudoacacia'),('Ulmus sp'),('Fraxinus excelsior')]
df = df[df.Nombre.isin(s)]

###############33
import random

s1 = [('Citrus aurantium')]
s2 = [('Pinus pinea')]
s3 =[('Tipuana tipu')]
s4 =[('Melia azedarach')]
s5 =[('Celtis australis')]
s6 = [('Olea europaea')]
s7 =[('Jacaranda mimosifolia')]
s8= [('Brachychiton populneus')]
s9= [('Ulmus pumila')]
s10=[('Ceratonia siliqua')]
s11=[('Quercus ilex')]
s12=[('Platanus x hybrida')]
s13=[('Casuarina equisetifolia')]
s14=[('Robinia pseudoacacia')]
s15=[('Ulmus sp')]
s16=[('Fraxinus excelsior')]

df1 = df[df.Nombre.isin(s1)]
df1 = df1.sample(n=50,random_state=1)

df2 = df[df.Nombre.isin(s2)]
df2 = df2.sample(n=50,random_state=1)

df3 = df[df.Nombre.isin(s3)]
df3 = df3.sample(n=50,random_state=1)

df4 = df[df.Nombre.isin(s4)]
df4 = df4.sample(n=50,random_state=1)

df5 = df[df.Nombre.isin(s5)]
df5 = df5.sample(n=50,random_state=1)

df6 = df[df.Nombre.isin(s6)]
df6 = df6.sample(n=50,random_state=1)

df7 = df[df.Nombre.isin(s7)]
df7 = df7.sample(n=50,random_state=1)

df8 = df[df.Nombre.isin(s8)]
df8 = df8.sample(n=50,random_state=1)

df9 = df[df.Nombre.isin(s9)]
df9 = df9.sample(n=50,random_state=1)

df10 = df[df.Nombre.isin(s10)]
df10 = df10.sample(n=50,random_state=1)

df11 = df[df.Nombre.isin(s11)]
df11 = df11.sample(n=50,random_state=1)

df12 = df[df.Nombre.isin(s12)]
df12 = df12.sample(n=50,random_state=1)

df13 = df[df.Nombre.isin(s13)]
df13 = df13.sample(n=50,random_state=1)


df14 = df[df.Nombre.isin(s14)]
df14 = df14.sample(n=50,random_state=1)

df15 = df[df.Nombre.isin(s15)]
df15 = df15.sample(n=50,random_state=1)

df16 = df[df.Nombre.isin(s16)]
df16 = df16.sample(n=50,random_state=1)
#df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16

input_data = pd.concat([df1, df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16], axis=0)

input_data=input_data.dropna()

###########################################################33
##############################################################

# Ver unos pocos datos
input_data.iloc[:30,:]



# rows = np.random.choice(input_data.Nombre.values, 200)
#df=df.drop(['Nombre'],axis=1)
#input_data=input_data.drop(['aux_arbol'],axis=1)
#input_data=input_data.drop(['Observ'],axis=1)
#input_data=input_data.drop(['Cat_Esp'],axis=1)
#input_data=input_data.iloc[:130,:]


#trunc_data = agri_data[["X","Y", "FID","xlo","ylo","Perimetro","Altura"]]
# Revolver los datos
agri_data = input_data.iloc[np.random.permutation(len(input_data))]
trunc_data = agri_data[["xlo","ylo","Perimetro","Altura"]]
trunc_data.iloc[:20,:]

# Normalizarlos
trunc_data = trunc_data / trunc_data.max()
trunc_data.iloc[:10,:]




from com.machinelearningnepal.som.online_som import SOM

# som = SOM(x_size, y_size, num_features)
agri_som = SOM(4,4,4)

# Pesos iniciales
init_fig = plt.figure()
agri_som.show_plot(init_fig, 1, 0)
plt.show()

agri_som.train(trunc_data.values,num_epochs=100,init_learning_rate=0.01)


def predict(df):
    bmu, bmu_idx = agri_som.find_bmu(df.values)
    df['bmu'] = bmu  		#unidad de mejor correspondencia 
    df['bmu_idx'] = bmu_idx  # el índice de la unidad de mejor correspondencia 
    return df


clustered_df = trunc_data.apply(predict, axis=1)
clustered_df.iloc[0:20]



c00=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==0 and x[1]==0)]
c01=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==0 and x[1]==1)]
c02=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==0 and x[1]==2)]
c03=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==0 and x[1]==3)]


c10=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==1 and x[1]==0)]
c11=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==1 and x[1]==1)]
c12=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==1 and x[1]==2)]
c13=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==1 and x[1]==3)]


c20=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==2 and x[1]==0)]
c21=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==2 and x[1]==1)]
c22=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==2 and x[1]==2)]
c23=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==2 and x[1]==3)]



c30=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==3 and x[1]==0)]
c31=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==3 and x[1]==1)]
c32=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==3 and x[1]==2)]
c33=clustered_df[clustered_df['bmu_idx'].apply(lambda x: x[0]==3 and x[1]==3)]







cc=clustered_df['bmu'] #len(754)x 
cc1=cc.as_matrix()

## bu ess de 754x4

############################################################
##########################################################33
"Constructing U-Matrix from SOM"
u_matrix = np.zeros(shape=(4,4), dtype=np.float64)







def matrixU(m):
	mu = np.zeros((m.shape[0]-2,m.shape[1]-2))
			
	for i in range(0,m.shape[0]-2):
		for j in range(0,m.shape[1]-2):
			mu[i,j] = np.linalg.norm(m[i,j]-m[i,j+1])+np.linalg.norm(m[i,j]-m[i,j-1])+np.linalg.norm(m[i,j]-m[i+1,j])+np.linalg.norm(m[i,j]-m[i-1,j])
	return mu

matrixU(cc1)

c000=c00['bmu']

xm = c000.tolist()
xmm=np.stack( xm, axis=0 )
xx = xmm.reshape(79,4)

matrixU(xx)



xx=c000.as_matrix()

xx.reshape(79,4)




xx = np.asmatrix(c000)

np.asarray(c000).reshape(-1)

matrixU(c000)



####################################################3
#######################################################333

m=clustered_df['bmu_idx']
matrixU(m)

joined_df = agri_data.join(clustered_df, rsuffix="_norm")
joined_df[0:20]




list(joined_df.columns.values)



### VER

from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines

fig = plt.figure(figsize=(10,8))
# Los ejes
ax = fig.add_subplot(111)
scale = 50
ax.set_xlim((0, agri_som.net.shape[0]*scale))
ax.set_ylim((0, agri_som.net.shape[1]*scale))
ax.set_title("Cash Crops Clustering by using SOM")

for x in range(0, agri_som.net.shape[0]):
    for y in range(0, agri_som.net.shape[1]):
        ax.add_patch(patches.Rectangle((x*scale, y*scale), scale, scale,
                     facecolor='white',
                     edgecolor='grey'))
legend_map = {}
        
for index, row in joined_df.iterrows():
    x_cor = row['bmu_idx'][0] * scale #Coordenadas de la neurona ganadora
    y_cor = row['bmu_idx'][1] * scale
    x_cor = np.random.randint(x_cor, x_cor + scale) #Lo dibuja aleatoriamente en esa celda.
    y_cor = np.random.randint(y_cor, y_cor + scale)
    #color = ['mo']
    color = [0,0,0,1]#row['bmu'][0]
    marker = "$\\ " + row['Nombre'][0:3]+"$" 	#Dibuja la primera letra del campo crop.
    marker = marker.lower()
    ax.plot(x_cor, y_cor, color=color, marker=marker, markersize=10)
    label = row['Nombre']
    if not label in legend_map:
        legend_map[label] =  mlines.Line2D([], [], color=color, marker=marker, linestyle='None',
                          markersize=10, label=label)
plt.legend(handles=list(legend_map.values()), bbox_to_anchor=(1, 1))
plt.show()




