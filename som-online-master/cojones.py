import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines

#link1 = '~/Escritorio/Carpetita/5to/redes neuronales/mnist/mnist_train.csv'


link1 = '~/Escritorio/Carpetita/5to/redes neuronales/som-online-master/paisesG.csv'

# Cargar datos
input_data = pd.read_csv(link1)

# Ver unos pocos datos
input_data.iloc[:30,:]

# Revolver los datos
agri_data = input_data.iloc[np.random.permutation(len(input_data))]
trunc_data = agri_data[["IDG","MortalidadMaterna","Posición","EnParlamento"]]
trunc_data
trunc_data.iloc[:20,:]

# Normalizarlos 
trunc_data = trunc_data / trunc_data.max()
trunc_data.iloc[:10,:]


from com.machinelearningnepal.som.online_som import SOM

# som = SOM(x_size, y_size, num_features)
agri_som = SOM(3,2,4)

# Pesos iniciales
init_fig = plt.figure()
agri_som.show_plot(init_fig, 1, 0)
plt.show()

agri_som.train(trunc_data.values,num_epochs=200,init_learning_rate=0.01)

def predict(df):
    bmu, bmu_idx = agri_som.find_bmu(df.values)
    df['bmu'] = bmu  		#unidad de mejor correspondencia 
    df['bmu_idx'] = bmu_idx  # el índice de la unidad de mejor correspondencia 
    return df
clustered_df = trunc_data.apply(predict, axis=1)
clustered_df.iloc[0:20]

joined_df = agri_data.join(clustered_df, rsuffix="_norm")
joined_df[0:20]


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
    color = row['bmu'][0]
    marker = "$\\ " + row['País'][0:3]+"$" 	#Dibuja la primera letra del campo crop.
    marker = marker.lower()
    ax.plot(x_cor, y_cor, color=color, marker=marker, markersize=10)
    label = row['País']
    if not label in legend_map:
        legend_map[label] =  mlines.Line2D([], [], color='black', marker=marker, linestyle='None',
                          markersize=10, label=label)
plt.legend(handles=list(legend_map.values()), bbox_to_anchor=(1, 1))
plt.show()



A = agri_som.net

a=A[:,1]-A[:,0]

np.linalg.det(a)






