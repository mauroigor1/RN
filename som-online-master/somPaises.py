import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines




#dt=pd.read_csv("paisesG.csv")
dt=pd.read_csv("vgsales.csv")

#trunc_data = dt[["MortalidadMaterna","FecundidadAdos","EnParlamento"]]
trunc_data = dt[["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]]
trunc_data=trunc_data.iloc[:100,:]

# normalizing data
trunc_data = trunc_data / trunc_data.max()
trunc_data.iloc[:10,:]


from com.machinelearningnepal.som.online_som import SOM

# som = SOM(x_size, y_size, num_features)
agri_som = SOM(10,10,4)

# Initial weights
init_fig = plt.figure()
agri_som.show_plot(init_fig, 1, 0)
plt.show()

agri_som.train(trunc_data.values,num_epochs=200,init_learning_rate=0.01)

def predict(df):
    bmu, bmu_idx = agri_som.find_bmu(df.values)
    df['bmu'] = bmu
    df['bmu_idx'] = bmu_idx
    return df

def matrixU(m):
	mu = np.zeros((m.shape[0]-2,m.shape[1]-2))
			
	for i in range(0,m.shape[0]-2):
		for j in range(0,m.shape[1]-2):
			mu[i,j] = np.linalg.norm(m[i,j]-m[i,j+1])+np.linalg.norm(m[i,j]-m[i,j-1])+np.linalg.norm(m[i,j]-m[i+1,j])+np.linalg.norm(m[i,j]-m[i-1,j])
	return mu

clustered_df = trunc_data.apply(predict, axis=1)
clustered_df.iloc[0:20]

joined_df = dt.join(clustered_df, rsuffix="_norm")
joined_df[0:20]


### VER

from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines

fig = plt.figure(figsize=(10, 8))
# setup axes
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
    x_cor = row['bmu_idx'][0] * scale
    y_cor = row['bmu_idx'][1] * scale
    x_cor = np.random.randint(x_cor, x_cor + scale)
    y_cor = np.random.randint(y_cor, y_cor + scale)
    color = [0,0,0,1]#row['bmu'][0]
    marker = "$\\ " + row['Name'][:1]+"$"
    marker = marker.lower()
    ax.plot(x_cor, y_cor, color=color, marker=marker, markersize=20)
    #label = row['Name']
    #if not label in legend_map:
    #    legend_map[label] =  mlines.Line2D([], [], color='black', marker=marker, linestyle='None',
    #                      markersize=10, label=label)
plt.legend(handles=list(legend_map.values()), bbox_to_anchor=(1, 1))
plt.show()