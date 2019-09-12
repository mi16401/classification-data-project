import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/content/data.csv")
print(df.head)

#STANDARDIZACIJA PODATAKA
x = df.loc[:, df.columns != 'class'].values
y = df.loc[:,['class']].values
x = StandardScaler().fit_transform(x)

#PCA projekcija u 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
pDf = pd.DataFrame(data = principalComponents,columns = ['pc1', 'pc2'])
print(pDf.head)

print(df[['class']].head)

finalDf = pd.concat([pDf, df[['class']]], axis = 1)
finalDf.head(5)

#vizuelizacija 2D projekcije
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['prva', 'druga', 'treca', 'cetvrta']
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
               , finalDf.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.savefig('plot_data.png', bbox_inches='tight')
