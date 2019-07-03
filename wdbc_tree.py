# Arbol de Decision

# Librerias
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_moons
from sklearn.tree import DecisionTreeClassifier

# Ya que el dataset descargado no viene con encabezado de columnas, pero luego obtengo los nombres
# de estas, puedo ligar los encabezados a los datos
# strencabezados: string con el nombre de las columnas, viene separado por comas 
# encabezados: separo el string por comas
strencabezados = 'id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst'
encabezados = strencabezados.split(',')
print("Cantidad encabezados:", len(encabezados)) # valido que la longitud de encabezados corresponda con el numero de columnas de datos

# Cargo el dataset
wdbc_dataset = pd.read_csv("wdbc.csv", header=None) # Indico que el dataset no tiene encabezados o nombre de columnas
wdbc_dataset.columns= encabezados # Llamo y asocio las columnas de acuerdo a la lista encabezados

#print(wdbc_dataset.head())

# Primeras 5 columnas del dataframe
print()
print ("Top five rows:\n", wdbc_dataset.head()) 

# Categorias del campo diagnosis - posiblemente atributo categorico
print()
print("Categorias en campo diagnosis:\n", wdbc_dataset["diagnosis"].value_counts())


# Description de la data
print()
print ("Data information:\n", wdbc_dataset.info())

print()
print ("Data describe:\n", wdbc_dataset.describe())

"""
# Como la data siempre va a ser la misma (no cambia ni se incrementa o disminuye) usamos la funcion
# train_test_split para usar seed o random state garantizando que siempre genere los mismos indices
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(wdbc_dataset, test_size = 0.3, random_state = 42)
print()
print(len(train_set), "Train Set data +", len(test_set), "Test Set data")
"""

# Funcion para determinar regiones de decision
def plot_decision_regions(X, y, classifier=None, resolution=0.02):
    """ Taken from Rashka's book """
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    x2_min, x2_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    
    if classifier is not None:
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor='black'
        )


#datasetold=pd.read_csv(r"/home/war-machine/Documentos/DIPLOMADO_EAFIT/0X_CLASIFICACION/DT/wbc.csv")

descriptores = ['id','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']
salida = ['diagnosis']

X = wdbc_dataset[descriptores]
#X=datasetold

y = wdbc_dataset.diagnosis=='M'
y *= 1

print()
#print("x {}", "y {}", x, y)

