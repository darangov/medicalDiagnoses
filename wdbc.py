# Arbol de Decision

# Librerias
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Cargo el dataset
wdbc_dataset = pd.read_csv("wdbc.csv", header=None) # Indico que el dataset no tiene encabezados o nombre de columnas

# Top five rows in the dataframe
print()
print ("Top five rows:\n", wdbc_dataset.head()) 

# Categories related to column 4 - its maybe a Categorical attribute
print("Categorias:\n", wdbc_dataset[4].value_counts())



# Description of the data
print()
print ("Data information:\n", iris.info())

print()
print ("Data describe:\n", iris.describe())


# Como la data siempre va a ser la misma (no cambia ni se incrementa o disminuye) usamos la funcion
# train_test_split para usar seed o random state garantizando que siempre genere los mismos indices

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(iris, test_size = 0.3, random_state = 42)
print()
print(len(train_set), "Train Set data +", len(test_set), "Test Set data")



