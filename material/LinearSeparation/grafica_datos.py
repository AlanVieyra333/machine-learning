# encoding: utf-8
#
# Un clasificador binario realizado
# con una regresión logística
#
# Fraga 23.1.2020
#
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris( )

# a = dir( iris )
# print( a )
# ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']

# Son tres clase, entonces tomamos las primeras
# dos para crear un problema de clasifición binario

vindex = iris.target != 2
datos = iris.data[ vindex ].astype( np.float32 )
target = iris.target[ vindex ].astype( np.float32 )
print(  np.shape( datos ), np.shape( target )  )

# Ya tenemos nuestro problema de clasificación binario
# Los graficamos:
plt.scatter( datos[:,0], datos[:,1], c=target, cmap=plt.cm.Paired, s=100 )
plt.xlabel( iris.feature_names[0] )
plt.ylabel( iris.feature_names[1] )
plt.show( )
