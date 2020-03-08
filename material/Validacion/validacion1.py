from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score
import numpy as np

#   Falta particionar los datos en los dos grupos
# de entrenamiento y prueba 
#   ¿Se podrían usar las funciones de OpenCV?
#   ¿Cómo cambiar la medida del estimador?
#   ¿Qué medida del clasificador está realizando?

iris = datasets.load_iris( )

# a = dir( iris )
# print( a )
# ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']

# Son tres clase, entonces tomamos las primeras
# dos para crear un problema de clasifición binario

vindex = iris.target != 2
datos1 = iris.data[ vindex ].astype( np.float32 )
y = iris.target[ vindex ].astype( np.int )

X = datos1[ :, 0:2 ]

modelo = KNeighborsClassifier( n_neighbors=1 )

mediciones = cross_val_score( modelo, X, y, cv=5 )

# Con cv=5, tenemos 5 pruebas:
print( mediciones )

print( mediciones.mean(), mediciones.std( ) )
