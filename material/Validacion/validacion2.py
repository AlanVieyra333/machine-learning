from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Si usamos los algoritmos de aprendizaje de OpenCV,
# "manualmente" se tienen que crear las particiones
# para la validación cruzada

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

X_prueba1, X_prueba2, y_prueba1, y_prueba2 = train_test_split( X, y, random_state=20, train_size=0.8 )

print( type( X_prueba1 ), len( X_prueba1 ) )
print( type( X_prueba2 ), len( X_prueba2 ) )
print( type( y_prueba1 ), len( y_prueba1 ) )
print( type( y_prueba2 ), len( y_prueba2 ) )

