from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

#   Falta particionar los datos en los dos grupos
# de entrenamiento y prueba 
#   ¿Se podrían usar las funciones de OpenCV? 
#     Hay que particionar "manualmente" los datos
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

# Repetimos 100 veces el estimador con los
# conjuntos de datos de entrenamiento y prueba seleccionados
# aleatoriamente
n = 500
mediciones = np.zeros( (n,1) )
i=0
while i < n :
	X_prueba1, X_prueba2, y_prueba1, y_prueba2 = train_test_split( X, y, train_size=0.8 )
	# Usamos el algotimo de k vecinos
	modelo.fit( X_prueba1, y_prueba1 )
	# Medimos su precisión
	y_predichas = modelo.predict( X_prueba2 )
	exactitud = metrics.accuracy_score( y_prueba2, y_predichas )
	mediciones[i] = exactitud
	i += 1

print( len( mediciones ) )
print( mediciones.mean(), mediciones.std( ) )
