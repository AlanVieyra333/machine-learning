# encoding: utf-8
#
# Un clasificador binario realizado
# con una regresión logística
#
# Fraga 23.1.2020
#
import numpy as np
from sklearn import datasets
from sklearn import model_selection
import cv2 
from sklearn import metrics

iris = datasets.load_iris( )

# a = dir( iris )
# print( a )
# ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']

# Son tres clase, entonces tomamos las primeras
# dos para crear un problema de clasifición binario

vindex = iris.target != 2
datos = iris.data[ vindex ].astype( np.float32 )
target = iris.target[ vindex ].astype( np.float32 )

# Ya tenemos nuestro problema de clasificación binario
# Separamos los datos en dos conjuntos:
# datos de entrenamiento y
# datos de prueba
Xentrena, Xprueba, yentrena, yprueba = model_selection.train_test_split( datos, target, test_size = 0.1, random_state = 10 )

# print( type( Xentrena), type( Xprueba), type( yentrena ), type( yprueba ) )
# print( Xentrena.shape, Xprueba.shape, yentrena.shape, yprueba.shape )

regLog = cv2.ml.LogisticRegression_create( )

regLog.setTrainMethod( cv2.ml.LogisticRegression_MINI_BATCH )
regLog.setMiniBatchSize( 1 )
regLog.setIterations( 100 )
regLog.train( Xentrena, cv2.ml.ROW_SAMPLE, yentrena )
modelo = regLog.get_learnt_thetas()
print( modelo )

ret, ypredichas = regLog.predict( Xentrena )
exactitud = metrics.accuracy_score( yentrena, ypredichas )
print( "Entrenamiento:", exactitud )


ret, ypredichas = regLog.predict( Xprueba )
precision = metrics.accuracy_score( yprueba, ypredichas )
print( "Prueba:", precision )
