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
import matplotlib.pyplot as plt

iris = datasets.load_iris( )

# a = dir( iris )
# print( a )
# ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']

# Son tres clase, entonces tomamos las primeras
# dos para crear un problema de clasifición binario

vindex = iris.target != 2
datos1 = iris.data[ vindex ].astype( np.float32 )
target = iris.target[ vindex ].astype( np.float32 )

datos = datos1[ :, 0:2 ]
print( datos.shape )

# Ya tenemos nuestro problema de clasificación binario
# Separamos los datos en dos conjuntos:
# datos de entrenamiento y
# datos de prueba
Xentrena, Xprueba, yentrena, yprueba = model_selection.train_test_split( datos, target, test_size = 0.1, random_state = 9 )

# print( type( Xentrena), type( Xprueba), type( yentrena ), type( yprueba ) )
# print( Xentrena.shape, Xprueba.shape, yentrena.shape, yprueba.shape )

regLog = cv2.ml.LogisticRegression_create( )

regLog.setTrainMethod( cv2.ml.LogisticRegression_MINI_BATCH )
regLog.setMiniBatchSize( 1 )
regLog.setIterations( 100 )

# print( cv2.ml.LogisticRegression_REG_DISABLE )
# print( cv2.ml.LogisticRegression_REG_L1 )
print( cv2.ml.LogisticRegression_REG_L2 )
print( cv2.ml_LogisticRegression.getRegularization(regLog) )
# Está por defecto habilitada la regulariación L2
# regLog.setRegularization( cv2.ml.LogisticRegression_REG_L2 )

r = regLog.train( Xentrena, cv2.ml.ROW_SAMPLE, yentrena )
print ( "Resp:", r )
modelo = regLog.get_learnt_thetas()
print( modelo, modelo.shape )

ret, ypredichas = regLog.predict( Xentrena )
exactitud = metrics.accuracy_score( yentrena, ypredichas )
print( "Entrenamiento:", exactitud )
print( ypredichas )


ret, ypredichas = regLog.predict( Xprueba )
precision = metrics.accuracy_score( yprueba, ypredichas )
print( "Prueba:", precision )

theta0 = modelo[0,0] 
theta1 = modelo[0,1] 
theta2 = modelo[0,2] 
print( "Thetas:", theta0, theta1, theta2 )

# Calculamos y = t0 + t1 * x1 + t2 * x2
# X \in nx2 2x1 = nx1
vthetas = np.zeros( (2,1), dtype=np.float32 )
vthetas[0,0] = theta1
vthetas[1,0] = theta2

yp = np.matmul( Xentrena,  vthetas ) + theta0
phis = 1.0/(1.0 + np.exp(-yp ))
print( phis )

# la grafica de los datos y el modelo aprendido
plt.ioff()
plt.scatter( Xentrena[ :,0], Xentrena[ :,1] )
plt.plot( Xentrena[ :,0], (- theta0 - theta1*Xentrena[ :,0])/theta2 )
plt.show( )

# pydoc sklearn.linear_model.LogisticRegression
