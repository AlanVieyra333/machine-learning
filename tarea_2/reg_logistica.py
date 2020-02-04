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
#print(iris.target, iris.target_names)
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
print( Xentrena.shape, Xprueba.shape, yentrena.shape, yprueba.shape )

regLog = cv2.ml.LogisticRegression_create( )

regLog.setLearningRate(0.003)
regLog.setIterations(100)
regLog.setRegularization(cv2.ml.LogisticRegression_REG_L2)
regLog.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
regLog.setMiniBatchSize(1)

#params.alpha = 0.5;
#params.norm = LogisticRegression::REG_L2;

# print( cv2.ml.LogisticRegression_REG_DISABLE )
# print( cv2.ml.LogisticRegression_REG_L1 )
#print( cv2.ml.LogisticRegression_REG_L2 )
#print( cv2.ml_LogisticRegression.getRegularization(regLog) )
# Está por defecto habilitada la regulariación L2
# regLog.setRegularization( cv2.ml.LogisticRegression_REG_L2 )

r = regLog.train( Xentrena, cv2.ml.ROW_SAMPLE, yentrena )
print ( "Resp:", r )
modelo = regLog.get_learnt_thetas()
print( "Thetas:", modelo.shape, modelo )

ret, ypredichas = regLog.predict( Xentrena )
exactitud = metrics.accuracy_score( yentrena, ypredichas )
print( "Entrenamiento:", exactitud )
#print( ypredichas )

ret, ypredichas = regLog.predict( Xprueba )
precision = metrics.accuracy_score( yprueba, ypredichas )
print( "Prueba:", precision )


theta0 = modelo[0,0] 
theta1 = modelo[0,1] 
theta2 = modelo[0,2] 
#print( "Thetas:", theta0, theta1, theta2 )

# Calculamos y = t0 + t1 * x1 + t2 * x2
# X \in nx2 2x1 = nx1
vthetas = np.zeros( (2,1), dtype=np.float32 )
vthetas[0,0] = theta1
vthetas[1,0] = theta2
#vthetas[2,0] = modelo[0,3]
#vthetas[3,0] = modelo[0,4]

yp = np.matmul( Xentrena,  vthetas ) + theta0
phis = 1.0/(1.0 + np.exp(-yp ))
#print( phis )

def sigmoide(t):
    return 1.0/(1.0 + np.exp(-t))

t = np.arange(-1.5, 1.5, 0.01)

plts = []
plt.subplot(212)
plt.title( "Sigmoide" )
plts += plt.plot(t, sigmoide(t), label='Sigmoide')
#plt.plot(yp, phis, 'bo', label='Muestras')
plts += plt.plot(yp[yentrena == 1], phis[yentrena == 1], 'ro', label=iris.target_names[1])
plts += plt.plot(yp[yentrena == 0], phis[yentrena == 0], 'bo', label=iris.target_names[0])
plt.xlabel( "Y" )
plt.ylabel( "Phi (Y)" )
plt.gca().add_artist(plt.legend(handles=[plts[0]], loc='lower right'))
plt.legend(handles=[plts[2], plts[1]], loc='upper left')
plt.grid()

# la grafica de los datos y el modelo aprendido
plts = []
plt.subplot(211)
plt.title( "Separación lineal" )
plts += plt.plot( Xentrena[:,0], (- theta0 - theta1*Xentrena[:,0] )/theta2, label="Separación lineal" )
plts += plt.plot( Xentrena[yentrena == 0][:,0], Xentrena[yentrena == 0][:,1], 'bo', label=iris.target_names[0])
plts += plt.plot( Xentrena[yentrena == 1][:,0], Xentrena[yentrena == 1][:,1], 'ro', label=iris.target_names[1])
plt.xlabel( iris.feature_names[0] )
plt.ylabel( iris.feature_names[1] )
#plt.legend(loc='upper center', ncol=3)
plt.gca().add_artist(plt.legend(handles=[plts[0]], loc='lower right'))
plt.legend(handles=[plts[1], plts[2]], loc='upper left')

#plt.ioff()
plt.subplots_adjust(hspace=0.3)
plt.show( )

# pydoc sklearn.linear_model.LogisticRegression