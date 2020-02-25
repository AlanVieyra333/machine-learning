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
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import statistics

iris = datasets.load_iris( )

# a = dir( iris )
# print( a )
# ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']
#print(iris.target, iris.target_names)
# Son tres clase, entonces tomamos las primeras
# dos para crear un problema de clasifición binario

vindex = iris.target != 3
datos1 = iris.data[ vindex ].astype( np.float32 )
target = iris.target[ vindex ].astype( np.float32 )

datos = datos1[ :, 0:4 ]
#datos = datos1

print( datos.shape )

# Ya tenemos nuestro problema de clasificación binario
# Separamos los datos en dos conjuntos:
# datos de entrenamiento y
# datos de prueba
Xentrena, Xprueba, yentrena, yprueba = model_selection.train_test_split( datos, target, test_size = 0.1, random_state = 9 )

# print( type( Xentrena), type( Xprueba), type( yentrena ), type( yprueba ) )
print( Xentrena.shape, Xprueba.shape, yentrena.shape, yprueba.shape )

# plt.plot( Xentrena[:,2], Xentrena[:,3], 'o', zorder=1 )
# plt.xlabel( "Característica 1" )
# plt.ylabel( "Característica 2" )
# plt.axis( 'equal' )

y = np.atleast_2d(yentrena).T
X = np.append( Xentrena, y, axis=1 )
# X = Xentrena

mu, eig = cv2.PCACompute( X, np.array( [] ), maxComponents=1)
print( mu )
print( eig )

# Calcular eigenvalores (energia)
XMedia0 = X-mu
covar = np.dot(np.transpose(XMedia0), XMedia0)
# print(covar.shape)

eVal, eVec = cv2.eigen(covar)[1:]
print('Eigenvalores')
print(eVal.shape, eVec.shape)
print(eVal, eVec)

# Proyeccion PCA
# print(mu[:,:2], eig[:2,:])
X2 = cv2.PCAProject(X, mu, eig)
print(X2.shape)

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# ax.scatter3D(X2[:,0], X2[:,1], X2[:,2], c=yentrena, cmap=plt.cm.Paired)
# print(np.zeros((90,1)).shape)
# plt.scatter( X2[:,0], np.zeros((135,1)), c=yentrena, cmap=plt.cm.Paired)

plt.plot(X2[yentrena == 0,0], np.zeros((135,1))[yentrena == 0], 'bo', mfc='none')
plt.plot(X2[yentrena == 1,0], np.zeros((135,1))[yentrena == 1], 'ro', )
plt.plot(X2[yentrena == 2,0], np.zeros((135,1))[yentrena == 2], 'yo', mfc='none')
plt.xlabel( "PC 1" )
# plt.ylabel( "PC 2" )
# ax.set_zlabel('PC 3')
plt.axis( 'equal' )

plt.show( ) 