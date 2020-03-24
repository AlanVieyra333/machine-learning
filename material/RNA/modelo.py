import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as m
import sklearn.neural_network as nn
#import sys

# El modelo
vx1 = np.linspace( 0, 10, 20 )

np.random.seed( 1234 )
# vy_ver = np.sin( vx/(20*np.pi) ) + (np.random.rand( vx.size ) - 0.5)*0.2
vy_entrena = np.sin( 2.0*np.pi*vx1/10.0 ) + (np.random.rand( vx1.size ) - 0.5)*0.2
# print( vy_entrena )
# plt.scatter( vx1, vy_entrena, marker='o' )

vx2 = np.linspace( 0, 10, 11 )
vy_prueba = np.sin( 2.0*np.pi*vx2/10.0 ) + (np.random.rand( vx2.size ) - 0.5)*0.1
# print( vy_prueba )
# plt.scatter( vx2, vy_prueba, marker='+' )

# plt.show( )
# sys.exit(1)

vr = np.zeros( (10,30) )
# Ya tenemos los datos de entrenamiento y de prueba
# Escogemos el modelo de redes neuronales de retropropagación
# 1 a 10 neuronas en la capa oculta
n = 1
while n <= 10 :
	# Repretimos el experimento 30 veces para cada arquitectura

	print( n )
	i = 0
	while i<30 :	
		clf = nn.MLPRegressor(solver='lbfgs', hidden_layer_sizes=(n,) )

		clf.fit( vx1.reshape(20,1), vy_entrena )
		vy_predichas = clf.predict( vx2.reshape(11,1)  )
		# print( vy_predichas.shape )
		vr[n-1,i] = m.mean_squared_error( vy_prueba, vy_predichas )

		i += 1

	n += 1

n = 0
while n < 10 :
	x = [n+1]*30
	plt.scatter( x, vr[n,:] )  
	n += 1

plt.show( )

