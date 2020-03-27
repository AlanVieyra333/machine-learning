import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as m
import sklearn.neural_network as nn
from sklearn.model_selection import cross_val_score

samples=40

# El modelo
vx1 = np.linspace( 0, 10, samples )

np.random.seed( 1234 )
# vy_ver = np.sin( vx/(20*np.pi) ) + (np.random.rand( vx.size ) - 0.5)*0.2
vy_entrena = np.sin( 2.0*np.pi*vx1/10.0 ) #+ (np.random.rand( vx1.size ) - 0.5)*0.2
# print( vy_entrena )
# plt.scatter( vx1, vy_entrena, marker='o' )

vx2 = np.linspace( 0, 10, 11 )
vy_prueba = np.sin( 2.0*np.pi*vx2/10.0 ) #+ (np.random.rand( vx2.size ) - 0.5)*0.1
# print( vy_prueba )
# plt.scatter( vx2, vy_prueba, marker='+' )

# plt.show( )
# sys.exit(1)

vr = np.zeros( (10,30) )

etapas = 10
media = -1000
std = 100.0

# Ya tenemos los datos de entrenamiento y de prueba
# Escogemos el modelo de redes neuronales de retropropagación
# 1 a 10 neuronas en la capa oculta
n=1
for nAux in range(1,11):
	clf = nn.MLPRegressor(solver='lbfgs', hidden_layer_sizes=(nAux,) )

    # Validación cruzada.
    # Con cv=5, tenemos 5 pruebas
	mediciones = cross_val_score( clf, vx1.reshape(samples,1), vy_entrena, cv=etapas, scoring="neg_mean_squared_error")
	print(nAux, mediciones * -1)
	# Grafica
	plt.scatter( np.repeat(nAux, etapas), mediciones*-1)

	print("  Media: %.8f" % -mediciones.mean())
	print("  Desviación estándar: %.8f" % mediciones.std())
	if mediciones.mean() > media and mediciones.std() < std:
		media = mediciones.mean()
		std = mediciones.std()
		n = nAux

print('\nNúmero de neuronas en la capa oculta =', n)
print('Media: %.8f' % -media)
print('Desviación estándar: %.8f' % std)

# Etapa de prueba
modelo = nn.MLPRegressor(solver='lbfgs', hidden_layer_sizes=(n,) )
modelo.fit(vx1.reshape(samples,1), vy_entrena)

r2_train = modelo.score(vx1.reshape(samples,1), vy_entrena)
r2_test = modelo.score(vx2.reshape(11,1), vy_prueba)

print('\nR2 entrenamiento: %.8f' % r2_train)
print('R2 prueba: %.8f' % r2_test)

# Grafica.
plt.xlabel( "Número de neuronas en la capa oculta" )
plt.ylabel( "Error cuadratico medio" )

plt.show( )

