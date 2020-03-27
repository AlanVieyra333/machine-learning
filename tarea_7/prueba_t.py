import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as m
import sklearn.neural_network as nn
from sklearn.model_selection import cross_val_score
import scipy.stats

samples=40

np.set_printoptions(precision=6, suppress=True)

# El modelo
vx1 = np.linspace( 0, 10, samples )

np.random.seed( 321 )
# vy_ver = np.sin( vx/(20*np.pi) ) + (np.random.rand( vx.size ) - 0.5)*0.2
vy_entrena = np.sin( 2.0*np.pi*vx1/10.0 ) #+ (np.random.rand( vx1.size ) - 0.5)*0.2
# print( vy_entrena )
# plt.scatter( vx1, vy_entrena, marker='o' )

vx2 = np.linspace( 0, 10, 11 )
vy_prueba = np.sin( 2.0*np.pi*vx2/10.0 ) #+ (np.random.rand( vx2.size ) - 0.5)*0.1
# print( vy_prueba )
# plt.scatter( vx2, vy_prueba, marker='+' )

# plt.xlabel( "x" )
# plt.ylabel( "f(x)" )
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
n_mediciones = [0] * 10
for nAux in range(1,11):
	clf = nn.MLPRegressor(solver='lbfgs', hidden_layer_sizes=(nAux,) )

    # Validación cruzada.
    # Con cv=5, tenemos 5 pruebas
	mediciones = cross_val_score( clf, vx1.reshape(samples,1), vy_entrena, cv=etapas, scoring="neg_mean_squared_error")
	print(nAux, mediciones * -1)
	n_mediciones[nAux-1] = mediciones * -1
	# Grafica
	plt.scatter( np.repeat(nAux, etapas), mediciones*-1)

	print("  Media: %.6f" % -mediciones.mean())
	print("  Desviación estándar: %.6f" % mediciones.std())
	if mediciones.mean() > media and mediciones.std() < std:
		media = mediciones.mean()
		std = mediciones.std()
		n = nAux

# print('n_mediciones', n_mediciones)
for i in range(1,11):
	for j in range(i+1,11):
		vx = n_mediciones[i-1]
		vy = n_mediciones[j-1]

		T, pvalue = scipy.stats.ttest_ind( vx, vy )
		# T, pvalue = scipy.stats.ttest_ind( vx, vy,  equal_var=False )
		
		if pvalue <= 0.05:	# Las secuencias son diferentes.
			print("Los conjuntos de n (%d, %d) son diferentes" % (i,j))
		# else:
		# 	print("Los conjuntos de n (%d, %d) son iguales" % (i,j))
		# print( T, "p=", pvalue, "\n")

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

