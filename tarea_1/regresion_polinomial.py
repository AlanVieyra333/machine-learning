import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random
from sklearn.preprocessing import PolynomialFeatures

random.seed(0)
pf = PolynomialFeatures(degree = 6)                         # Polinomios de grado 2-10

X_train = np.arange(10)
X_train_poly = pf.fit_transform(X_train.reshape(-1,1))      # Transformar la entrada en polinómica
Y_train = np.zeros(10)
for i, x in enumerate(X_train):
    Y_train[i] = np.sin(2.0*np.pi * x /10.0) + random.uniform(0.0, 0.5)

print('Datos de entrenamiento: ')
print(X_train)
print(Y_train)


X_test = np.arange(10) + random.uniform(0.0, 0.5)
X_test_poly = pf.fit_transform(X_test.reshape(-1,1))      # Transformar la entrada en polinómica
Y_test = np.zeros(10)
for i, x in enumerate(X_test):
    Y_test[i] = np.sin(2.0*np.pi * x /10.0) + random.uniform(0.0, 0.5)

print('Datos de prueba: ')
print(X_test)
print(Y_test)

# Fase de entrenamiento.
lr = linear_model.LinearRegression()
# Entrenar el modelo.
lr.fit(X_train_poly, Y_train)
# Realizar una predicción
Y_pred = lr.predict(X_test_poly)

print('\nPrecisión del modelo: ', lr.score(X_train_poly, Y_train))
print('Exactitud del modelo: ', lr.score(X_test_poly, Y_test))
# The coefficients
print('\nCoefficientes: ', lr.coef_)
print('Termino independiente: ', lr.intercept_)

print('## Error de ajuste')
# Error cuadratico medio
print('Error cuadratico medio: %.6f'
      % mean_squared_error(Y_train, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('R2: %.6f'
      % r2_score(Y_train, Y_pred))

print('## Error de prueba')
# Error cuadratico medio
print('Error cuadratico medio: %.6f'
      % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('R2: %.6f'
      % r2_score(Y_test, Y_pred))

def f(x):
  X_poly = pf.fit_transform(x.reshape(-1,1))
  return lr.predict(X_poly)

x_plot = np.linspace(0, 10, 1000)

plt.scatter( X_train, Y_train)
plt.scatter( X_test, Y_test)
plt.plot(x_plot, f(x_plot), color='red', linewidth=3)
plt.xlabel( "X" )
plt.ylabel( "Y" )
# plt.axis( 'equal' )
plt.show( )
