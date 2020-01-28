import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random

X_train = np.array( [np.arange(10)] ).transpose()
Y_train = np.zeros(10)
for i, x in enumerate(X_train):
    Y_train[i] = np.sin(2.0*np.pi * x /10.0) + random.uniform(0.0, 0.5)

print('Datos de entrenamiento: ')
print(X_train)
print(Y_train)

X_test = 10 * np.random.random_sample((10, 1))
Y_test = np.zeros(10)
for i, x in enumerate(X_test):
    Y_test[i] = np.sin(2.0*np.pi * x /10.0) + random.uniform(0.0, 0.5)

print('Datos de prueba: ')
print(X_test)
print(Y_test)

# Fase de entrenamiento.
lr = linear_model.LinearRegression()
# Entrenar el modelo.
lr.fit(X_train, Y_train)
# Realizar una predicción
Y_pred = lr.predict(X_test)

print('Precisión del modelo: ', lr.score(X_train, Y_train))
print('Exactitud del modelo: ', lr.score(X_test, Y_test))
# The coefficients
print('Coefficientes: ', lr.coef_)
print('Termino independiente: ', lr.intercept_)

print('## Error de ajuste')
# Error cuadratico medio
print('Error cuadratico medio: %.2f'
      % mean_squared_error(Y_train, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('R2: %.2f'
      % r2_score(Y_train, Y_pred))

print('## Error de prueba')
# Error cuadratico medio
print('Error cuadratico medio: %.2f'
      % mean_squared_error(Y_test, Y_pred))
# The coefficient of determination: 1 is perfect prediction
print('R2: %.2f'
      % r2_score(Y_test, Y_pred))

plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.scatter( X_train, Y_train)
plt.scatter( X_test, Y_test)
plt.xlabel( "X" )
plt.ylabel( "Y" )
plt.show( )
