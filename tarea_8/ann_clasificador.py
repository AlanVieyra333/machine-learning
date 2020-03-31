import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as m
import sklearn.neural_network as nn
from sklearn import datasets
from sklearn.model_selection import cross_val_score, train_test_split

np.set_printoptions(precision=6, suppress=True)
np.random.seed(12345)

# El modelo
iris = datasets.load_iris()

X = iris.data.astype(np.float32)  # 4 Caracteristicas.
Y = iris.target.astype(np.int)    # 3 Clases.

# Partir los datos en grupos de entrenamiento y prueba.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8)
# print(X_train.shape, Y_train.shape)

# Ya tenemos los datos de entrenamiento y de prueba
# Escogemos el modelo de redes neuronales de retropropagación
# 1 a 10 neuronas en la capa oculta
etapas = 10
mean = 0
std = 100.0
n = 0

for nAux in range(1, 11):
    model = nn.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(nAux,))

    # Validación cruzada.
    scores = cross_val_score(model, X_train, Y_train,
                             cv=etapas, scoring='accuracy')
    print(nAux, scores)

    # Grafica
    plt.scatter(np.repeat(nAux, etapas), scores)

    print('  Media: %.6f' % scores.mean())
    print('  Desviación estándar: %.6f' % scores.std())
    if scores.mean() > mean and scores.std() < std:
        mean = scores.mean()
        std = scores.std()
        n = nAux

print('\nNúmero de neuronas en la capa oculta =', n)
print('Media: %.6f' % mean)
print('Desviación estándar: %.6f' % std)

# Etapa de prueba
model = nn.MLPClassifier(solver='lbfgs', hidden_layer_sizes=(n,))
model.fit(X_train, Y_train)

exactitud = model.score(X_train, Y_train)
precision = model.score(X_test, Y_test)

print('Exactitud: ', exactitud)
print('Precision: ', precision)

# Grafica.
plt.xlabel('Número de neuronas en la capa oculta')
plt.ylabel('Exactitud del clasificador')
plt.show()
