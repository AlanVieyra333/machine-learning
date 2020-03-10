from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data.astype(np.float32)  # 4 Caracteristicas.
Y = iris.target.astype(np.int)    # 3 Clases.

# Particionar los datos en grupos de entrenamiento y prueba.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, random_state=20)

# Algoritmo de entrenamiento. K-NN
k = 0
media = 0
variabilidad = 100.0

# Elegir el mejor valor de K, usando validación cruzada.
for kAux in range(1,11):
    modelo = KNeighborsClassifier(n_neighbors=kAux)

    # Validación cruzada.
    # Con cv=5, tenemos 5 pruebas
    mediciones = cross_val_score(
        modelo, X_train, Y_train, cv=5, scoring="accuracy")
    print(kAux, mediciones)
    # Grafica
    plt.scatter( np.repeat(kAux, 5), mediciones, c='b', s=5)

    if mediciones.mean() > media and mediciones.std() < variabilidad:
        media = mediciones.mean()
        variabilidad = mediciones.std()
        k = kAux

modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X_train, Y_train)

print('k =', k)
print('media', media)
print('variabilidad', variabilidad)

# Etapa de prueba
precision = modelo.score(X_train, Y_train)
exactitud = modelo.score(X_test, Y_test)

print('precision', precision)
print('exactitud', exactitud)

# Grafica.
plt.xlabel( "Valor de K" )
plt.ylabel( "Exactitud del clasificador" )
# plt.axis( 'equal' )

plt.show( ) 