# encoding: utf-8
#
# Clasificador con regresión logística aplicado a
# un conjunto de datos cuya separación no es lineal.
#
# Vieyra 3.2.2020
#
import numpy as np
from sklearn import datasets
from sklearn import model_selection
import cv2
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# DATOS DE ENTRENAMIENTO
dataset = make_blobs(n_samples=450, n_features=2, centers=3,
                     cluster_std=1.4, random_state=31)

data = dataset[0].astype(np.float32)
target = dataset[1].astype(np.float32)
target[target == 2] = 1

print(data.shape, target.shape)

Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(
    data, target, test_size=0.1, random_state=3)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# ENTRENAMIENTO
regLog = cv2.ml.LogisticRegression_create()
regLog.setLearningRate(0.005)
regLog.setIterations(100)
regLog.setRegularization(cv2.ml.LogisticRegression_REG_L2)
regLog.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
regLog.setMiniBatchSize(1)

r = regLog.train(Xtrain, cv2.ml.ROW_SAMPLE, Ytrain)
print("Entrenamiento:", r)

vthetas = regLog.get_learnt_thetas()
theta0 = vthetas[0, 0]
theta1 = vthetas[0, 1]
theta2 = vthetas[0, 2]
print("Thetas:", vthetas.shape, vthetas)

ret, Ypredict = regLog.predict(Xtrain)
exactitud = metrics.accuracy_score(Ytrain, Ypredict)
print("Exactitud:", exactitud)

ret, Ypredict = regLog.predict(Xtest)
precision = metrics.accuracy_score(Ytest, Ypredict)
print("Precision:", precision)

# GRAFICACION
plts = []
plt.subplot(211)
plt.title("Separación no lineal")
plts += plt.plot(Xtrain[:, 0], (- theta0 - theta1 *
                                Xtrain[:, 0])/theta2, label="Separación lineal")
plts += plt.plot(Xtrain[Ytrain == 0][:, 0],
                 Xtrain[Ytrain == 0][:, 1], 'bo', label="Clase 1")
plts += plt.plot(Xtrain[Ytrain == 1][:, 0],
                 Xtrain[Ytrain == 1][:, 1], 'ro', label="Clase 2")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend(handles=plts, loc='lower right')

# SIGMOIDE
def sigmoide(t):
    return 1.0/(1.0 + np.exp(-t))
t = np.arange(-3, 3, 0.1)

yp = np.matmul( Xtrain,  vthetas[0,1:] ) + theta0
phis = 1.0/(1.0 + np.exp(-yp ))
#print( phis )

plts = []
plt.subplot(212)
plt.title( "Sigmoide" )
plts += plt.plot(t, sigmoide(t), label='Sigmoide')
#plt.plot(yp, phis, 'bo', label='Muestras')
plts += plt.plot(yp[Ytrain == 1], phis[Ytrain == 1], 'ro', label="Clase 2")
plts += plt.plot(yp[Ytrain == 0], phis[Ytrain == 0], 'bo', label="Clase 1")
plt.xlabel( "Y" )
plt.ylabel( "Phi (Y)" )
plt.gca().add_artist(plt.legend(handles=[plts[0]], loc='lower right'))
plt.legend(handles=[plts[2], plts[1]], loc='upper left')
plt.grid()
plt.subplots_adjust(hspace=0.4)

# plt.ioff()
plt.show()
