from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import metrics

X, y = datasets.make_blobs(100, 2, centers=2, random_state=123, 
cluster_std=2)

Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=1)

# print(Xtest)

# Bayes naive
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(Xtrain, Ytrain)

# Metricas
Ypredict = clf.predict(Xtrain)
exactitud = metrics.accuracy_score(Ytrain, Ypredict)
print("Exactitud:", exactitud)

Ypredict = clf.predict(Xtest)
precision = metrics.accuracy_score(Ytest, Ypredict)
print("Precision:", precision)
# print('Ingenuo', YPredict)

# Bayes Gaussian
import cv2
Xtrain = np.array(Xtrain, dtype=np.float32)
Ytrain = np.array(Ytrain, dtype=np.int32)
Xtest = np.array(Xtest, dtype=np.float32)
Ytest = np.array(Ytest, dtype=np.int32)

modelBG = cv2.ml.NormalBayesClassifier_create()
modelBG.train(Xtrain, cv2.ml.ROW_SAMPLE ,Ytrain)

# Metricas
_, Ypredict = modelBG.predict(Xtrain)
exactitud = metrics.accuracy_score(Ytrain, Ypredict)
print("Exactitud:", exactitud)

_, Ypredict = modelBG.predict(Xtest)
precision = metrics.accuracy_score(Ytest, Ypredict)
print("Precision:", precision)
# print('Gaussian', Ypredict)

# Grafica
# plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, s=50 )
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=Ytest, s=50 )
# plt.scatter(Xtest[:, 0], Xtest[:, 1], c=Ypredict[:,0] )
plt.show()