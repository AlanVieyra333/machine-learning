import numpy as np
from svmutil import *
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt

# Datos
iris = datasets.load_iris()

vindex = iris.target != 3   # 3 Clases
datos = iris.data[ vindex ].astype(np.float32)
target = iris.target[ vindex ].astype(np.float32)
datos = datos[ :, 0:4 ]     # 4 Caracteristicas.
print(datos.shape)

Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(
    datos, target, test_size=0.1, random_state=9)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# Fase de entrenamiento
prob = svm_problem(Ytrain, Xtrain)
param = svm_parameter('-t 0 -c 200')   # -t 0  (kernel_type: linear)
model = svm_train(prob, param)
# svm_save_model('svm_test2.model',model)

support_vector_coefficients = model.get_sv_coef()
support_vectors = model.get_SV()

print('\nModelo:')
print('Vectores de soporte:\n', support_vectors)
print('\nVectores de soporte (coeficientes):\n', support_vector_coefficients)

# Fase de prueba
p_label, p_acc, p_val = svm_predict(Ytrain, Xtrain, model)
p_label, p_acc, p_val = svm_predict(Ytest, Xtest, model)

# Grafica
# plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, cmap='autumn')
# plt.axis( 'equal' )
# plt.show()