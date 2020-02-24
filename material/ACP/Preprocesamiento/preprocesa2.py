import numpy as np
import sklearn.preprocessing as preprocessing 

X = np.array( [[ 1.0, -2.0,  2.0 ],
               [ 3.0,  0.0,  0.0 ],
               [ 0.0,  1.0, -1.0 ]] )

# Normalizamos los datos
print( X, "\n" )
X_normalizada = preprocessing.normalize( X, axis=0 )
print( X_normalizada )

