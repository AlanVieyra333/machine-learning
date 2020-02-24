import numpy as np
import sklearn.preprocessing as preprocessing 

X = np.array( [[ 1.0, -2.0,  2.0 ],
               [ 3.0,  0.0,  0.0 ],
               [ 0.0,  1.0, -1.0 ]] )

# Para escalar los datos
print( X, "\n" )

minmax = preprocessing.MinMaxScaler( feature_range=(-10,10) )


X_minmax = minmax.fit_transform( X )
print( X_minmax )

