import numpy as np
import sklearn.preprocessing as preprocessing 

X = np.array( [[ 1.0, -2.0,  2.0 ],
               [ 3.0,  0.0,  0.0 ],
               [ 0.0,  1.0, -1.0 ]] )

print( X, "\n" )

# Para binarizar los datos

binariza = preprocessing.Binarizer( threshold=0.5 )


X_bin = binariza.transform( X )
print( X_bin )

