import numpy as np
import sklearn.preprocessing as preprocessing 

X = np.array( [[ np.nan,  0,  3.0 ],
               [ 3.0,    0.0,  0.0 ],
               [ 0.0,  np.nan, -1.0 ]] )

print( X, "\n" )

# Cómo quitar los 'NaN's

imp = preprocessing.Imputer( strategy='mean' )
X2 = imp.fit_transform( X )

print( X2 )

