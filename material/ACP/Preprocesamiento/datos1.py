import numpy as np
import sklearn.preprocessing as preprocessing

X = np.genfromtxt( "enteros.csv", dtype=np.int32, delimiter=',' )
print( X.shape )
print( X ) 

imp = preprocessing.Imputer( missing_values=-1, strategy='mean' )
X2 = imp.fit_transform( X )

print( X2 )
