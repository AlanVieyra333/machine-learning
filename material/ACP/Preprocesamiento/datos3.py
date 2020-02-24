import sys
import numpy as np
import sklearn.preprocessing as preprocessing

# A = np.loadtxt( 'enteros.csv', delimiter=',', dtype='str,int' )
# A = np.loadtxt( 'enteros.csv', delimiter=',', unpack=True )
X = np.loadtxt( 'enteros.csv', delimiter=',' )
print( X )

# imp = preprocessing.Imputer( missing_values=-1, strategy='mean' )
X2 = imp.fit_transform( X )

print( X2 )
