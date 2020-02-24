import sys
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing

A = pd.read_csv( 'enteros.csv' )
print( A )
print( type(A), A.shape )

# X = np.asarray( list(csv.reader(arch, csv.QUOTE_NONE) ), dtype=np.int )
# X = np.array( A, dtype=np.float, copy=True )
# print( X.shape )
# print( X ) 

# imp = preprocessing.Imputer( missing_values=-1, strategy='mean' )
#X2 = imp.fit_transform( X )

#print( X2 )
