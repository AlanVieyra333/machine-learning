import numpy as np
import sklearn.preprocessing as preprocessing 

# 13.02.2020
#
# ¿Cómo llenar los datos enteros faltantes?
# Leer una matriz con datos faltantes de un
# archivo CVS

X = np.array( [[ 'nil',  0,  3 ],
               [ 3, 0,  0 ],
               [ 0,  'nil', -1 ]] )

print( X, "\n" )

# Cómo quitar los 'NaN's

# imp = preprocessing.Imputer( strategy='mean' )
# X2 = imp.fit_transform( X )

# print( X2 )

