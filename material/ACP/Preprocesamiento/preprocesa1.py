import numpy as np
import sklearn.preprocessing as preprocessing 

X = np.array( [[ 1.0, -2.0,  2.0 ],
               [ 3.0,  0.0,  0.0 ],
               [ 0.0,  1.0, -1.0 ]] )
print( X, "\n" )
X_escalada = preprocessing.scale( X )
print( X_escalada )

# Calculamos las medias
medias = X_escalada.mean( axis=0 )
print( "Medias:", medias )

# Calculamos las desviaciones estándar
std = X_escalada.std( axis=0 )
print( "Std:", std )
