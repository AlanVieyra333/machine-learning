import numpy as np
from sklearn import datasets

iris = datasets.load_iris( )

# a = dir( iris )
# print( a )
# ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']

# print( iris.DESCR )

# print( iris.feature_names )
# print( iris.filename )
# print( iris.target_names )
print( type( iris.data ) )
print( iris.data.shape )
print( iris.data[0] )

print( type( iris.target ) )
print( iris.target.shape )
print( iris.target[0] )


