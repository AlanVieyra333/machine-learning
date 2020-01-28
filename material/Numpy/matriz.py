import numpy as np
#
# a  es una variable
# va es un vector
# A  es una matriz
#
# una función es un verbo rojo(), pone_en_rojo( )
# Los objetos si pueden ser nombres

# No se les quita el nombre a los módulos
# import numero
# numero.pone_color( )
# numero.PoneColor( )


A = np.zeros( (3,3) )
print( A )

B = np.eye( 3 )
print( B )

C = 4.0*np.eye( 3 )

C.dot( B )
print( C )

print( C.ndim )
print( C.shape )
print( C.size )
print( C.dtype )
