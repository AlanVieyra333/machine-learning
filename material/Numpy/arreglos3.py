import numpy as np

int_list = list(range(10))

int_arr = np.array( int_list )
print( int_arr )
print( type(int_arr) )

arrb = int_arr * 2
print( arrb )

print( arrb.ndim )
print( arrb.shape )
print( arrb.size )
print( arrb.dtype )
