import numpy as np

int_arr = np.array( range(10) )
print( int_arr )
print( type(int_arr) )

arrb = int_arr * 2
print( arrb )

print( arrb[-1], arrb[-2] )


print( arrb[1:3] )
print( arrb[::3] )

