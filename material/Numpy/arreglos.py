import numpy as np

int_list = list(range(10))
print( int_list )
print( type(int_list) )

listb = int_list * 2
print( listb )

int_arr = np.array( int_list )
print( int_arr )
print( type(int_arr) )

arrb = int_arr * 2
print( arrb )

# arrc = int_arr[::3]
# print( arrc )
