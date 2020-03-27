import sys
import numpy as np
import scipy.stats 

m = len( sys.argv )
if m != 3 :
	print( "Args: m1 m2" )
	sys.exit(1)

n = 30
m1 = float( sys.argv[1] )
m2 = float( sys.argv[2] )

vx = 0.1*np.random.randn( n ) + m1
vy = 0.1*np.random.randn( n ) + m2

# print vx
# print vy

T, pvalue = scipy.stats.ttest_ind( vx, vy )
print( T, "p=", pvalue )
