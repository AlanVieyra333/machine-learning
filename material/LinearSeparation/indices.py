import numpy as np

vx = np.zeros( (10,), dtype=np.int32 )
M = np.zeros( (10,2) )

# Una matriz de 10 renglones y dos columnas

print( vx.shape )

i = 5
while i<10 :
 	vx[i] = 2 
 	i += 1

print( vx )

vindices = vx != 2
print( vindices.shape )
print( vindices )

i = 0
while i<10 :
 	M[i][0] = i 
 	M[i][1] = i*i 
 	i += 1

# D = M[ vindices ].astype( np.float32 )
D = M[ vindices ]
print( D.shape )
print( D )

