import numpy as np
import matplotlib.pyplot as plt
import cv2

# generamos unos datos
media = [20, 20]
cov = [ [ 5, 10],
        [10, 25] ] 

x, y = np.random.multivariate_normal( media, cov, 1000 ).T

# Hacemos la gráfica de los datos
# plt.style.use( 'ggplot' )
plt.plot( x, y, 'o', zorder=1 )
plt.xlabel( "Característica 1" )
plt.ylabel( "Característica 2" )
plt.axis( 'equal' )

# plt.show( )

X = np.vstack( (x,y) ).T

mu, eig = cv2.PCACompute( X, np.array( [] ) )
print( mu )
print( eig )

plt.quiver( media[0], media[1], eig[:,0], eig[:,1], zorder=2, scale=3 )
plt.show( ) 
