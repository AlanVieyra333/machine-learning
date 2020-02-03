import numpy as np
import cv2

A = np.zeros( (3,2) ).astype( np.float32 )
i = 0
while i < 3:
	A[i,0] = 1
	A[i,1] = 2
	i += 1

print( A.shape )
print( A )

# hconcat( cv::Mat::ones( data.rows, 1, CV_32F ), data, data_t );

C = np.zeros((3,1)).astype( np.float32 )
print( C.shape  )
print( C )

E = [C, A]
print( E )

B = cv2.hconcat( [C, A] )
print( B.shape )
print( B )
