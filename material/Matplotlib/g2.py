import matplotlib
matplotlib.use('PDF') 

import matplotlib.pyplot as plt
import numpy as np

vx = np.linspace(0, 10, 100)

plt.ioff()
plt.plot( vx, np.sin(vx) )

# plt.show()
# export MPLBACKEND=PS
# unset MPLBACKEND
plt.savefig("g22.pdf")

# Y en la terminal:
# ps2pdf -sEPSCrop g2.eps g2.pdf
