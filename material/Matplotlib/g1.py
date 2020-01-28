import matplotlib.pyplot as plt
import numpy as np

vx = np.linspace(0, 10, 100)

plt.plot( vx, np.sin(vx) )
plt.xlabel("x (u.a.)")
plt.ylabel("sen(x)")
plt.grid()
plt.show()

