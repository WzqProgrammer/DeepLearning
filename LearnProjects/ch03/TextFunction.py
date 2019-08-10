import Common.math_functions as mf
import numpy as np
import matplotlib.pylab as plt

x = np.arange(-5.0, 5.0, 0.1)
#y = pr.step_function(x)
y = mf.sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)   #指定y轴的范围
plt.show()