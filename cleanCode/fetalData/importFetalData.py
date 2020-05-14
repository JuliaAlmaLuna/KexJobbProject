import numpy as np
import matplotlib.pyplot as plt

t = np.loadtxt('1007_t.txt')
v = np.loadtxt('1007_v.txt')

plt.plot(t, v)
plt.show()