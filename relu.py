import matplotlib.pyplot as plt
import numpy as np


data = [7, 1, 10, 0.5, 11, 4, 18, 13]
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = np.multiply(x, 1.8)

plt.plot(x, data, '-', label='a)')
plt.plot(x, y, label='b)')
plt.plot(x, np.exp(x)*0.01 + 3, label='c)')
plt.scatter(x, data, c='k', label='data')
plt.title('Example of a Regression Problem')
plt.legend()
plt.show()