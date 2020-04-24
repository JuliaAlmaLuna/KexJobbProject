from scipy.interpolate import interp1d
import dataManipulationFunctions as dmf
import matplotlib.pyplot as plt
import numpy as np

targets = np.load("../mlpcolordoppler/targets_good_medium.npy")
x = np.load("../mlpcolordoppler/x.npy")
targets_old = np.array(targets)


plt.plot(x, targets[0])
plt.plot(x, targets_old[0])
plt.show()

plt.plot(x, targets[1])
plt.plot(x, targets_old[1])
plt.show()

plt.plot(x, targets[3])
plt.plot(x, targets_old[3])
plt.show()


