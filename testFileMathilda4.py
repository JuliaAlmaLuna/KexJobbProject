from scipy.interpolate import interp1d
import dataManipulationFunctions as dmf
import matplotlib.pyplot as plt
import numpy as np


file_names = ["2_ecg.txt", "3_ecg.txt", "4_ecg.txt", "5_ecg.txt"]
ecg, v = dmf.import_td_text_file(file_names[0])
vx = v[:, 0]
vy = v[:, 1]
f = interp1d(vx, vy)
x = np.linspace(0.02, 2.31904, 250)
plt.plot(vx, vy, 'o', x, f(x), '-')

plt.show()


