import numpy as np
import dataManipulationFunctions as dmf
import matplotlib.pyplot as plt

# This file plots all ecg and doppler in graphs together (different scales) to check for inconsistencies in time

file_folder = "MLPregressor_sklearn/ECG_Folder/"
file_names = [file_folder + "2_ecg.txt", file_folder + "3_ecg.txt", file_folder + "6_ecg.txt", file_folder +
              "7_ecg.txt", file_folder + "4_ecg.txt", file_folder + "9_ecg.txt", file_folder + "11_ecg.txt",
              file_folder + "5_ecg.txt", file_folder + "8_ecg.txt"]


def get_data(f_names):
    targets = []
    inputs = []

    for name in f_names:
        f_ecg, f_v, x = dmf.create_interpolation_function(name, 0.02, 2.31, 950)
        targets.append(f_ecg(x))
        inputs.append(f_v(x))

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x


targets, inputs, x = get_data(file_names)
for index in range(9):
    input_ = inputs[index, :]
    target = targets[index, :]
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(x, input_, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, target, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
