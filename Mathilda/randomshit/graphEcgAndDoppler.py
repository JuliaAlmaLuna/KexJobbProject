import numpy as np
import dataManipulationFunctions as dmf
import matplotlib.pyplot as plt

# This file plots all ecg and doppler in graphs together (different scales) to check for inconsistencies in time

file_folder = "ecg_Folder/"
number_of_samples = 97


def init_file_names(number_of_samples, file_folder_, suffix):
    file_names_ = []

    for index in range(number_of_samples):
        string = file_folder_ + str(index+1) + suffix
        file_names_.append(string)
    return file_names_


file_names = init_file_names(number_of_samples, file_folder, "_ecg.txt")
print(file_names)


# If data is upside-down it is fixed
def get_data(f_names):
    targets = []
    inputs = []

    for name in f_names:
        ecg, v = dmf.import_td_text_file(name)
        start = v[0, 0]
        end = v[len(v)-1, 0]*0.95
        f_ecg, f_v, x = dmf.create_interpolation_function_ecg_v(ecg, v, start, end, 500)
        if name == "ecg_Folder/9_ecg.txt" or name == "ecg_Folder/19_ecg.txt" or name == "ecg_Folder/62_ecg.txt" or name == "ecg_Folder/65_ecg.txt" or name == "ecg_Folder/69_ecg.txt" or name == "ecg_Folder/92_ecg.txt":
            targets.append(-f_ecg(x))
            inputs.append(f_v(x))
        else:
            targets.append(f_ecg(x))
            inputs.append(f_v(x))
    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x


def graph_ecg_and_doppler(x, ecg, v, name):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(name)
    ax1.set_ylabel('velocity', color=color)
    ax1.plot(x, v, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('ecg', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, ecg, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


'''targets, inputs, x = get_data(file_names)
for index in range(100):
    input_ = inputs[index, :]
    target = targets[index, :]
    graph_ecg_and_doppler(x, target, input_, file_names[index])'''




