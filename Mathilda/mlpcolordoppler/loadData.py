import numpy as np
import dataManipulationFunctions as dmf

number_of_sample_data = 97
file_folder = "../ecg_folder/"
suffix = "_ecg.txt"


def init_file_names(number_of_samples, file_folder_, suffix):
    file_names_ = []

    for index in range(number_of_samples):
        string = file_folder_ + str(index+1) + suffix
        file_names_.append(string)
    return file_names_


def double_data(array, curr_end_time):
    array_x = array[:, 0]
    array_y = array[:, 1]
    array2_x = array_x + curr_end_time
    array2 = np.stack((array2_x, array_y), axis=1)

    doubled_array = np.concatenate((array, array2))
    return doubled_array


def loop_data(ecg, v, end_time):
    curr_end_time = v[len(v) - 1, 0]
    while curr_end_time < end_time:
        v = double_data(v, curr_end_time)
        ecg = double_data(ecg, curr_end_time)
        curr_end_time = v[len(v) - 1, 0]
    return ecg, v


# If ecg is upside-down it is fixed
def get_data(f_names, end_time):
    targets = []
    inputs = []

    for name in f_names:
        ecg, v = dmf.import_td_text_file(name)
        ecg, v = loop_data(ecg, v, end_time)
        f_ecg, f_v, x = dmf.create_interpolation_function_ecg_v(ecg, v, 0.035, end_time, 500)
        if name == "ecg_Folder/9_ecg.txt" or name == "ecg_Folder/19_ecg.txt" or name == "ecg_Folder/62_ecg.txt" or name == "ecg_Folder/65_ecg.txt" or name == "ecg_Folder/69_ecg.txt" or name == "ecg_Folder/92_ecg.txt":
            targets.append(-f_ecg(x))
            inputs.append(f_v(x))
        else:
            targets.append(f_ecg(x))
            inputs.append(f_v(x))

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x


filenames = init_file_names(number_of_sample_data, file_folder, suffix)
targets, inputs, x = get_data(filenames, 3)
np.save("inputs", inputs)
np.save("targets", targets)
np.save("x", x)