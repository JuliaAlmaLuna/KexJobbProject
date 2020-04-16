import numpy as np
from scipy.interpolate import interp1d
from scipy.signal._savitzky_golay import savgol_filter
from Mathilda.randomshit.graphEcgAndDoppler import graph_ecg_and_doppler
import matplotlib.pyplot as plt


# Help function for import td txt file - finds where there are no longer 4 columns by throwing exception
def find_max_rows(fname):
    i = 1  # The smallest array size possible for TD data
    while 1:
        try:
            np.loadtxt(fname, skiprows=3, usecols=(2, 3), max_rows=i)
        except IndexError:
            break
        else:
            i = i + 1
    return i - 1


# Function to import TD+ECG data from txt file
def import_td_text_file(fname):
    max_rows = find_max_rows(fname)
    v = np.loadtxt(fname, skiprows=3, usecols=(0, 1), max_rows=max_rows)
    ecg1 = np.loadtxt(fname, skiprows=3, usecols=(2, 3), max_rows=max_rows)
    ecg2 = np.loadtxt(fname, skiprows=max_rows + 3)
    ecg = np.concatenate((ecg1, ecg2), axis=0)

    return ecg, v


# Same as above only taking ecg, v directly instead of fname
def create_interpolation_function_ecg_v(ecg, v, min_x, max_x, sample_number):
    vx = v[:, 0]
    vy = v[:, 1]
    f_v = interp1d(vx, vy)
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    f_ecg = interp1d(ecgx, ecgy)
    # np.lins
    x = np.linspace(min_x, max_x, sample_number)
    return f_ecg, f_v, x


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


def savgol_filter_ecg(ecg):
    filtered_ecg = savgol_filter(ecg, 33, 5)
    return filtered_ecg


def savgol_filter_v(v):
    filtered_v = savgol_filter(v, 33, 10)
    return filtered_v


# If ecg is upside-down it is fixed
def get_data(f_names, end_time):
    targets = []
    inputs = []

    for name in f_names:
        ecg, v = import_td_text_file(name)
        ecg, v = loop_data(ecg, v, end_time)
        f_ecg, f_v, x = create_interpolation_function_ecg_v(ecg, v, 0.035, end_time, 500)
        if name == "../ecg_folder/9_ecg.txt" or name == "../ecg_folder/19_ecg.txt" or name == "../ecg_folder/32_ecg.txt" or name == "../ecg_folder/62_ecg.txt" or name == "../ecg_folder/65_ecg.txt" or name == "../ecg_folder/69_ecg.txt" or name == "../ecg_folder/92_ecg.txt":
            targets.append(savgol_filter_ecg(-f_ecg(x)))
            inputs.append(savgol_filter_v(f_v(x)))
        else:
            targets.append(savgol_filter_ecg(f_ecg(x)))
            inputs.append(savgol_filter_v(f_v(x)))

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x


