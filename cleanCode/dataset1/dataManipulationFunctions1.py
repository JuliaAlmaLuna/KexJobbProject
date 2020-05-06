import numpy as np
from scipy.interpolate import interp1d
from scipy.signal._savitzky_golay import savgol_filter


# Help function for import txt file - locates index where there are no longer 4 columns by throwing exception
def find_max_rows(fname):
    i = 10  # The smallest array size possible for data
    while 1:
        try:
            np.loadtxt(fname, skiprows=3, usecols=(2, 3), max_rows=i)
        except IndexError:
            break
        else:
            i = i + 1
    return i - 1 # returns the index where velocity data ends


# Function to import velocity + ECG data from txt file - uses find max rows to know end of velocity data
def import_td_text_file(fname):
    max_rows = find_max_rows(fname)
    v = np.loadtxt(fname, skiprows=3, usecols=(0, 1), max_rows=max_rows)
    ecg1 = np.loadtxt(fname, skiprows=3, usecols=(2, 3), max_rows=max_rows)
    ecg2 = np.loadtxt(fname, skiprows=max_rows + 3)
    ecg = np.concatenate((ecg1, ecg2), axis=0)

    return ecg, v


# Inputs ecg and velocity data and returns functions of them and an x-axis
def create_interpolation_function_ecg_v(ecg, v, min_x, max_x, sample_number):
    vx = v[:, 0]
    vy = v[:, 1]
    f_v = interp1d(vx, vy)
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    f_ecg = interp1d(ecgx, ecgy)
    x = np.linspace(min_x, max_x, sample_number)
    return f_ecg, f_v, x


# Inputs an index array to pick specified files - returns a list of file names to open and import
def init_file_names(file_folder_, suffix, index_array):
    file_names_ = []

    for index in index_array:
        string = file_folder_ + str(index) + suffix
        file_names_.append(string)
    return file_names_


# Function to extend data to normalised length
def loop_data(ecg, v, end_time):
    curr_end_time = v[len(v) - 1, 0]
    while curr_end_time < end_time:
        v = double_data(v, curr_end_time)
        ecg = double_data(ecg, curr_end_time)
        curr_end_time = v[len(v) - 1, 0]
    return ecg, v


# Helpfunction for loop_data, used to double data until desired length is found
def double_data(array, curr_end_time):
    array_x = array[:, 0]
    array_y = array[:, 1]
    array2_x = array_x + curr_end_time
    array2 = np.stack((array2_x, array_y), axis=1)

    doubled_array = np.concatenate((array, array2))
    return doubled_array


# Smoothing filter with parameters tuned for ECG data
def savgol_filter_ecg(ecg):
    filtered_ecg = savgol_filter(ecg, 33, 5)
    return filtered_ecg


# Smoothing filter with parameters tuned for velocity data
def savgol_filter_v(v):
    filtered_v = savgol_filter(v, 33, 10)
    return filtered_v


# Function to import data from .txt files to usable input and target numpy arrays
def get_data(f_names, end_time):
    targets = []
    inputs = []

    for name in f_names:
        ecg, v = import_td_text_file(name)
        ecg, v = loop_data(ecg, v, end_time)
        f_ecg, f_v, x = create_interpolation_function_ecg_v(ecg, v, 0.035, end_time, 500)
        # If ecg is upside-down it is fixed
        if name == "../data/9_ecg.txt" or name == "../data/19_ecg.txt" or name == "../data/32_ecg.txt" or name == "../data/62_ecg.txt" or name == "../data/65_ecg.txt" or name == "../data/69_ecg.txt" or name == "../data/92_ecg.txt":
            targets.append(savgol_filter_ecg(-f_ecg(x)))
            inputs.append(savgol_filter_v(f_v(x)))
        else:
            targets.append(savgol_filter_ecg(f_ecg(x)))
            inputs.append(savgol_filter_v(f_v(x)))

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x