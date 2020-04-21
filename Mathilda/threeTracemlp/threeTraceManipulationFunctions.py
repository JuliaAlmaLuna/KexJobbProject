import numpy as np
from scipy.interpolate import interp1d


# Help function for import td txt file - finds where there are no longer 4 columns by throwing exception
def find_max_rows_3Trace(fname):
    i = 1  # The smallest array size possible for TD data
    while 1:
        try:
            np.loadtxt(fname, skiprows=3, usecols=(4, 5), max_rows=i)
        except IndexError:
            break
        else:
            i = i + 1
    return i - 1


# Function to import TD+ECG data from txt file
def import_3trace_text_file(fname):
    max_rows = find_max_rows_3Trace(fname)
    time = np.loadtxt(fname, skiprows=3, usecols=0, max_rows=max_rows)
    # trace1 = np.loadtxt(fname, skiprows=3, usecols=1, max_rows=max_rows)  Septal wall trace
    trace2 = np.loadtxt(fname, skiprows=3, usecols=2, max_rows=max_rows)
    trace3 = np.loadtxt(fname, skiprows=3, usecols=3, max_rows=max_rows)
    ecg1 = np.loadtxt(fname, skiprows=3, usecols=(4, 5), max_rows=max_rows)
    ecg2 = np.loadtxt(fname, skiprows=max_rows + 3)
    ecg = np.concatenate((ecg1, ecg2), axis=0)

    return ecg, trace2, trace3, time


# create an interpolation function - range of function is from lowest value in file to highest value in file
def create_interpolation_function_3trace(fname, sample_number):
    ecg, trace2, trace3, time = import_3trace_text_file(fname)
    f_trace2 = interp1d(time, trace2)
    f_trace3 = interp1d(time, trace3)
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    f_ecg = interp1d(ecgx, ecgy)
    min_x = time[0]
    max_x = time[len(time)-1]
    x = np.linspace(min_x, max_x, sample_number)
    return f_ecg, f_trace2, f_trace3, x


def get_data(f_names, sample_number=0):
    targets = []
    inputs = []

    for name in f_names:
        if sample_number == 0:
            f_ecg, f_trace2, f_trace3, x = create_interpolation_function_3trace(name, 950)
        else:
            f_ecg, f_trace2, f_trace3, x = create_interpolation_function_3trace(name, sample_number)
        input_ = (np.array(f_trace2(x)) + np.array(f_trace3(x))) / 2.0
        targets.append(f_ecg(x))
        inputs.append(input_)

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x


def init_file_names(number_of_samples, file_folder_, suffix):
    file_names_ = []

    for index in range(number_of_samples):
        string = file_folder_ + str(index+1) + suffix
        file_names_.append(string)
    return file_names_

