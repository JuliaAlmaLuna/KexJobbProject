import numpy as np
from scipy.interpolate import interp1d


# Help function for import td txt file - finds where there are no longer 4 columns by throwing exception
def find_max_rows(fname):
    i = 250  # The smallest array size possible for TD data
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


# create an interpolation function - range of function is from lowest value in file to highest value in file
# for 2,3,4 & 5 max range is 0.02 to 2.31
def create_interpolation_function(fname, min_x, max_x, sample_number):
    ecg, v = import_td_text_file(fname)
    vx = v[:, 0]
    vy = v[:, 1]
    f_v = interp1d(vx, vy)
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    f_ecg = interp1d(ecgx, ecgy)
   #np.lins
    x = np.linspace(min_x, max_x, sample_number)
    return f_ecg, f_v, x


# Help function to trim array, creates an array of indexes for uniform removal of values
def create_index_remove_list(remove_quantity, remove_step):
    rmv_index = [0]
    value = 0

    while len(rmv_index) < remove_quantity:
        value = value + remove_step
        rmv_index.append(int(value))
    return rmv_index


# Function to reduce the size of an array by uniformly removing values
def trim_array(array, goal_size):
    size = len(array)
    rmv_quantity = size - goal_size
    rmv_step = size / rmv_quantity
    rmv_index = create_index_remove_list(rmv_quantity, rmv_step)
    trimmed_array = np.delete(array, obj=rmv_index, axis=0)
    return trimmed_array
