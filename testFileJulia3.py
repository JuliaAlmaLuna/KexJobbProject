import numpy as np
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from datetime import datetime
import dataManipulationFunctions as dmf

from scipy.interpolate import interp1d

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm



#The architecture of the layers of the Neural Network in this functon
#(Maybe here we could have 3 values as input dim or something
nn_architecture = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

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
def create_interpolation_function(fname):
    ecg, v = import_td_text_file(fname)
    vx = v[:, 0]
    vy = v[:, 1]
    f_v = interp1d(vx, vy)
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    f_ecg = interp1d(ecgx, ecgy)

    return f_ecg, f_v


#Grabs the data via the Data manipulation function and turns it into clear input and target arrays
def get_data(f_names):
    targets = []
    inputs = []
    i = 0
    for name in f_names:
        i = i + 1

        #Interpolation function
        f_ecg, f_v = create_interpolation_function(name)

        #Creating 2 input values for each target to match with the NN where 2 inputs go to 1 output
        x1 = np.linspace(0.02, 2.31, 250)
        x2 = np.linspace(0.02, 2.31, 500)

        temp_targets = np.asarray(f_ecg(x1))
        temp_targets.reshape(250,1)
        temp_inputs = np.asarray(f_v(x2))
        temp_inputs.reshape(250,2)
        targets.extend(temp_targets)
        inputs.extend(temp_inputs)


    targets = np.array(targets)
    inputs = np.array(inputs)
    targets.reshape(250 * i, 1)
    inputs.reshape(250 * i, 2)
    return targets, inputs, x1, x2


#Just initializes the layers and gives random values to the weights and biases. Might be useful to
def init_layers(nn_architecture, seed=88):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1

        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values