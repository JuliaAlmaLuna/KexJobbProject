import numpy as np
import dataManipulationFunctions as dmf
import matplotlib.pyplot as plt
from scipy.special import expit, logit

file_names = ["2_ecg.txt", "3_ecg.txt", "4_ecg.txt"]
input_size = 250
output_size = 16
first_layer_size = 16
second_layer_size = 16
first_layer_weights = np.random.rand(2, first_layer_size, input_size)
second_layer_weights = np.random.rand(2, first_layer_size, first_layer_size)
output_layer_weights = np.random.rand(2, second_layer_size, second_layer_size)


def get_input_vector():
    t_data = []
    e_output = []
    for file_name in file_names:
        ecg, v = dmf.import_td_text_file(file_name)
        v = dmf.trim_array(v, input_size)
        ecg = dmf.trim_array(ecg, output_size)
        e_output.append(ecg)
        t_data.append(v)
    t_data = np.array(t_data)
    e_output = np.array(e_output)
    return t_data, e_output


def get_layer_activation(layer_input, weight, layer_size):
    layer = np.dot(weight, layer_input)
    layer_x = layer[0, :, 0]
    # print(layer_x)
    layer_x = np.reshape(layer_x,(layer_size, 1))
    layer_y = layer[1, :, 1]
    layer_y = np.reshape(layer_y, (layer_size, 1))
    layer = np.concatenate((layer_x, layer_y), axis=1)
    # layer = expit(layer)  # Applying sigmoid function, maybe we don't want this?
    return layer


def get_output_activation():
    return


def visualise_output(o_layer):
    x = o_layer[:, 0]
    y = o_layer[:, 1]
    plt.plot(x,y)
    plt.show()


def calculate_loss_function(o_layer, e_output):
    l_function = o_layer - e_output
    l_function = np.square(l_function)
    return l_function


def backpropagate():
    return


training_data, expected_output = get_input_vector()
first_training_data = training_data[0, :, :]
first_expected_output = expected_output[0, :, :]

first_layer = get_layer_activation(first_training_data, first_layer_weights, first_layer_size)
second_layer = get_layer_activation(first_layer, second_layer_weights, second_layer_size)
output_layer = get_layer_activation(second_layer, output_layer_weights, output_size)
loss_function = calculate_loss_function(output_layer, first_expected_output)


