import numpy as np
import dataManipulationFunctions as dmf
from scipy.special import expit, logit

file_names = ["2_ecg.txt", "3_ecg.txt", "4_ecg.txt"]
input_size = 5
output_size = 1000
first_layer_size = 3
first_layer_weights = np.random.rand(2, first_layer_size, input_size)
second_layer_weights = np.random.rand(first_layer_size, first_layer_size, 2)


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


def get_first_layer_activation(t_data):
    f_layer = np.dot(first_layer_weights, t_data)
    f_layer_x = f_layer[0, :, 0]  # The matrix multiplication gets weird, should fix this
    f_layer_x = np.reshape(f_layer_x, (first_layer_size, 1))
    f_layer_y = f_layer[1, :, 1]
    f_layer_y = np.reshape(f_layer_y, (first_layer_size, 1))
    f_layer = np.concatenate((f_layer_x, f_layer_y), axis=1)
    f_layer = expit(f_layer)  # Applying sigmoid function
    return f_layer


def get_second_layer_activation():
    return


def get_output_activation():
    return


def calculate_loss_function():
    return


def backpropagate():
    return


training_data, expected_output = get_input_vector()
first_training_data = training_data[0, :, :]
first_layer = get_first_layer_activation(first_training_data)
print(first_layer)
