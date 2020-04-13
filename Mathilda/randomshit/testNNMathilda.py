import numpy as np
import matplotlib.pyplot as plt
import dataManipulationFunctions as dmf


file_folder = "ECG_Folder/"
training_file_names = [file_folder + "2_ecg.txt", file_folder + "3_ecg.txt", file_folder + "6_ecg.txt", file_folder + "7_ecg.txt", file_folder + "4_ecg.txt", file_folder + "9_ecg.txt",
                       file_folder + "11_ecg.txt"]
testing_file_names = [file_folder + "5_ecg.txt"]

nn_architecture = [
    {"input_dim": 5, "output_dim": 3, "activation": "tanh"},
    {"input_dim": 3, "output_dim": 3, "activation": "tanh"},
    {"input_dim": 3, "output_dim": 3, "activation": "tanh"},
    {"input_dim": 3, "output_dim": 3, "activation": "tanh"},
    {"input_dim": 3, "output_dim": 5, "activation": "tanh"},
]


def initialise_layers(nn_architecture):
    np.random.seed(99)
    params_values = {}

    for index, layer in enumerate(nn_architecture):
        layer_idx = index + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.rand(
            layer_output_size, layer_input_size)
        params_values['b' + str(layer_idx)] = np.random.rand(
            layer_output_size, 1)

    return params_values


def tanh(Z):
    return np.tanh(Z)


def tanh_backward(dA, Z):
    return dA * (1 - np.square(tanh(Z)))


def layer_forward_propagation(A_prev, W_curr, b_curr, activation):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation is "tanh":
        activation_func = tanh
    else:
        raise Exception('Non-supported activation function')
    return activation_func(Z_curr), Z_curr


def forward_propagation(input, params_values, nn_architecture):
    memory = {} # creating a temporary memory to store the information needed for a backward step
    A_curr = input

    for index, layer in enumerate(nn_architecture):
        layer_idx = index + 1
        A_prev = A_curr
        activ_function = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]

        A_curr, Z_curr = layer_forward_propagation(A_prev, W_curr, b_curr, activ_function)

        # saving calculated values in the memory
        memory["A" + str(index)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory


# Alternative cost function that uses euclidean distance instead.
def get_cost_value(predicted, true):
    number_of_values = predicted.shape[1]
    # calculation of the cost according to euclidean distance
    cost = (1/number_of_values) * np.linalg.norm(predicted - true)
    return np.squeeze(cost) # TODO: Should we squeeze this?


def get_cost_value_derivative(predicted, true):
    return (predicted-true)/np.linalg.norm(predicted-true)


def layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation):
    number_of_values = A_prev.shape[1]

    if activation is "tanh":
        backward_activation_func = tanh_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / number_of_values
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / number_of_values
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def backward_propagation(predicted, true, memory, params_values, nn_architecture):
    grads_values = {}
    number_of_values = true.shape[1]
    true = true.reshape(predicted.shape) # a hack ensuring the same shape of the prediction vector and labels vector
    dA_prev = get_cost_value_derivative(predicted, true)

    for layer_index_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_index_curr = layer_index_prev + 1
        activ_function = layer["activation"]
        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_index_prev)]
        Z_curr = memory["Z" + str(layer_index_curr)]

        W_curr = params_values["W" + str(layer_index_curr)]
        b_curr = params_values["b" + str(layer_index_curr)]

        dA_prev, dW_curr, db_curr = layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function)

        grads_values["dW" + str(layer_index_curr)] = dW_curr
        grads_values["db" + str(layer_index_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values


def train(inputs, targets, nn_architecture, epochs, learning_rate):
    params_values = initialise_layers(nn_architecture)
    cost_history = []
    training_samples = len(inputs)

    for i in range(epochs):
        predicion, cache = forward_propagation(inputs, params_values, nn_architecture)

        cost = get_cost_value(predicion, targets)
        cost_history.append(cost)

        grads_values = backward_propagation(predicion, targets, cache, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

    return params_values


def get_data(f_names):
    targets = []
    inputs = []
    for name in f_names:
        if 1:
            f_ecg, f_v, x = dmf.create_interpolation_function(name, 0.02, 2.31, 250)
            targets.append(f_ecg(x))
            inputs.append(f_v(x))
        if 0:
            f_ecg, f_v, x = dmf.other_not_interpolating_function(name, 0.02, 2.31, 250)
            targets.append(f_ecg(x))
            inputs.append(f_v(x))

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets.T, inputs.T, x


def get_prediction(testing_data_x, training_data_x, training_data_y, epochs, learning_rate):
    params_values = train(training_data_x, training_data_y, nn_architecture, epochs, learning_rate)
    prediction, cache = forward_propagation(testing_data_x, params_values, nn_architecture)
    return prediction


def plot_prediction(prediction, testing_data_y, time_axis):
    plt.plot(time_axis, prediction)
    plt.plot(time_axis, testing_data_y)
    plt.show()


training_data_targets, training_data_inputs, time_axis = get_data(training_file_names)
testing_data_targets, testing_data_inputs, time_axis2 = get_data(testing_file_names)
prediction = get_prediction(testing_data_inputs, training_data_inputs, training_data_targets, 7, 0.01)
plot_prediction(prediction, testing_data_targets, time_axis2)
