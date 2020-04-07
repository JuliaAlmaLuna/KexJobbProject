from sklearn.neural_network import MLPRegressor
import dataManipulationFunctions as dmf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from scipy.stats import pearsonr
from Mathilda import genericEcg as ge

number_of_sample_data = 100
file_folder = "Mathilda/ecg_folder/"
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


def get_data(f_names, end_time):
    targets = []
    inputs = []

    for name in f_names:
        ecg, v = dmf.import_td_text_file(name)
        ecg, v = loop_data(ecg, v, end_time)
        f_ecg, f_v, x = dmf.create_interpolation_function_ecg_v(ecg, v, 0.035, end_time, 500)
        targets.append(f_ecg(x))
        inputs.append(f_v(x))

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x


def graph_predictions(mlp, testing_inputs, testing_targets, x, rows, columns):
    size = len(testing_inputs)
    if rows*columns < size:
        return "Graph rows/columns too few"
    else:
        for index in range(size):
            prediction = mlp.predict(testing_inputs[index, :].reshape(1, -1))
            true = testing_targets[index, :]
            plt.subplot(rows, columns, index+1)
            plt.plot(x, prediction[0, :], 'r')
            plt.plot(x, true, 'g')
        plt.show()


mlp = MLPRegressor(
    hidden_layer_sizes=(100,),  activation='tanh', solver='lbfgs', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=100000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


def train_with_generic_ecg(mlp, _inputs, t):
    generic_ecg = ge.cut_generic_ecg_(t)
    generic_ecgs = []
    for index in range(len(_inputs)):
        generic_ecgs.append(generic_ecg)
    print(np.shape(_inputs))
    print(np.shape(generic_ecgs))
    mlp.fit(_inputs, generic_ecgs)


def evaluate_performance(mlp, testing_inputs, testing_targets, training_inputs, training_targets, message=""):
    r2_score = mlp.score(testing_inputs, testing_targets)
    train_prediction = mlp.predict(training_inputs)
    test_prediction = mlp.predict(testing_inputs)
    ev_score = explained_variance_score(testing_inputs, test_prediction, multioutput='uniform_average')
    train_mse = mean_squared_error(training_targets, train_prediction)
    test_mse = mean_squared_error(testing_targets, test_prediction)
    pearson_corr, _ = pearsonr(testing_targets[0, :], test_prediction[0, :])

    curr_max = 0  # Finding the maximum error in all the training data
    for index in range(len(testing_inputs)):
        max_error_ = max_error(testing_targets[index, :], test_prediction[index, :])
        if max_error_ > curr_max:
            curr_max = max_error_

    print(message)
    print("__________________________________________________________________________")
    print("R2 score:\t\t\t" + str(r2_score) + "\t\t\tGoal: positive & close to 1" + "\ntest MSE:\t\t\t" + str(test_mse)
          + "\t\t\tGoal: lower than train MSE" + "\ntrain MSE:\t\t\t" + str(train_mse) + "\nEVS, ua:\t\t\t" +
          str(ev_score) + "\t\t\tGoal: Same as R2" + "\nMax error:\t\t\t" + str(curr_max) +
          "\t\t\tGoal: As low as possible" + "\nPearson:\t\t\t" + str(pearson_corr) + "\t\t\tGoal: close to 1")
    print("__________________________________________________________________________")


file_names = init_file_names(number_of_sample_data, file_folder, suffix)
targets, inputs, X = get_data(file_names, 3)
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3, random_state=42)
mlp.fit(training_inputs, training_targets)
graph_predictions(mlp, testing_inputs=testing_inputs, testing_targets=testing_targets, x=X, rows=5, columns=7)
evaluate_performance(mlp, testing_inputs, testing_targets, training_inputs, training_targets)



