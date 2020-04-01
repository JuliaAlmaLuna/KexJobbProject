from sklearn.neural_network import MLPRegressor
import dataManipulationFunctions as dmf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from scipy.stats import pearsonr

file_folder = "MLPregressor_sklearn/ECG_Folder/"
file_names = [file_folder + "2_ecg.txt", file_folder + "3_ecg.txt", file_folder + "6_ecg.txt", file_folder + "7_ecg.txt", file_folder + "4_ecg.txt", file_folder + "9_ecg.txt",
                       file_folder + "11_ecg.txt", file_folder + "5_ecg.txt", file_folder + "8_ecg.txt"]


def get_data(f_names):
    targets = []
    inputs = []

    for name in f_names:
        f_ecg, f_v, x = dmf.create_interpolation_function(name, 0.02, 2.31, 950)
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


targets, inputs, X = get_data(file_names)
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3, random_state=42)
mlp.fit(training_inputs, training_targets)
graph_predictions(mlp, testing_inputs=testing_inputs, testing_targets=testing_targets, x=X, rows=1, columns=3)


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
    print("____________________________________________________")
    print("R2 score:\t\t\t" + str(r2_score) + "\ntest MSE:\t\t\t" + str(test_mse) + "\ntrain MSE:\t\t\t" +
          str(train_mse) + "\nEVS, ua:\t\t\t" + str(ev_score) + "\nMax error:\t\t\t" + str(curr_max) +
          "\nPearson corr:\t\t\t" + str(pearson_corr))
    print("____________________________________________________")


evaluate_performance(mlp, testing_inputs, testing_targets, training_inputs, training_targets)










