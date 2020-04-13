from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from scipy.stats import pearsonr
from Mathilda.randomshit import genericEcg as ge


def graph_predictions(mlp, testing_inputs, testing_targets, x, rows, columns):
    size = len(testing_inputs)
    if rows*columns < size:
        return "Graph rows/columns too few"
    else:
        for index in range(size):
            prediction = mlp.predict(testing_inputs[index, :].reshape(1, -1))
            true = testing_targets[index, :]
            plt.subplot(rows, columns, index+1)
            plt.plot(x, true, 'g')
            plt.plot(x, prediction[0, :], 'r')
        plt.show()


def train_with_generic_ecg(mlp, _inputs, t):
    generic_ecg = ge.cut_generic_ecg_(t)
    generic_ecgs = []
    for index in range(len(_inputs)):
        generic_ecgs.append(generic_ecg)
    print(np.shape(_inputs))
    print(np.shape(generic_ecgs))
    mlp.fit(_inputs, generic_ecgs)


def evaluate_performance(mlp, testing_inputs, testing_targets, training_inputs, training_targets, message=""):
    train_prediction = mlp.predict(training_inputs)
    test_prediction = mlp.predict(testing_inputs)
    r2_score_ = mlp.score(testing_inputs, testing_targets)
    ev_score = explained_variance_score(testing_inputs, test_prediction, multioutput='uniform_average')
    train_mse = mean_squared_error(training_targets, train_prediction)
    test_mse = mean_squared_error(testing_targets, test_prediction)

    total = 0  # Finding the mean pearson score of testing data
    for index in range(len(testing_inputs)):
        pearson_corr, _ = pearsonr(testing_targets[index, :], test_prediction[index, :])
        total = total + pearson_corr
    pearson = total/len(testing_inputs)

    curr_max = 0  # Finding the maximum error in all the training data
    for index in range(len(testing_inputs)):
        max_error_ = max_error(testing_targets[index, :], test_prediction[index, :])
        if max_error_ > curr_max:
            curr_max = max_error_

    print(message)
    print("__________________________________________________________________________")
    print("R2 score:\t\t\t" + str(r2_score_) + "\t\t\tGoal: positive & close to 1" + "\ntest MSE:\t\t\t" + str(test_mse)
          + "\t\t\tGoal: lower than train MSE" + "\ntrain MSE:\t\t\t" + str(train_mse) + "\nEVS, ua:\t\t\t" +
          str(ev_score) + "\t\t\tGoal: Same as R2" + "\nMax error:\t\t\t" + str(curr_max) +
          "\t\t\tGoal: As low as possible" + "\nPearson:\t\t\t" + str(pearson) + "\t\t\tGoal: close to 1")
    print("__________________________________________________________________________")


inputs = np.load("Mathilda/mlpcolordoppler/inputs.npy")
targets = np.load("Mathilda/mlpcolordoppler/targets.npy")
X = np.load("Mathilda/mlpcolordoppler/x.npy")
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3, random_state=66)


def find_best_beta(training_inputs_, training_targets_, testing_inputs_, testing_targets_):
    best_MSE = 10000
    best_beta1 = 0
    best_beta2 = 0
    for beta1 in np.arange(0, 1, 0.1):
        for beta2 in np.arange(0, 1, 0.1):
            mlp = MLPRegressor(
                hidden_layer_sizes=(96,), activation='tanh', solver='adam', alpha=10, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                early_stopping=True, validation_fraction=0.2, beta_1=beta1, beta_2=beta2, epsilon=1e-08)
            mlp.fit(training_inputs_, training_targets_)
            score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
            if score < best_MSE:
                best_MSE = score
                best_beta1 = beta1
                best_beta2 = beta2
    print(best_MSE)
    return best_beta1, best_beta2


def find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_):
    best_MSE = 10000
    best_layer_size = 1
    for layer_size in range(1, 100):
        mlp = MLPRegressor(
            hidden_layer_sizes=(layer_size,), activation='tanh', solver='adam', alpha=10, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_layer_size = layer_size
    print(best_MSE)
    return best_layer_size,


def find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_):
    best_MSE = 10000
    best_alpha = 0.0001
    for alpha in np.arange(0.0001, 1, 0.001):
        mlp = MLPRegressor(
            hidden_layer_sizes=(96,), activation='tanh', solver='adam', alpha=alpha, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.2, beta_1=0.3, beta_2=0.5, epsilon=1e-08)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_alpha = alpha
    return best_alpha


def find_best_epsilon(training_inputs_, training_targets_, testing_inputs_, testing_targets_):
    best_MSE = 10000
    best_epsilon = 0.0001
    for epsilon in np.arange(1e-09, 1e-07, 1e-09):
        mlp = MLPRegressor(
            hidden_layer_sizes=(96,), activation='tanh', solver='adam', alpha=0.4211, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.2, beta_1=0.3, beta_2=0.5, epsilon=epsilon)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_epsilon = epsilon
    return best_epsilon


mlp = MLPRegressor(
    hidden_layer_sizes=(96,), activation='tanh', solver='adam', alpha=0.4211, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=45, tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.1, beta_1=0.3, beta_2=0.5, epsilon=1e-08)
mlp.fit(training_inputs, training_targets)
graph_predictions(mlp, testing_inputs=testing_inputs, testing_targets=testing_targets, x=X, rows=5, columns=7)
evaluate_performance(mlp, testing_inputs, testing_targets, training_inputs, training_targets)








