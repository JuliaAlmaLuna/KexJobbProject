from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from scipy.stats import pearsonr
from Mathilda.randomshit import genericEcg as ge
from Mathilda.mlpcolordoppler import solverAdamOptimisation
from Mathilda.randomshit.graphEcgAndDoppler import graph_ecg_and_doppler


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
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3)

# print(sao.start(training_inputs, training_targets, testing_inputs, testing_targets))
mlp_sgd = MLPRegressor(
    hidden_layer_sizes=(92,), activation='tanh', solver='sgd', alpha=0.4751, batch_size='auto',
    learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.1, beta_1=0.2, beta_2=0.7, epsilon=1e-08, n_iter_no_change=10)

mlp_adam = MLPRegressor(
    hidden_layer_sizes=(60,), activation='tanh', solver='adam', alpha=10, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=100, shuffle=True,
    tol=0.0001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.9, epsilon=1e-08, n_iter_no_change=10)

print(solverAdamOptimisation.start(training_inputs, training_targets, testing_inputs, testing_targets))

'''mlp_adam.fit(training_inputs, training_targets)
predictions = mlp_adam.predict(testing_inputs)

for index in range(len(predictions)):
    graph_ecg_and_doppler(X, predictions[index], testing_inputs[index], str(index))

graph_predictions(mlp_adam, testing_inputs=testing_inputs, testing_targets=testing_targets, x=X, rows=5, columns=7)
evaluate_performance(mlp_adam, testing_inputs, testing_targets, training_inputs, training_targets)'''








