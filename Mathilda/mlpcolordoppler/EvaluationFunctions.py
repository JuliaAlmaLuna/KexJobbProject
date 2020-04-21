import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from scipy.stats import pearsonr
from Mathilda.mlpcolordoppler.dopplerManipulationFunctions import savgol_filter_ecg


def graph_predictions(mlp, testing_inputs, testing_targets, x, rows, columns):
    size = len(testing_inputs)
    if rows*columns < size:
        return "Graph rows/columns too few"
    else:
        for index in range(size):
            prediction = mlp.predict(testing_inputs[index, :].reshape(1, -1))
            true = testing_targets[index, :]
            ax1 = plt.subplot(rows, columns, index+1)
            ax1.plot(x, testing_inputs[index], 'y')

            ax2 = ax1.twinx()
            ax2.plot(x, true, 'g')
            ax2.plot(x, prediction[0, :], 'r')

        plt.show()


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

    line = "__________________________________________________________________________"
    evaluations = message + "\n" + line + "\nR2 score:\t\t\t" + str(r2_score_) + "\t\t\tGoal: positive & close to 1" + "\ntest MSE:\t\t\t" + \
                  str(test_mse) + "\t\t\tGoal: lower than train MSE" + "\ntrain MSE:\t\t\t" + str(train_mse) + \
                  "\nEVS, ua:\t\t\t" + str(ev_score) + "\t\t\tGoal: Same as R2" + "\nMax error:\t\t\t" + \
                  str(curr_max) + "\t\t\tGoal: As low as possible" + "\nPearson:\t\t\t" + str(pearson) + \
                  "\t\t\tGoal: close to 1\n" + line
    return evaluations