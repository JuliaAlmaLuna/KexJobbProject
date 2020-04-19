from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error


def start(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    mlp.fit(training_inputs_, training_targets_)
    beta1, beta2 = find_best_beta(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(beta_1=beta1, beta_2=beta2)
    alpha = find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(alpha=alpha)
    val_frac = find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(validation_fraction=val_frac)
    layer_size = find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(hidden_layer_sizes=layer_size)
    epoch = find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(n_iter_no_change=epoch)
    message = "Here are your parameters: \nBeta1: " + str(beta1) + " Beta2: " + str(beta2) + "\nLayer size: " +\
              str(layer_size) + "\nalpha: " + str(alpha) + "\nval frac: " + str(val_frac) + "\nn_iter_no_change: " + str(epoch)
    return message, mlp


def find_best_beta(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_beta1 = 0
    best_beta2 = 0
    for beta1_ in np.arange(0.1, 1, 0.1):
        for beta2_ in np.arange(0.1, 1, 0.1):
            mlp.set_params(beta_1=beta1_, beta_2=beta2_)
            mlp.fit(training_inputs_, training_targets_)
            score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
            if score < best_MSE:
                best_MSE = score
                best_beta1 = beta1_
                best_beta2 = beta2_
    print("Best_beta1:{}".format(best_beta1) + "And best best_beta2{}".format(best_beta2))
    return best_beta1, best_beta2


def find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_alpha = 0.0001
    for alpha_ in np.arange(0.0001, 1, 0.001):
        mlp.set_params(alpha=alpha_)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_alpha = alpha_
    print("Best_alpha:{}".format(best_alpha))
    return best_alpha


def find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_valfrac = 0.05
    for val_frac in np.arange(0.05, 0.55, 0.05):
        mlp.set_params(validation_fraction=val_frac)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_valfrac = val_frac
    print("Best_valfrac:{}".format(best_valfrac))
    return best_valfrac


def find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_epoch = 0.05
    for epoch_ in range(1, 200):
        mlp.set_params(n_iter_no_change=epoch_)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_epoch = epoch_
    print("Best_epoch number:{}".format(best_epoch))
    return best_epoch


def find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_layer_size = 1
    for layer_size in range(1, 150):
        mlp.set_params(hidden_layer_sizes=layer_size)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_layer_size = layer_size
    print("Best_layer size:{}".format(best_layer_size))
    return best_layer_size
