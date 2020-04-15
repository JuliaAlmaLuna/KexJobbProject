from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error


def start(training_inputs_, training_targets_, testing_inputs_, testing_targets_):
    beta1, beta2 = find_best_beta(training_inputs_, training_targets_, testing_inputs_, testing_targets_)
    alpha = find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, beta1, beta2)
    val_frac = find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, beta1, beta2, alpha)
    epoch = find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, beta1, beta2, alpha, val_frac)
    layer_size = find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, beta1,
                                      beta2, alpha, val_frac, epoch)
    message = "Here are your parameters: \nBeta1: " + str(beta1) + " Beta2: " + str(beta2) + "\nLayer size: " +\
              str(layer_size) + "\nalpha: " + str(alpha) + "val frac: " + str(val_frac) + "epoch: " + str(epoch)
    return message


def find_best_beta(training_inputs_, training_targets_, testing_inputs_, testing_targets_):
    best_MSE = 10000
    best_beta1 = 0
    best_beta2 = 0
    for beta1 in np.arange(0, 1, 0.1):
        for beta2 in np.arange(0, 1, 0.1):
            mlp = MLPRegressor(
                hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=0.0001, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                early_stopping=True, validation_fraction=0.1, beta_1=beta1, beta_2=beta2, epsilon=1e-08)
            mlp.fit(training_inputs_, training_targets_)
            score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
            if score < best_MSE:
                best_MSE = score
                best_beta1 = beta1
                best_beta2 = beta2
    print(best_MSE)
    return best_beta1, best_beta2


def find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, beta1, beta2):
    best_MSE = 10000
    best_alpha = 0.0001
    for alpha in np.arange(0.0001, 1, 0.001):
        mlp = MLPRegressor(
            hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=alpha, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.1, beta_1=beta1, beta_2=beta2, epsilon=1e-08)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_alpha = alpha
    return best_alpha


def find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, beta1, beta2, alpha):
    best_MSE = 10000
    best_valfrac = 0.05
    for val_frac in np.arange(0.05, 0.55, 0.05):
        mlp = MLPRegressor(
            hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=alpha, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=val_frac, beta_1=beta1, beta_2=beta2, epsilon=1e-08)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_valfrac = val_frac
    return best_valfrac


def find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, beta1, beta2, alpha, val_frac):
    best_MSE = 10000
    best_epoch = 0.05
    for epoch in range(1, 100):
        mlp = MLPRegressor(
            hidden_layer_sizes=(100,), activation='tanh', solver='adam', alpha=alpha, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=val_frac, beta_1=beta1, beta_2=beta2, epsilon=1e-08, n_iter_no_change=epoch)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_epoch = epoch
    return best_epoch


def find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, beta1, beta2, alpha, val_frac, epoch):
    best_MSE = 10000
    best_layer_size = 1
    for layer_size in range(1, 100):
        mlp = MLPRegressor(
            hidden_layer_sizes=(layer_size,), activation='tanh', solver='adam', alpha=alpha, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=val_frac, beta_1=beta1, beta_2=beta2, epsilon=1e-08, n_iter_no_change=epoch)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_layer_size = layer_size
    print(best_MSE)
    return best_layer_size
