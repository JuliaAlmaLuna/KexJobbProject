from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error


def start(training_inputs_, training_targets_, testing_inputs_, testing_targets_):
    learning_rate = find_best_lr(training_inputs_, training_targets_, testing_inputs_, testing_targets_)
    alpha = find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, learning_rate)
    val_frac = find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, learning_rate, alpha)
    momentum, nev_momentum = find_best_momentum(training_inputs_, training_targets_, testing_inputs_, testing_targets_, learning_rate, alpha, val_frac)
    layer_size = find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_,
                                      learning_rate, alpha, val_frac, momentum, nev_momentum)
    message = "Here are your parameters: \nLearning rate: " + str(learning_rate) + "\nLayer size: " +\
              str(layer_size) + "\nalpha: " + str(alpha) + "\nval frac: " + str(val_frac) + "\nmomentum: " \
              + str(momentum) + " nev-mom: " + str(nev_momentum)
    return message


def find_best_lr(training_inputs_, training_targets_, testing_inputs_, testing_targets_):
    best_MSE = 10000
    best_lr = 'constant'
    lr = ['constant', 'invscaling', 'adaptive']
    for learning_rate in lr:
        mlp = MLPRegressor(
            hidden_layer_sizes=(100,), activation='tanh', solver='sgd', alpha=0.0001, batch_size='auto',
            learning_rate=learning_rate, learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.1)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_lr = learning_rate
    return learning_rate


def find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, lr):
    best_MSE = 10000
    best_alpha = 0.0001
    for alpha in np.arange(0.0001, 1, 0.001):
        mlp = MLPRegressor(
            hidden_layer_sizes=(100,), activation='tanh', solver='sgd', alpha=alpha, batch_size='auto',
            learning_rate=lr, learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=0.1)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_alpha = alpha
    return best_alpha


def find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, lr, alpha):
    best_MSE = 10000
    best_valfrac = 0.05
    for val_frac in np.arange(0.05, 0.55, 0.05):
        mlp = MLPRegressor(
            hidden_layer_sizes=(100,), activation='tanh', solver='sgd', alpha=alpha, batch_size='auto',
            learning_rate=lr, learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=True, validation_fraction=val_frac)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_valfrac = val_frac
    return best_valfrac


def find_best_momentum(training_inputs_, training_targets_, testing_inputs_, testing_targets_, lr, alpha, val_frac):
    best_MSE = 10000
    best_momentum = 0
    best_nev_momentum = 0
    nev_moms = [True, False]
    for momentum in np.arange(0.001, 1, 0.1):
        for nev_mom in nev_moms:
            mlp = MLPRegressor(
                hidden_layer_sizes=(100,), activation='tanh', solver='sgd', alpha=alpha, batch_size='auto',
                learning_rate=lr, learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
                random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=nev_mom,
                early_stopping=True, validation_fraction=val_frac)
            mlp.fit(training_inputs_, training_targets_)
            score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
            if score < best_MSE:
                best_MSE = score
                best_momentum = momentum
                best_nev_momentum = nev_mom
    return best_momentum, best_nev_momentum


def find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, lr, alpha, val_frac, mom, nev_mom):
    best_MSE = 10000
    best_layer_size = 1
    for layer_size in range(1, 100):
        mlp = MLPRegressor(
            hidden_layer_sizes=(layer_size,), activation='tanh', solver='sgd', alpha=alpha, batch_size='auto',
            learning_rate=lr, learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
            random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=mom, nesterovs_momentum=nev_mom,
            early_stopping=True, validation_fraction=val_frac)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_layer_size = layer_size
    return best_layer_size
