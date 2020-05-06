from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error


def start(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    mlp.fit(training_inputs_, training_targets_)

    activation = find_best_activation(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(activation=activation)

    learning_rate = find_best_learning_rate(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(learning_rate=learning_rate)

    learning_rate_init = find_best_learning_rate_init(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(learning_rate_init=learning_rate_init)

    power_t = find_best_power_t(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(power_t=power_t)

    alpha = find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(alpha=alpha)

    val_frac = find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(validation_fraction=val_frac)

    tolerance = find_best_tolerance(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(tol=tolerance)

    iterations = find_best_iterations(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(n_iter_no_change=iterations)

    layer_size = find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(hidden_layer_sizes=layer_size)

    epoch = find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(max_iter=epoch)

    momentum, nest_momentum = find_best_momentum(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
    mlp.set_params(momentum=momentum, nesterovs_momentum=nest_momentum)

    return mlp


def find_best_activation(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_activation = 'none'
    activations = ['tanh', 'logistic']
    for activation in activations:
        mlp.set_params(activation=activation)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_activation = activation
    print("Best_activation:{}".format(activation))
    return best_activation


def find_best_iterations(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_iter = 0
    for iter in np.arange(0.01, 10000, 10):
        mlp.set_params(n_iter_no_change=iter)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_iter = iter
    print("Best_n_iter_no_change:{}".format(best_iter))
    return best_iter


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


def find_best_learning_rate_init(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_lr = 0
    for lr in np.arange(0.0001, 10, 0.001):
        mlp.set_params(learning_rate_init=lr)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_lr = lr
    print("Best_learning_rate_init:{}".format(lr))
    return best_lr


def find_best_learning_rate(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_lr = 'none'
    learning_rates = ['constant', 'invscaling', 'adaptive']
    for learning_rate in learning_rates:
        mlp.set_params(learning_rate=learning_rate)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_lr = learning_rate
    print("Best_learning_rate:{}".format(learning_rate))
    return best_lr


def find_best_power_t(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_pt = 0
    for pt in np.arange(0, 2, 0.1):
        mlp.set_params(power_t=pt)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_pt = pt
    print("Best_pt:{}".format(pt))
    return best_pt


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


def find_best_tolerance(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_tol = 0
    for tol in np.arange(0.00001, 0.001, 0.00001):
        mlp.set_params(tol=tol)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_tol = tol
    print("Best_tolerance:{}".format(best_tol))
    return best_tol


def find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_epoch = 0.05
    for epoch_ in range(1, 500):
        mlp.set_params(max_iter=epoch_)
        mlp.fit(training_inputs_, training_targets_)
        score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
        if score < best_MSE:
            best_MSE = score
            best_epoch = epoch_
    print("Best_epoch number:{}".format(best_epoch))
    return best_epoch


def find_best_momentum(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_MSE = 10000
    best_momentum = 0
    best_nev_momentum = 0
    nev_moms = [True, False]
    for momentum in np.arange(0.001, 1, 0.1):
        for nev_mom in nev_moms:
            mlp.set_params(momentum=momentum, nesterovs_momentum=nev_mom)
            mlp.fit(training_inputs_, training_targets_)
            score = mean_squared_error(testing_targets_, mlp.predict(testing_inputs_))
            if score < best_MSE:
                best_MSE = score
                best_momentum = momentum
                best_nev_momentum = nev_mom
    print("Best_momentum:{}".format(best_momentum))
    print("Best_nesterovs_momentum:{}".format(best_nev_momentum))
    return best_momentum, best_nev_momentum


