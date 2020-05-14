import numpy as np
from scipy.stats import pearsonr
import math


def start(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp, learning_rate=True,
          learning_rate_init=True, power_t=True, alpha=True, val_frac=True, tolerance=True, n_iter_no_change=True,
          hidden_layer_sizes=True, max_iter=True, momentum=True):
    mlp.fit(training_inputs_, training_targets_)

    if learning_rate:
        learning_rate = find_best_learning_rate(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(learning_rate=learning_rate)

    if learning_rate_init:
        learning_rate_init = find_best_learning_rate_init(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(learning_rate_init=learning_rate_init)

    if power_t:
        power_t = find_best_power_t(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(power_t=power_t)

    if alpha:
        alpha = find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(alpha=alpha)

    if val_frac:
        val_frac = find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(validation_fraction=val_frac)

    if tolerance:
        tolerance = find_best_tolerance(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(tol=tolerance)

    if n_iter_no_change:
        iterations = find_best_iterations(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(n_iter_no_change=iterations)

    if hidden_layer_sizes:
        layer_size = find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(hidden_layer_sizes=layer_size)

    if max_iter:
        epoch = find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(max_iter=epoch)

    if momentum:
        momentum, nest_momentum = find_best_momentum(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(momentum=momentum, nesterovs_momentum=nest_momentum)

    return mlp


def pearson_score(testing_inputs, testing_targets, mlp, best_score):
    test_prediction = mlp.predict(testing_inputs)
    total = 0  # Finding the mean pearson score of testing data
    for index in range(len(testing_inputs)):
        try:
            pearson_corr, _ = pearsonr(testing_targets[index, :], test_prediction[index, :])
        except ValueError:
            print('Value error')
            total = 0
            break
        else:
            total = total + pearson_corr

    pearson = total / len(testing_inputs)
    return pearson


def find_best_iterations(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_iter = 0
    best_pearson = -10000
    for iter in np.arange(10, 60, 10):
        mlp.set_params(n_iter_no_change=iter)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson > best_pearson:
            best_pearson = pearson
            best_iter = iter

    print("Best_n_iter_no_change:{}".format(best_iter))
    return best_iter


def find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_pearson = -10000
    best_valfrac = 0.05
    for val_frac in np.arange(0.05, 0.55, 0.05):
        mlp.set_params(validation_fraction=val_frac)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson > best_pearson:
            best_pearson = pearson
            best_valfrac = val_frac

    print("Best_valfrac:{}".format(best_valfrac))
    return best_valfrac


def find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_pearson = -10000
    best_alpha = 0.0002
    for alpha_ in np.arange(0.00002, 0.0002, 0.00002):
        mlp.set_params(alpha=alpha_)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson > best_pearson:
            best_pearson = pearson
            best_alpha = alpha_

    print("Best_alpha:{}".format(best_alpha))
    return best_alpha


def find_best_learning_rate_init(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_pearson = -10000
    best_lr = 0
    for lr in np.arange(0.01, 0.2, 0.01):
        mlp.set_params(learning_rate_init=lr)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson > best_pearson:
            best_pearson = pearson
            best_lr = lr

    print("Best_learning_rate_init:{}".format(lr))
    return best_lr


def find_best_learning_rate(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_pearson = -10000
    best_lr = 'none'
    learning_rates = ['constant', 'invscaling', 'adaptive']
    for learning_rate in learning_rates:
        mlp.set_params(learning_rate=learning_rate)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson > best_pearson:
            best_pearson = pearson
            best_lr = learning_rate

    print("Best_learning_rate:{}".format(learning_rate))
    return best_lr


def find_best_power_t(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_pearson = -10000
    best_pt = 0
    for pt in np.arange(0, 2, 0.1):
        mlp.set_params(power_t=pt)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson < best_pearson:
            best_pearson = pearson
            best_pt = pt

    print("Best_pt:{}".format(pt))
    return best_pt


def find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_layer_size = 1
    best_pearson = -10000
    for layer_size in np.arange(1, 300, 20):
        mlp.set_params(hidden_layer_sizes=layer_size)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson > best_pearson:
            best_pearson = pearson
            best_layer_size = layer_size

    print("Best_layer size:{}".format(best_layer_size))
    return best_layer_size


def find_best_tolerance(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_tol = 0
    best_pearson = -10000
    for tol in np.arange(0.0002, 0.001, 0.0002):
        mlp.set_params(tol=tol)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson > best_pearson:
            best_pearson = pearson
            best_tol = tol

    print("Best_tolerance:{}".format(best_tol))
    return best_tol


def find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_epoch = 0.05
    best_pearson = -10000
    for epoch_ in np.arange(5, 75, 5):
        mlp.set_params(max_iter=epoch_)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson > best_pearson:
            best_pearson = pearson
            best_epoch = epoch_

    print("Best_epoch number:{}".format(best_epoch))
    return best_epoch


def find_best_momentum(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_pearson = 10000
    best_momentum = 0.9
    best_nev_momentum = True
    nev_moms = [True, False]
    for momentum in np.arange(0.001, 1, 0.1):
        for nev_mom in nev_moms:
            mlp.set_params(momentum=momentum, nesterovs_momentum=nev_mom)
            mlp.fit(training_inputs_, training_targets_)
            pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

            if pearson > best_pearson:
                best_pearson = pearson
                best_momentum = momentum
                best_nev_momentum = nev_mom

    print("Best_momentum:{}".format(best_momentum))
    print("Best_nesterovs_momentum:{}".format(best_nev_momentum))
    return best_momentum, best_nev_momentum



