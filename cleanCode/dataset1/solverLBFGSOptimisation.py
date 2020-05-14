import numpy as np
from scipy.stats import pearsonr
import math


def start(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp, alpha=True, val_frac=True,
          tolerance=True, hidden_layer_sizes=True, max_iter=True, max_fun=True):
    mlp.fit(training_inputs_, training_targets_)

    if alpha:
        alpha = find_best_alpha(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(alpha=alpha)

    if val_frac:
        val_frac = find_best_val_frac(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(validation_fraction=val_frac)

    if tolerance:
        tolerance = find_best_tolerance(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(tol=tolerance)

    if max_fun:
        maxfun = find_best_maxfun(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(max_fun=maxfun)

    if hidden_layer_sizes:
        layer_size = find_best_layer_size(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(hidden_layer_sizes=layer_size)

    if max_iter:
        epoch = find_best_epoch(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp)
        mlp.set_params(max_iter=epoch)

    return mlp


def pearson_score(testing_inputs, testing_targets, mlp, best_score):
    test_prediction = mlp.predict(testing_inputs)
    total = 0  # Finding the mean pearson score of testing data
    for index in range(len(testing_inputs)):
        pearson_corr, _ = pearsonr(testing_targets[index, :], test_prediction[index, :])
        if math.isnan(pearson_corr):
            print('')
        else:
            total = total + pearson_corr

    pearson = total / len(testing_inputs)
    return pearson


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


def find_best_maxfun(training_inputs_, training_targets_, testing_inputs_, testing_targets_, mlp):
    best_epoch = 10000
    best_pearson = -10000
    for epoch_ in np.arange(10000, 50000, 1000):
        mlp.set_params(max_fun=epoch_)
        mlp.fit(training_inputs_, training_targets_)
        pearson = pearson_score(testing_inputs_, testing_targets_, mlp, best_pearson)

        if pearson > best_pearson:
            best_pearson = pearson
            best_epoch = epoch_

    print("Best_max_fun number:{}".format(best_epoch))
    return best_epoch

