from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from cleanCode.dataset1.solverAdamOptimisation import start as adam_start
from cleanCode.dataset1.solverSgdOptimisation import start as sgd_start
from cleanCode.dataset1.solverLBFGSOptimisation import start as lbfgs_start
import pickle
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Load inputs and targets from computer
inputs = np.load("data/inputs_good_medium.npy")
targets = np.load("data/targets_good_medium.npy")
X = np.load("data/x.npy")

# Normalizing the data
input_mean = np.mean(inputs)
input_std = np.std(inputs)
for rownumber, rows  in enumerate(inputs):
    inputs[rownumber] = (rows - input_mean) / input_std

# Split into train and test
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3)


# Init MLP regressor, add parameter random_state if you want to use same portion

'''mlp = MLPRegressor(
    hidden_layer_sizes=(100,), activation='relu', solver='sgd', alpha=0.2851, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.95, power_t=0.5, max_iter=319, shuffle=True,
    tol=0.00011, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.35, beta_1=0.3, beta_2=0.5, epsilon=1e-08, n_iter_no_change=5010)
'''

mlp = MLPRegressor(solver='sgd', activation='logistic')

# Use solverAdamOptimisation to find high startparameters for adam solver (This step takes a long time)
# Use sgd_start if you want to use sgd solver
# This step can be done multiple times
optimized_mlp = sgd_start(training_inputs, training_targets, testing_inputs, testing_targets, mlp)

# Pickle the model to save it
filename = 'mlp_algorithm_sgd_logistic'
pickle.dump(optimized_mlp, open(filename, 'wb'))

mlp.fit(training_inputs, training_targets)
