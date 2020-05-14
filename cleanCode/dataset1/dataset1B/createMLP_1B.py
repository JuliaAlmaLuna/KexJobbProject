from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from cleanCode.dataset1.solverLBFGSOptimisation import start as lbfgs_start
from cleanCode.dataset1.solverAdamOptimisation import start as adam_start
from cleanCode.dataset1.solverSgdOptimisation import start as sgd_start
import pickle
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Load inputs and targets from computer
inputs = np.load("data/inputs_pieces.npy", allow_pickle=True)
targets = np.load("data/targets_pieces.npy", allow_pickle=True)
X = np.load("data/X_pieces.npy", allow_pickle=True)

# Split into train and test
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3)


# Init MLP regressor, add parameter random_state if you want to use same portion

mlp = MLPRegressor(activation='relu', solver='sgd')

# Optimise MLP with optimiser of choice
optimised_mlp = lbfgs_start(training_inputs, training_targets, testing_inputs, testing_targets, mlp)

# Pickle the model to save it
filename = 'mlp_algorithm_sgd_relu'
pickle.dump(mlp, open(filename, 'wb'))