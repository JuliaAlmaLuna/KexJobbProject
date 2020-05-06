from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Load inputs and targets from computer
inputs = np.load("../../cleanCode/dataset1/dataset 1B/data/inputs_pieces.npy", allow_pickle=True)
targets = np.load("../../cleanCode/dataset1/dataset 1B/data/targets_pieces.npy", allow_pickle=True)
X = np.load("../../cleanCode/dataset1/dataset 1B/data/X_pieces.npy", allow_pickle=True)
sampling_size = len(inputs[0])

# Concatenate inputs and targets with x-axis
x_inputs = np.concatenate((X, inputs), axis=1)
x_targets = np.concatenate((X, targets), axis=1)

# Split into train and test
x_training_inputs, x_testing_inputs, x_training_targets, x_testing_targets = train_test_split(x_inputs, x_targets, test_size=0.3)


# Init MLP regressor, add parameter random_state if you want to use same portion

mlp = MLPRegressor(
    hidden_layer_sizes=(100,), activation='tanh', solver='lbfgs', alpha=0.2851, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.95, power_t=0.5, max_iter=319, shuffle=True,
    tol=0.00011, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.35, beta_1=0.3, beta_2=0.5, epsilon=1e-08, n_iter_no_change=5010)

# Fit the MLP with the new parameters
mlp.fit(x_training_inputs[:, sampling_size:2*sampling_size], x_training_targets[:, sampling_size:2*sampling_size])


predictions = mlp.predict(x_testing_inputs[:, sampling_size:2*sampling_size])





