from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from Mathilda.mlpcolordoppler.solverAdamOptimisation import start as adam_start
from Mathilda.mlpcolordoppler import EvaluationFunctions as ef


# Load inputs and targets from computer
inputs = np.load("Mathilda/mlpcolordoppler/inputs.npy")
targets = np.load("Mathilda/mlpcolordoppler/targets.npy")
X = np.load("Mathilda/mlpcolordoppler/x.npy")
# Split into train and test
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3)

# Init MLP regressor, add parameter random_state if you want to use same portion
mlp_adam = MLPRegressor(
    hidden_layer_sizes=(83,), activation='tanh', solver='adam', alpha=0.999, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=100, shuffle=True,
    tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.3, beta_1=0.1, beta_2=0.1, epsilon=1e-08, n_iter_no_change=59)

# Use solverAdamOptimisation to find good startparameters for adam solver (This step takes a long time)
# Use solverSgdOptimisation if you want to use sgd solver
# This step can be done multiple times
message, optimized_mlp = adam_start(training_inputs, training_targets, testing_inputs, testing_targets, mlp_adam)
print(message)
text_file = open("Mathilda/mlpcolordoppler/mlp_parameters.txt", "w")
n = text_file.write(message)
text_file.close()
# Saving parameters so we can update the init of MLP regressor if we want to run script again

# Fit the MLP with the new parameters
optimized_mlp.fit(training_inputs, training_targets)

# Check performance of MLPRegressor, message is text over evaluation that will store in .txt document
ef.graph_predictions(optimized_mlp, testing_inputs, testing_targets, X, rows=5, columns=6)
evaluations = ef.evaluate_performance(optimized_mlp, testing_inputs, testing_targets, training_inputs, training_targets,
                        message="Trial run 1")
print(evaluations)
text_file2 = open("Mathilda/mlpcolordoppler/mlp_performance.txt", "w")
n2 = text_file.write(evaluations)
text_file2.close()









