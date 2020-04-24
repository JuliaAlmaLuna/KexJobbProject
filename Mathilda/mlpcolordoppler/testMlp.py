from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from Mathilda.mlpcolordoppler import EvaluationFunctions as ef
import warnings
warnings.simplefilter(action='ignore', category=Warning)

inputs = np.load("inputs_good_medium.npy")
targets = np.load("targets_good_medium.npy")
X = np.load("x.npy")

input_mean = np.mean(inputs)
input_std = np.std(inputs)



#Normalizing the ecg data
for rownumber, rows  in enumerate(inputs):
    inputs[rownumber] = (rows - input_mean) / input_std


training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3)

mlp = MLPRegressor(
    hidden_layer_sizes=(119,), activation='tanh', solver='adam', alpha=0.8111, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.95, power_t=0.5, max_iter=53, shuffle=True,
    tol=0.00001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.45, beta_1=0.6, beta_2=0.8, epsilon=1e-08, n_iter_no_change=8150)

mlp.fit(training_inputs, training_targets)

ef.graph_predictions(mlp, testing_inputs[2:4], testing_targets[2:4], X, rows=4, columns=4)
evaluations = ef.evaluate_performance(mlp, testing_inputs[2:4], testing_targets[2:4], training_inputs, training_targets,
                        message="\nTrial run on good & medium data")
print(evaluations)