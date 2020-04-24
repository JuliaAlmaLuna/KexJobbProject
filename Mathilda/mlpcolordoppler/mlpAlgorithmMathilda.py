from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from Mathilda.mlpcolordoppler.solverAdamOptimisation import start as adam_start
from Mathilda.mlpcolordoppler.solverSgdOptimisation import start as sgd_start
from Mathilda.mlpcolordoppler import EvaluationFunctions as ef
import warnings
warnings.simplefilter(action='ignore', category=Warning)



# Load inputs and targets from computer
inputs = np.load("Mathilda/mlpcolordoppler/inputs_good_medium.npy")
targets = np.load("Mathilda/mlpcolordoppler/targets_good_medium.npy")
X = np.load("Mathilda/mlpcolordoppler/x.npy")


#temp_inputs = np.delete(inputs, [77,79,90], 0)
#temp_targets = np.delete(targets, [77,79,90], 0)

#inputs = temp_inputs
#targets = temp_targets


input_mean = np.mean(inputs)
input_std = np.std(inputs)



#Normalizing the ecg data
for rownumber, rows  in enumerate(inputs):
    inputs[rownumber] = (rows - input_mean) / input_std



# Split into train and test
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3,random_state=38)


# Init MLP regressor, add parameter random_state if you want to use same portion

mlp_adam = MLPRegressor(
    hidden_layer_sizes=(124,), activation='tanh', solver='adam', alpha=0.2851, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.95, power_t=0.5, max_iter=319, shuffle=True,
    tol=0.00011, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.35, beta_1=0.3, beta_2=0.5, epsilon=1e-08, n_iter_no_change=5010)


'''mlp_sgd = MLPRegressor(
    hidden_layer_sizes=(100,), activation='tanh', solver='sgd', alpha=0.999, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=100, shuffle=True,
    tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.3, beta_1=0.1, beta_2=0.1, epsilon=1e-08, n_iter_no_change=59)'''

#Bara värden från en som funkade bra för mig
'''
mlp_adam = MLPRegressor(
    hidden_layer_sizes=(137,), activation='relu', solver='adam', alpha=0.4421, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=160, shuffle=True,
    tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.2, beta_1=0.8, beta_2=0.3, epsilon=1e-08, n_iter_no_change=154)
'''

# Use solverAdamOptimisation to find good startparameters for adam solver (This step takes a long time)
# Use solverSgdOptimisation if you want to use sgd solver
# This step can be done multiple times
optimized_mlp = adam_start(training_inputs, training_targets, testing_inputs, testing_targets, mlp_adam)
text_file = open("mlp_parameters_1.txt", "w")
n = text_file.write(str(optimized_mlp.get_params))
# Saving parameters so we can update the init of MLP regressor if we want to run script again

# Fit the MLP with the new parameters
optimized_mlp.fit(training_inputs, training_targets)

# Check performance of MLPRegressor, message is text over evaluation that will store in .txt document
ef.graph_predictions(optimized_mlp, testing_inputs, testing_targets, X, rows=5, columns=6)
evaluations = ef.evaluate_performance(optimized_mlp, testing_inputs, testing_targets, training_inputs, training_targets,
                        message="Trial run 2")
print(evaluations)
n = text_file.write(str(evaluations))
text_file.close()









