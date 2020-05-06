from sklearn.model_selection import train_test_split
from cleanCode.dataset1.dataset1A import EvaluationFunctions1A as ef
import numpy as np
import pickle
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Load inputs and targets from computer
inputs = np.load("data/inputs_good_medium.npy")
targets = np.load("data/targets_good_medium.npy")
X = np.load("data/x.npy")

input_mean = np.mean(inputs)
input_std = np.std(inputs)

# Normalizing the ecg data
for rownumber, rows  in enumerate(inputs):
    inputs[rownumber] = (rows - input_mean) / input_std

# Split into train and test
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3)

# Import model
filename = 'mlp_algorithm'
loaded_model = pickle.load(open(filename, 'rb'))

# re-train model to new partitioning of data
loaded_model.fit(training_inputs, training_targets)


# Check performance of MLPRegressor, message is text over evaluation that will store in .txt document
ef.graph_predictions(loaded_model, testing_inputs, testing_targets, X, rows=5, columns=6)
evaluations = ef.evaluate_performance(loaded_model, testing_inputs, testing_targets, training_inputs, training_targets,
                        message="Trial run 2")
print(evaluations)
text_file = open("mlp_eval_1.txt", "w")
n = text_file.write(str(evaluations))
text_file.close()