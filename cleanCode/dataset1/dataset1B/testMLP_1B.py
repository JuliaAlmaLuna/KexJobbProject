import numpy as np
from sklearn.model_selection import train_test_split
from cleanCode.dataset1.dataset1B import evaluationFunctions_1B as ef
import pickle
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Load inputs and targets from computer
inputs = np.load("data/inputs_pieces.npy", allow_pickle=True)
targets = np.load("data/targets_pieces.npy", allow_pickle=True)
X = np.load("data/X_pieces.npy", allow_pickle=True)
sampling_size = len(inputs[0])

# Concatenate inputs and targets with x-axis
x_inputs = np.concatenate((X, inputs), axis=1)
x_targets = np.concatenate((X, targets), axis=1)

# Split into train and test
x_training_inputs, x_testing_inputs, x_training_targets, x_testing_targets = train_test_split(x_inputs, x_targets, test_size=0.3)

# Import model
filename = 'mlp_algorithm'
loaded_model = pickle.load(open(filename, 'rb'))

# re-train model to new partitioning of data
loaded_model.fit(x_training_inputs[:, sampling_size:2*sampling_size], x_training_targets[:, sampling_size:2*sampling_size])

# Predict on training data
predictions = loaded_model.predict(x_testing_inputs[:, sampling_size:2*sampling_size])

# Graph and print score
score = ef.graph_long(x_testing_targets, x_testing_inputs, predictions, sampling_size)
print(score)

