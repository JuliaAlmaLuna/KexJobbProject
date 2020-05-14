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
filename = 'mlp_algorithm_adam_relu'
loaded_model = pickle.load(open(filename, 'rb'))

# re-train model to new partitioning of data
loaded_model.fit(x_training_inputs[:, sampling_size:2*sampling_size], x_training_targets[:, sampling_size:2*sampling_size])

# Predict on training data
predictions = loaded_model.predict(x_testing_inputs[:, sampling_size:2*sampling_size])

# Graph and print score
score, pearson, mse, r2 = ef.graph_long(x_testing_targets, x_testing_inputs, predictions, sampling_size)
visual_score = input("Please input visual score: ")
print(score)

# Print parameters of mlp
print(str(loaded_model.get_params()))

dir = 'evaluations/adam_relu/'

pearson_arr = np.load(dir + 'pearson.npy', allow_pickle=True)
pearson_arr = np.append(pearson_arr, pearson)
mse_arr = np.load(dir + 'mse.npy', allow_pickle=True)
mse_arr = np.append(mse_arr, mse)
r2_arr = np.load(dir + 'r2.npy', allow_pickle=True)
r2_arr = np.append(r2_arr, r2)
visual_score_arr = np.load(dir + 'visual_score.npy', allow_pickle=True)
visual_score_arr = np.append(visual_score_arr, visual_score)

np.save(dir + 'pearson', pearson_arr)
np.save(dir + 'mse', mse_arr)
np.save(dir + 'r2', r2_arr)
np.save(dir + 'visual_score', visual_score_arr)
