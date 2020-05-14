from sklearn.model_selection import train_test_split
from cleanCode.dataset1.dataset1A import evaluationFunctions_1A as ef
import numpy as np
import pickle
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# Load inputs and targets from computer
inputs = np.load("data/inputs_good.npy")
targets = np.load("data/targets_good.npy")
X = np.load("data/x.npy")

input_mean = np.mean(inputs)
input_std = np.std(inputs)

# Normalizing the ecg data
for rownumber, rows  in enumerate(inputs):
    inputs[rownumber] = (rows - input_mean) / input_std

# Split into train and test
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.3)

# Import model
filename = 'mlp_algorithm_sgd_tanh'
loaded_model = pickle.load(open(filename, 'rb'))

# re-train model to new partitioning of data
loaded_model.fit(training_inputs, training_targets)

# Check performance of MLPRegressor, message is text over evaluation that will store in .txt document
ef.graph_predictions(loaded_model, testing_inputs, testing_targets, X, rows=5, columns=6)
visual_score = input("Please input visual score: ")
evaluations, pearson, r2, mse = ef.evaluate_performance(loaded_model, testing_inputs, testing_targets, training_inputs, training_targets,
                        message="Trial run 2")
print(evaluations)

dir = 'evaluations/sgd_tanh/'

np.save(dir + 'pearson', np.append(np.load(dir + 'pearson.npy', allow_pickle=True), pearson))
np.save(dir + 'mse', np.append(np.load(dir + 'mse.npy', allow_pickle=True), mse))
np.save(dir + 'r2', np.append(np.load(dir + 'r2.npy', allow_pickle=True), r2))
np.save(dir + 'visual_score', np.append(np.load(dir + 'visual_score.npy', allow_pickle=True), visual_score))
