import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from scipy.stats import pearsonr
from cleanCode.dataset1.dataset1A import evaluationFunctions_1A as ef
from cleanCode.dataset1 import solverAdamOptimisation as adam
import math

import warnings
warnings.simplefilter(action='ignore', category=Warning)

number_of_files = 10 # TODO: set correct amount of saved files
deriveSize = 4
chunksize = 64

def load_files_to_one_array(name_of_files, amount_of_files):
    array = []

    for index in range(1, amount_of_files+1):
        name = name_of_files + str(index) + ".npy"
        array_piece = np.load(name, allow_pickle=True)
        array.extend(array_piece)

    array = np.array(array)
    return array

arrayFolder = "dataset2F"

videos = load_files_to_one_array(arrayFolder + "/video_list", number_of_files) # TODO: set right path
ecg = load_files_to_one_array(arrayFolder + "/ecg_list", number_of_files)
X = load_files_to_one_array(arrayFolder + "/x_list", number_of_files)
# TODO: Call either decrese method and update videos and ecg with them (This will most likely take a while)

templength = int(np.size(videos,0))
templength2 = int(np.size(videos,1))
videos = np.reshape(videos, [int(templength/chunksize), int(chunksize * templength2)])

ecg = np.squeeze(ecg)

ecg = np.reshape(ecg, [int(np.size(videos,0)), int(chunksize)])
X = np.reshape(X, ecg.shape)

print("Shapes of arrays sent used in the Neural Net and for evaluation")
print("shape of video array: {}".format(np.shape(videos)))
print("shape of ecg array: {}".format(np.shape(ecg)))
print("shape of X (time) array: {}".format(np.shape(X)))

videoNames = []

for u in range(np.size(X,0)):
    videoNames.append("Video {}".format(u+1))


#Used this for loop to run through different test/train splits to find average
for runs in range(1,10):
    training_inputs, testing_inputs, training_targets, testing_targets, training_x, testing_x, training_videoNames, testing_videoNames = train_test_split(videos, ecg, X, videoNames, test_size=0.3, random_state=88)

    mlp_lbfgs = MLPRegressor(
        hidden_layer_sizes=(384,320,256,256,192,128), activation='relu', max_fun=90, solver='lbfgs', alpha=0.0006, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.1, power_t=0.5, max_iter=80, shuffle=True,
        random_state=20, tol=0.1, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=True, validation_fraction=0.1, beta_1=0.6, beta_2=0.6, epsilon=1e-08, n_iter_no_change=10)

    optimized_mlp = mlp_lbfgs

    #Use this if you want to optimize the parameters
    #for i in range(1):
    #   optimized_mlp = adam.startLbfgs(training_inputs, training_targets, testing_inputs, testing_targets, optimized_mlp)

    # Fit the MLP with the new parameters
    optimized_mlp.fit(training_inputs, training_targets)


    # Check performance of MLPRegressor, message is text over evaluation that will store in .txt documentp

    #Comment this out to skip the graphing part
    ef.graph_predictions_multi_x(optimized_mlp, testing_inputs, testing_targets, testing_x, rows=int(math.floor(len(testing_targets)/5)+1), columns=5, videoNames=testing_videoNames)

    evaluations = ef.evaluate_performance(optimized_mlp, testing_inputs, testing_targets, training_inputs, training_targets,
                            message="Trial run {} with random state {}".format(runs, runs*11))

    print(evaluations)
    text_file2 = open("mlp__video_performance{}.txt".format(runs), "w")
    n2 = text_file2.write(evaluations)
    text_file2.close()


