import numpy as np
import dataManipulationFunctions as dmf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from scipy.stats import pearsonr
from Mathilda.mlpcolordoppler import EvaluationFunctions as ef
from Mathilda.mlpcolordoppler import solverAdamOptimisation as adam
import math

import warnings
warnings.simplefilter(action='ignore', category=Warning)

number_of_files = 10 # TODO: set correct amount of saved files
deriveSize = 4
chunksize = 64


def evaluate_performance(mlp, testing_inputs, testing_targets, training_inputs, training_targets, message=""):
    train_prediction = mlp.predict(training_inputs)
    test_prediction = mlp.predict(testing_inputs)
    r2_score_ = mlp.score(testing_inputs, testing_targets)
    ev_score = explained_variance_score(testing_inputs, test_prediction, multioutput='uniform_average')
    train_mse = mean_squared_error(training_targets, train_prediction)
    test_mse = mean_squared_error(testing_targets, test_prediction)

    total = 0  # Finding the mean pearson score of testing data
    for index in range(len(testing_inputs)):
        pearson_corr, _ = pearsonr(testing_targets[index, :], test_prediction[index, :])
        total = total + pearson_corr
    pearson = total/len(testing_inputs)

    curr_max = 0  # Finding the maximum error in all the training data
    for index in range(len(testing_inputs)):
        max_error_ = max_error(testing_targets[index, :], test_prediction[index, :])
        if max_error_ > curr_max:
            curr_max = max_error_

    print(message)
    print("__________________________________________________________________________")
    print("R2 score:\t\t\t" + str(r2_score_) + "\t\t\tGoal: positive & close to 1" + "\ntest MSE:\t\t\t" + str(test_mse)
          + "\t\t\tGoal: lower than train MSE" + "\ntrain MSE:\t\t\t" + str(train_mse) + "\nEVS, ua:\t\t\t" +
          str(ev_score) + "\t\t\tGoal: Same as R2" + "\nMax error:\t\t\t" + str(curr_max) +
          "\t\t\tGoal: As low as possible" + "\nPearson:\t\t\t" + str(pearson) + "\t\t\tGoal: close to 1")
    print("__________________________________________________________________________")

def load_files_to_one_array(name_of_files, amount_of_files):
    array = []

    for index in range(1, amount_of_files+1):
        name = name_of_files + str(index) + ".npy"
        array_piece = np.load(name, allow_pickle=True)
        array.extend(array_piece)


    array = np.array(array)



    #if np.size(array)%238576 == 0:
    #    array.reshape([int(np.size(array)/238576), 238576])


    return array


# Leads to less samples eg. from (750, 23000) to (375, 23000)
def decrease_array_size_averaging(video, ecg, average_of):
    averaged_video = []
    averaged_ecg = []

    for index in range(0, len(video), average_of):
        temp1 = video[index:index+average_of, :]
        average_value1 = np.average(temp1, axis=0)
        averaged_video.append(average_value1)

        temp2 = ecg[index:index + average_of]
        average_value2 = np.average(temp2)
        averaged_ecg.append(average_value2)
    return averaged_video, averaged_ecg


# Leads to less features eg. from (750, 230 000) to (750, 47 000)
def decrease_array_size_less_pixels(videos, average_of):
    less_pixels_video = []

    for sample in videos:
        averaged_sample = []

        for index in range(0, len(sample), average_of):
            temp = sample[index:index+average_of]
            average_pixel = np.average(temp)
            averaged_sample.append(average_pixel)
        less_pixels_video.append(averaged_sample)
    print(np.shape(less_pixels_video))


videos = load_files_to_one_array("AEV_Div8Deri4Good/video_list", number_of_files) # TODO: set right path
ecg = load_files_to_one_array("AEV_Div8Deri4Good/ecg_list", number_of_files)
X = load_files_to_one_array("AEV_Div8Deri4Good/x_list", number_of_files)
# TODO: Call either decrese method and update videos and ecg with them (This will most likely take a while)

print(np.shape(videos))

templength = int(np.size(videos,0))
templength2 = int(np.size(videos,1))
videos = np.reshape(videos, [int(templength/chunksize), int(chunksize * templength2)])



ecg = np.squeeze(ecg)
print(np.shape(ecg))

ecg = np.reshape(ecg, [int(np.size(videos,0)), int(chunksize)])
X = np.reshape(X, ecg.shape)

print("oh")
print(np.shape(videos))
print(np.shape(ecg))
print(np.shape(X))

videoNames = []

for u in range(np.size(X,0)):
    videoNames.append("Video {}".format(u+1))
    print(videoNames[u])

#Testing out a resize to make it clump together ecg parts
#np.reshape(ecg, [np.size(ecg)/3,3])


#Used this for loop to run through different test/train splits to find average
for runs in range(1,2):
    training_inputs, testing_inputs, training_targets, testing_targets, training_x, testing_x, training_videoNames, testing_videoNames = train_test_split(videos, ecg, X, videoNames, test_size=0.3, random_state=88)


    #training_inputs2, testing_inputs2, training_targets2, testing_targets2 = train_test_split(videos, ecg, test_size=0.3)
    #training_inputs3, testing_inputs3, training_targets3, testing_targets3 = train_test_split(videos, ecg, test_size=0.3)

    print(np.shape(training_inputs))
    print(np.shape(testing_inputs))
    print(np.shape(training_targets))
    print(np.shape(testing_targets))




    mlp_adam = MLPRegressor(
        hidden_layer_sizes=(200), activation='relu', solver='lbfgs', alpha=0.4, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.1, power_t=0.5, max_iter=20, shuffle=True,
        random_state=20, tol=0.000001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=True, validation_fraction=0.1, beta_1=0.6, beta_2=0.6, epsilon=1e-08, n_iter_no_change=10)

    mlp_lbfgs = MLPRegressor(
        hidden_layer_sizes=(200), activation='relu', max_fun=40, solver='lbfgs', alpha=0.0018, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.1, power_t=0.5, max_iter=15, shuffle=True,
        random_state=20, tol=0.00002, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=True, validation_fraction=0.1, beta_1=0.6, beta_2=0.6, epsilon=1e-08, n_iter_no_change=10)

    optimized_mlp = mlp_adam
    optimized_mlp = mlp_lbfgs


    #for i in range(1):
    #    optimized_mlp = adam.startLbfgs(training_inputs, training_targets, testing_inputs, testing_targets, optimized_mlp)

    #optimized_mlp = adam.start(training_inputs, training_targets, testing_inputs, testing_targets, optimized_mlp)

    text_file = open("mlp_parameters_1.txt", "w")
    n = text_file.write(str(optimized_mlp.get_params))
    text_file.close()
    # Saving parameters so we can update the init of MLP regressor if we want to run script again

    # Fit the MLP with the new parameters



    #mlp_adam.fit(training_inputs, training_targets)
    optimized_mlp.fit(training_inputs, training_targets)


    # Check performance of MLPRegressor, message is text over evaluation that will store in .txt documentp


    #! HERE FIX AN X !
    #ef.graph_predictions(optimized_mlp, testing_inputs, testing_targets, X, rows=5, columns=6)


    ef.graph_predictions_multi_x(optimized_mlp, testing_inputs, testing_targets, testing_x, rows=int(math.floor(len(testing_targets)/5)+1), columns=5, videoNames=testing_videoNames)

    evaluations = ef.evaluate_performance(optimized_mlp, testing_inputs, testing_targets, training_inputs, training_targets,
                            message="Trial run {} with random state {}".format(runs, runs*11))

    #evaluations = ef.evaluate_performance(mlp_adam, testing_inputs, testing_targets, training_inputs, training_targets,
    #                        message="Trial run 1")
    print(evaluations)
    text_file2 = open("mlp__video_performance.txt", "w")
    n2 = text_file2.write(evaluations)
    text_file2.close()

    #mlp.fit(training_inputs, training_targets)
    #ef.evaluate_performance(mlp, testing_inputs, testing_targets, training_inputs, training_targets, message="")



