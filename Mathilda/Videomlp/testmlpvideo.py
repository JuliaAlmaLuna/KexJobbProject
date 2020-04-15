import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from Mathilda.mlpcolordoppler.mlpAlgorithmMathilda import evaluate_performance

number_of_files = 30 # TODO: set correct amount of saved files


def load_files_to_one_array(name_of_files, amount_of_files):
    array = []

    for index in range(1, amount_of_files+1):
        name = name_of_files + str(index) + ".npy"
        array_piece = np.load(name, allow_pickle=True)
        array.extend(array_piece)
    array = np.array(array)
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


videos = load_files_to_one_array("../video_list", number_of_files) # TODO: set right path
ecg = load_files_to_one_array("../ecg_list", number_of_files)
# TODO: Call either decrese method and update videos and ecg with them (This will most likely take a while)

training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(videos, ecg, test_size=0.3, random_state=66)

mlp = MLPRegressor(
    hidden_layer_sizes=(96,), activation='tanh', solver='adam', alpha=0.4211, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=45, tol=0.0001, verbose=True, warm_start=True, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.1, beta_1=0.3, beta_2=0.5, epsilon=1e-08)

mlp.fit(training_inputs, training_targets)
evaluate_performance(mlp, testing_inputs, testing_targets, training_inputs, training_targets, message="")


