import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

videos = np.load("video_list.npy", allow_pickle=True)
ecg = np.load("ecg_list.npy", allow_pickle=True)
training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(videos, ecg, test_size=0.3, random_state=66)

mlp = MLPRegressor(
    hidden_layer_sizes=(96,), activation='tanh', solver='adam', alpha=0.4211, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=45, tol=0.0001, verbose=True, warm_start=True, momentum=0.9, nesterovs_momentum=True,
    early_stopping=True, validation_fraction=0.1, beta_1=0.3, beta_2=0.5, epsilon=1e-08)

for index in range(len(training_inputs)):
    mlp.fit(training_inputs[index], training_targets[index])

for jindex in range(len(testing_inputs)):
    print(mlp.score(testing_inputs[jindex], testing_targets[jindex]))
    # TODO: maybe take the mean of the score?



