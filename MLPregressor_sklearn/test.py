
import dataManipulationFunctions as dmf
import numpy as np
import matplotlib.pyplot as plt
from MLPregressor_sklearn import code

file_folder = "ECG_Folder/"
training_file_names = [file_folder + "2_ecg.txt", file_folder + "3_ecg.txt", file_folder + "6_ecg.txt", file_folder + "7_ecg.txt", file_folder + "4_ecg.txt", file_folder + "9_ecg.txt",
                       file_folder + "11_ecg.txt"]
testing_file_names = [file_folder + "5_ecg.txt", file_folder + "8_ecg.txt"]


def get_data(f_names):
    targets = []
    inputs = []

    for name in f_names:
        f_ecg, f_v, x = dmf.create_interpolation_function(name, 0.02, 2.31, 250)
        targets.append(f_ecg(x))
        inputs.append(f_v(x))

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x


training_targets, training_inputs, X = get_data(training_file_names)
testing_targets, testing_input, X = get_data(testing_file_names)

mlp = code.MLPRegressor(hidden_layer_sizes=(5,),  activation='tanh', solver='lbfgs', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=100000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


mlp.fit(training_inputs, training_targets)
prediction = mlp.predict(testing_input)
prediction1 = prediction[0, :]
prediction2 = prediction[1, :]
plt.plot(X, prediction1, 'y')
plt.plot(X, prediction2, 'm')
testing_targets1 = testing_targets[0, :]
testing_targets2 = testing_targets[1, :]
plt.plot(X, testing_targets1, 'g')
plt.plot(X, testing_targets2, 'r')
plt.show()


