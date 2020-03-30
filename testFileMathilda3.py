from sklearn.neural_network import MLPRegressor
import dataManipulationFunctions as dmf
import numpy as np
import matplotlib.pyplot as plt

file_folder = "ECG_Folder/"
file_names = [file_folder + "2_ecg.txt", file_folder + "3_ecg.txt", file_folder + "4_ecg.txt"]
targets = []
inputs = []


for name in file_names:
    f_ecg, f_v, x = dmf.create_interpolation_function(name, 0.02, 2.31, 250)
    plt.plot(x, f_ecg(x))
    plt.show()
    targets.append(f_ecg(x))
    inputs.append(f_v(x))
targets = np.array(targets)
inputs = np.array(inputs)


f_ecg_test, f_v_test, x = dmf.create_interpolation_function(file_folder + "5_ecg.txt", 0.02, 2.31, 250)
test_input = np.array(f_v_test(x))
test_target = np.array(f_ecg_test(x))
test_input = np.reshape(test_input, (1, 250))
test_target = np.reshape(test_target, (1, 250))


mlp = MLPRegressor(
    hidden_layer_sizes=(5,),  activation='tanh', solver='lbfgs', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=10000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

mlp.fit(inputs, targets)
mlp.fit(inputs, targets)
mlp.fit(inputs, targets)
y = mlp.predict(test_input)
y = np.reshape(y, 250)
test_target = np.reshape(test_target, 250)
test_input = np.reshape(test_input, 250)
plt.plot(x, y, 'y')
plt.plot(x, test_target, 'b')
plt.plot(x, test_input, 'r')
plt.show()

performance = np.sum(np.square(y-test_target))
print(performance)



