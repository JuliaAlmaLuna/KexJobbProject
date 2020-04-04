
import numpy as np
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from datetime import datetime
import dataManipulationFunctions as dmf

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import testFileJulia3 as NN

file_folder = "MLPregressor_sklearn/ECG_Folder/"
training_file_names = [file_folder + "2_ecg.txt", file_folder + "3_ecg.txt", file_folder + "6_ecg.txt", file_folder + "7_ecg.txt", file_folder + "4_ecg.txt", file_folder + "9_ecg.txt",
                       file_folder + "11_ecg.txt"]
testing_file_names = [file_folder + "5_ecg.txt", file_folder + "8_ecg.txt"]


training_targets, training_inputs, temp1, temp2 = NN.get_data(training_file_names)
testing_targets, testing_inputs, temp1, temp2 = NN.get_data(testing_file_names)
training_inputs  = training_inputs.reshape(1750,2)

print(training_targets.shape)
print(training_inputs.shape)
print(testing_inputs.shape)
print(testing_targets.shape)

training_targets, norm_targets = dmf.normalizeData(training_targets)
training_inputs, norm_inputs = dmf.normalizeData(training_inputs)

print(np.average(training_targets))
print(np.average(training_inputs))

print(training_inputs[15][0])
print(training_inputs[15][1])

print(training_inputs[333][0])
print(training_inputs[333][1])