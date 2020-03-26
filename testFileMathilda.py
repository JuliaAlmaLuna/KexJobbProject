import numpy as np
import dataManipulationFunctions as dmf


# SKIT I DENNA :)


input_dim = 250
learning_rate = 0.01
weights = np.random.rand(input_dim,2)

# Behövs en funktion som samlar all training data!

ecg8, v8 = dmf.import_td_text_file("8_ecg.txt")
v8 = dmf.trim_array(v8, 250)
ecg2, v2 = dmf.import_td_text_file("2_ecg.txt")
v2 = dmf.trim_array(v2, 250)

v = np.array([v8, v2])
training_count = len(v[:, 0, 0])

# Each complete pass through the entire training set is called an epoch
for epoch in range(0,1):
    for dataset in range(0, training_count):
        Output_Sum = np.sum(np.multiply(v[dataset,:,:], weights))
        print(Output_Sum)