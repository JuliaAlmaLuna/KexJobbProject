import numpy as np
from Mathilda.mlpcolordoppler import dopplerManipulationFunctions as dmf

file_folder = "ecg_folder/goodAndMedium/"
suffix = "_ecg.txt"
good_array = [1, 2, 3, 6, 7, 10, 15, 16, 19, 20, 24, 27, 29, 33, 34, 39, 40, 41, 44, 47, 48, 49, 51, 53, 55, 56, 57, 58,
              60, 61, 63, 64, 66, 67, 68, 69, 71, 72, 76, 77, 79, 83, 84, 86, 87, 90, 92, 93, 94, 95, 97]
medium_and_good_array = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 23, 24, 27, 29, 30, 31, 32, 33,
                         34, 39, 40, 41, 44, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                         67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 79, 81, 82, 83, 84, 86, 87, 88, 90, 91, 92, 93, 94, 95,
                         97]
filenames = dmf.init_file_names(file_folder, suffix, medium_and_good_array)
targets, inputs, x = dmf.get_data(filenames, 3)
np.save("inputs_good_medium", inputs)
np.save("targets_good_medium", targets)
np.save("x", x)

'''import numpy as np
from Mathilda.mlpcolordoppler import dopplerManipulationFunctions as dmf

file_folder = "ecg_folder/goodAndMedium/"
suffix = "_ecg.txt"
good_array = [1, 2, 3, 6, 7, 10, 15, 16, 19, 20, 24, 27, 29, 33, 34, 39, 40, 41, 44, 47, 48, 49, 51, 53, 55, 56, 57, 58,
              60, 61, 63, 64, 66, 67, 68, 69, 71, 72, 76, 77, 79, 83, 84, 86, 87, 90, 92, 93, 94, 95, 97]
medium_and_good_array = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 23, 24, 27, 29, 30, 31, 32, 33,
                         34, 39, 40, 41, 44, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                         67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 79, 81, 82, 83, 84, 86, 87, 88, 90, 91, 92, 93, 94, 95,
                         97]
filenames = dmf.init_file_names(file_folder, suffix, medium_and_good_array)
targets, inputs, x = dmf.get_data(filenames, 3)

new_inputs = []
new_targets = []
x1 = np.array(x)
x2 = np.array(x)

for index in range(len(targets)):
    temp = np.concatenate((x1.reshape(-1, 1), inputs[index].reshape(-1, 1)), axis=1)
    temp2 = np.concatenate((x2.reshape(-1, 1), targets[index].reshape(-1, 1)), axis=1)
    new_inputs.append(temp)
    new_targets.append(temp2)

new_inputs = np.array(new_inputs)
new_targets = np.array(new_targets)
new_inputs = np.float32(new_inputs)
new_targets = np.float32(new_targets)

np.save("inputs_good_medium_pelt", new_inputs)
np.save("targets_good_medium_pelt", new_targets)'''
