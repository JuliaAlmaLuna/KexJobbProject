import numpy as np
from Mathilda.mlpcolordoppler import dopplerManipulationFunctions as dmf

file_folder = "../../cleanCode/dataset1/dataset1A/ecg_folder/goodAndMedium/"
suffix = "_ecg.txt"
good_array = [1, 2, 3, 6, 7, 10, 15, 16, 19, 20, 24, 27, 29, 33, 34, 39, 40, 41, 44, 47, 48, 49, 51, 53, 55, 56, 57, 58,
              60, 61, 63, 64, 66, 67, 68, 69, 71, 72, 76, 77, 79, 83, 84, 86, 87, 90, 92, 93, 94, 95, 97]
medium_and_good_array = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 23, 24, 27, 29, 30, 31, 32, 33,
                         34, 39, 40, 41, 44, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                         67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 79, 81, 82, 83, 84, 86, 87, 88, 90, 91, 92, 93, 94, 95,
                         97]
filenames = dmf.init_file_names(file_folder, suffix, medium_and_good_array)
targets, inputs, x = dmf.get_data(filenames, 3)
np.save("data/inputs_good_medium", inputs)
np.save("data/targets_good_medium", targets)
np.save("data/x", x)
