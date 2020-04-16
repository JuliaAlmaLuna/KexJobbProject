import numpy as np
from Mathilda.mlpcolordoppler import dopplerManipulationFunctions as dmf

number_of_sample_data = 97
file_folder = "../ecg_folder/"
suffix = "_ecg.txt"
filenames = dmf.init_file_names(number_of_sample_data, file_folder, suffix)
targets, inputs, x = dmf.get_data(filenames, 3)
np.save("inputs", inputs)
np.save("targets", targets)
np.save("x", x)