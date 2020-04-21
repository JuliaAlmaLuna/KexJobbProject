from Mathilda.threeTracemlp import threeTraceManipulationFunctions as tmf
import numpy as np
from Mathilda.randomshit.graphEcgAndDoppler import graph_ecg_and_doppler

file_folder = "threeTrace_data/Pat"
suffix = "_3Trace.txt"
file_names = tmf.init_file_names(9, file_folder, suffix)
targets, inputs, x = tmf.get_data(file_names, 950)
for index in range(len(inputs)):
    graph_ecg_and_doppler(x, targets[index], inputs[index], "3Trace")