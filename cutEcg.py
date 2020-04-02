import dataManipulationFunctions as dmf
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals.ecg import engzee_segmenter
from biosppy.signals.ecg import ssf_segmenter

file_folder = "MLPregressor_sklearn/ECG_Folder/"
file_names = [file_folder + "2_ecg.txt", file_folder + "3_ecg.txt", file_folder + "4_ecg.txt"]
ecg_signals, td_signals, t = dmf.get_data(file_names)


def cut_ecg_into_rpeak_to_peak(ecg_signal, time):
    rpeaks_tuple = ssf_segmenter(ecg_signal)
    rpeaks_array = np.array(rpeaks_tuple[0])
    print("Found " + str(len(rpeaks_array)) + " peaks")
    cut_ecg = []
    cut_t = []

    for index in range(len(rpeaks_array) - 1): # TODO: implement a check if it only finds one peak
        cut_ecg.append(ecg_signal[rpeaks_array[index]:rpeaks_array[index+1]])
        cut_t.append(time[rpeaks_array[index]:rpeaks_array[index+1]])
    cut_ecg = np.array(cut_ecg)
    cut_t = np.array(cut_t)
    return cut_ecg, cut_t, rpeaks_array


def cut_td_from_ecg(td, rpeaks):
    cut_td = []

    for index in range(len(rpeaks)-1): # TODO: implement a check if it only finds one peak
        cut_td.append(td[rpeaks[index]:rpeaks[index+1]])
    cut_td = np.array(cut_td)
    return cut_td


cut_ecg, cut_t, rpeak = cut_ecg_into_rpeak_to_peak(ecg_signals[1], t)
cut_td = cut_td_from_ecg(td_signals[1], rpeak)
plt.subplot(3, 2, 1)
plt.plot(cut_t[0], cut_ecg[0])
plt.subplot(3, 2, 2)
plt.plot(cut_t[0], cut_td[0])
plt.subplot(3, 2, 3)
plt.plot(cut_t[1], cut_ecg[1])
plt.subplot(3, 2, 4)
plt.plot(cut_t[1], cut_td[1])
plt.subplot(3, 2, 5)
plt.plot(t, ecg_signals[1])
plt.subplot(3, 2, 6)
plt.plot(t, td_signals[1])
plt.show()








