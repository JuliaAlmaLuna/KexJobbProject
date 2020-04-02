from scipy.misc import electrocardiogram
import numpy as np
from scipy.interpolate import interp1d


def cut_generic_ecg_(t):
    generic_ecg = electrocardiogram()
    fs = 360
    time = np.arange(generic_ecg.size) / fs
    f_ecg = interp1d(time, generic_ecg)
    cut_generic_ecg = np.array(f_ecg(t))
    return cut_generic_ecg







