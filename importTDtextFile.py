
import numpy as np

def importTDtextFile(fname):
    v = np.loadtxt(fname,skiprows=3,usecols=(0,1),max_rows=363)
    ecg1 = np.loadtxt(fname,skiprows=3,usecols=(2,3),max_rows=363)
    ecg2 = np.loadtxt(fname,skiprows=366)
    ecg = np.concatenate((ecg1, ecg2), axis=0)

    return ecg, v

