import numpy as np
from scipy.interpolate import interp1d

inputs = []
targets = []
X = []

for index in range(1, 70):
    name = '/data/cut_data_dataset1/data_' + str(index) + '.npy'
    data = np.load(name, allow_pickle=True)
    ecg = data.item()['ecg']
    doppler = data.item()['doppler']
    x = data.item()['x']
    for index in range(len(ecg)):
        min_x = min(x[index])
        max_x = max(x[index])
        f_ecg = interp1d(x[index], ecg[index])
        f_doppler = interp1d(x[index], doppler[index])
        time = np.linspace(min_x, max_x, 200)

        target = f_ecg(time)
        input_ = f_doppler(time)

        targets.append(target)
        inputs.append(input_)
        X.append(time)

np.save('data/inputs_pieces', inputs)
np.save('data/targets_pieces', targets)
np.save('data/X_pieces', X)
