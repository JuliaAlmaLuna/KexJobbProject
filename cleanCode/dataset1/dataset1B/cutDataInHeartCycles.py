import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import cleanCode.dataset1.dataset1B.peakFunctions as pf

''' Disclaimer: this is a semiautomatic function - follow step by step instructions:

1. Change index for every file
2. Check graph, if graph is not correct or undefined - stop the run
3. Update height to correctly classify
4. Rerun the code
5. If graph is now correct let the code run through'''

inputs_whole = np.load('../../cleanCode/dataset1/dataset 1A/data/inputs_good_medium.npy')
targets_whole = np.load('../../cleanCode/dataset1/dataset 1A/data/targets_good_medium.npy')
X = np.load('../../cleanCode/dataset1/dataset 1A/data/x.npy')
index = 75

target_whole = targets_whole[index]
input_whole = inputs_whole[index]

# Find height of R-peaks
max = 0
for datapoint in target_whole[5:len(target_whole)-5]:
    if datapoint > max:
        max = datapoint
height = max*0.7 # 70% of the max R-peak as minimum height for find_peaks, might need to tune this value

# Find R-peaks
peaks = find_peaks(target_whole, height=height)
peaks = peaks[0]

# Find which function to use
if target_whole[0] > max:
    # Graph starts on R-peak
    if len(peaks) == 1:
        function = pf.one_peak
        message = "one peak"
    elif len(peaks) == 2:
        function = pf.two_peaks
        message = "two peaks"
    elif len(peaks) == 3:
        function = pf.three_peaks
        message = "three peaks"
    elif len(peaks) == 4:
        function = pf.four_peaks
        message = "four peaks"
    else:
        message = "undetermined"
else:
    # Graph starts at random part of ECG
    if len(peaks) == 2:
        function = pf.two_peaks_one_piece
        message = "two peaks one piece"
    elif len(peaks) == 3:
        function = pf.three_peaks_two_piece
        message = "three peaks two piece"
    elif len(peaks) == 4:
        function = pf.four_peaks_tree_pieces
        message = "four peaks three piece"
    else:
        message = "undetermined"

# Plot peaks for visual confirmation
for peak_index in peaks:
    plt.plot(X[peak_index], target_whole[peak_index], '*')
plt.plot(X, targets_whole[index])
plt.title(message + ". Please check that this is correct")
plt.show()

# Save data to numpy array
data_index = index - 4
data = np.array(function(peaks, target_whole, input_whole, X))
np.save('data/data_' + str(data_index), data)








