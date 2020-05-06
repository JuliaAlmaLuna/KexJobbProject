import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.simplefilter(action='ignore', category=Warning)


def score(long_target, long_prediction):
    # pearson calculation
    pearson_corr, _ = pearsonr(long_target, long_prediction)
    score = "Pearson: " + str(pearson_corr)
    return score


def graph_long(x_testing_targets, x_testing_inputs, predictions, sampling_size):
    # Showing result as one long ecg
    long_prediction = []
    long_target = []
    long_input = []
    long_x = []

    X_axis = x_testing_targets[:, 0:sampling_size]
    targets_axis = x_testing_targets[:, sampling_size:sampling_size * 2]
    inputs_axis = x_testing_inputs[:, sampling_size:sampling_size * 2]
    previous_last_x_axis_value = 0

    for index in range(len(predictions)):
        long_prediction.extend(predictions[index])
        long_target.extend(targets_axis[index])
        long_input.extend(inputs_axis[index])
        long_x.extend(X_axis[index] + previous_last_x_axis_value)
        previous_last_x_axis_value = long_x[len(long_x) - 1]

    # plot
    fig, ax1 = plt.subplots()
    ax1.plot(long_x, long_input, 'y')

    ax2 = ax1.twinx()
    ax2.plot(long_x, long_target, 'g')
    ax2.plot(long_x, long_prediction, 'r')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    score_ = score(long_target, long_prediction)
    return score_
