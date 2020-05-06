import matplotlib.pyplot as plt
import numpy as np


def one_peak(peaks, target_whole, input_whole, X):
    one_ecg = target_whole[0:peaks[0]]
    one_x = np.linspace(0, X[peaks[0]], len(one_ecg))
    one_doppler = input_whole[0:peaks[0]]
    plt.plot(one_x, one_ecg)
    plt.show()

    pieces = {
        "ecg": [one_ecg],
        "doppler": [one_doppler],
        "x": [one_x]
    }
    return pieces


def two_peaks(peaks, target_whole, input_whole, X):
    one_ecg = target_whole[0:peaks[0]]
    one_x = np.linspace(0, X[peaks[0]], len(one_ecg))
    one_doppler = input_whole[0:peaks[0]]
    plt.plot(one_x, one_ecg)
    plt.show()

    two_ecg = target_whole[peaks[0]:peaks[1]]
    two_x = np.linspace(0, X[peaks[1]] - X[peaks[0]], len(two_ecg))
    two_doppler = input_whole[peaks[0]:peaks[1]]
    plt.plot(two_x, two_ecg)
    plt.show()

    pieces = {
        "ecg": [one_ecg, two_ecg],
        "doppler": [one_doppler, two_doppler],
        "x": [one_x, two_x]
    }
    return pieces


def two_peaks_one_piece(peaks, target_whole, input_whole, X):
    one_ecg = target_whole[peaks[0]:peaks[1]]
    one_x = np.linspace(0, X[peaks[1]]-X[peaks[0]], len(one_ecg))
    one_doppler = input_whole[peaks[0]:peaks[1]]
    plt.plot(one_x, one_ecg)
    plt.show()

    pieces = {
        "ecg": [one_ecg],
        "doppler": [one_doppler],
        "x": [one_x]
    }
    return pieces


def three_peaks(peaks, target_whole, input_whole, X):
    one_ecg = target_whole[0:peaks[0]]
    one_x = np.linspace(0, X[peaks[0]], len(one_ecg))
    one_doppler = input_whole[0:peaks[0]]
    plt.plot(one_x, one_ecg)
    plt.show()

    two_ecg = target_whole[peaks[0]:peaks[1]]
    two_x = np.linspace(0, X[peaks[1]] - X[peaks[0]], len(two_ecg))
    two_doppler = input_whole[peaks[0]:peaks[1]]
    plt.plot(two_x, two_ecg)
    plt.show()

    three_ecg = target_whole[peaks[1]:peaks[2]]
    three_x = np.linspace(0, X[peaks[2]] - X[peaks[1]], len(three_ecg))
    three_doppler = input_whole[peaks[1]:peaks[2]]
    plt.plot(three_x, three_ecg)
    plt.show()

    pieces = {
        "ecg": [one_ecg, two_ecg, three_ecg],
        "doppler": [one_doppler, two_doppler, three_doppler],
        "x": [one_x, two_x, three_x]
    }
    return pieces


def three_peaks_two_piece(peaks, target_whole, input_whole, X):
    one_ecg = target_whole[peaks[0]:peaks[1]]
    one_x = np.linspace(0, X[peaks[1]]-X[peaks[0]], len(one_ecg))
    one_doppler = input_whole[peaks[0]:peaks[1]]
    plt.plot(one_x, one_ecg)
    plt.show()

    two_ecg = target_whole[peaks[1]:peaks[2]]
    two_x = np.linspace(0, X[peaks[2]] - X[peaks[1]], len(two_ecg))
    two_doppler = input_whole[peaks[1]:peaks[2]]
    plt.plot(two_x, two_ecg)
    plt.show()

    pieces = {
        "ecg": [one_ecg, two_ecg],
        "doppler": [one_doppler, two_doppler],
        "x": [one_x, two_x]
    }
    return pieces


def four_peaks(peaks, target_whole, input_whole, X):
    one_ecg = target_whole[0:peaks[0]]
    one_x = np.linspace(0, X[peaks[0]], len(one_ecg))
    one_doppler = input_whole[0:peaks[0]]
    plt.plot(one_x, one_ecg)
    plt.show()

    two_ecg = target_whole[peaks[0]:peaks[1]]
    two_x = np.linspace(0, X[peaks[1]] - X[peaks[0]], len(two_ecg))
    two_doppler = input_whole[peaks[0]:peaks[1]]
    plt.plot(two_x, two_ecg)
    plt.show()

    three_ecg = target_whole[peaks[1]:peaks[2]]
    three_x = np.linspace(0, X[peaks[2]] - X[peaks[1]], len(three_ecg))
    three_doppler = input_whole[peaks[1]:peaks[2]]
    plt.plot(three_x, three_ecg)
    plt.show()

    four_ecg = target_whole[peaks[2]:peaks[3]]
    four_x = np.linspace(0, X[peaks[3]] - X[peaks[2]], len(four_ecg))
    four_doppler = input_whole[peaks[2]:peaks[3]]
    plt.plot(four_x, four_ecg)
    plt.show()

    pieces = {
        "ecg": [one_ecg, two_ecg, three_ecg, four_ecg],
        "doppler": [one_doppler, two_doppler, three_doppler, four_doppler],
        "x": [one_x, two_x, three_x, four_x]
    }
    return pieces


def four_peaks_tree_pieces(peaks, target_whole, input_whole, X):
    one_ecg = target_whole[peaks[0]:peaks[1]]
    one_x = np.linspace(0, X[peaks[1]]-X[peaks[0]], len(one_ecg))
    one_doppler = input_whole[peaks[0]:peaks[1]]
    plt.plot(one_x, one_ecg)
    plt.show()

    two_ecg = target_whole[peaks[1]:peaks[2]]
    two_x = np.linspace(0, X[peaks[2]] - X[peaks[1]], len(two_ecg))
    two_doppler = input_whole[peaks[1]:peaks[2]]
    plt.plot(two_x, two_ecg)
    plt.show()

    three_ecg = target_whole[peaks[2]:peaks[3]]
    three_x = np.linspace(0, X[peaks[3]] - X[peaks[2]], len(three_ecg))
    three_doppler = input_whole[peaks[2]:peaks[3]]
    plt.plot(three_x, three_ecg)
    plt.show()

    pieces = {
        "ecg": [one_ecg, two_ecg, three_ecg],
        "doppler": [one_doppler, two_doppler, three_doppler],
        "x": [one_x, two_x, three_x]
    }
    return pieces