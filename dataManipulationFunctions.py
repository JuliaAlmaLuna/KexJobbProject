import numpy as np
from scipy.interpolate import interp1d
import cv2
import os
from PIL import Image
from math import sqrt


# Help function for import td txt file - finds where there are no longer 4 columns by throwing exception
def find_max_rows(fname):
    i = 250  # The smallest array size possible for TD data
    while 1:
        try:
            np.loadtxt(fname, skiprows=3, usecols=(2, 3), max_rows=i)
        except IndexError:
            break
        else:
            i = i + 1
    return i - 1


# Function to import TD+ECG data from txt file
def import_td_text_file(fname):
    max_rows = find_max_rows(fname)
    v = np.loadtxt(fname, skiprows=3, usecols=(0, 1), max_rows=max_rows)
    ecg1 = np.loadtxt(fname, skiprows=3, usecols=(2, 3), max_rows=max_rows)
    ecg2 = np.loadtxt(fname, skiprows=max_rows + 3)
    ecg = np.concatenate((ecg1, ecg2), axis=0)

    return ecg, v

def normalizeData(data):
    data_norm = np.linalg.norm(data, np.inf)
    data = np.divide(data, data_norm)
    data = np.add(data, 0.5)

    return data, data_norm


# create an interpolation function - range of function is from lowest value in file to highest value in file
# for 2,3,4 & 5 max range is 0.02 to 2.31
def create_interpolation_function(fname, min_x, max_x, sample_number):
    ecg, v = import_td_text_file(fname)
    vx = v[:, 0]
    vy = v[:, 1]
    f_v = interp1d(vx, vy)
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    f_ecg = interp1d(ecgx, ecgy)
    # np.lins
    x = np.linspace(min_x, max_x, sample_number)
    return f_ecg, f_v, x


# Den blev en blandning av interpolering och inte interpolering.
def other_not_interpolating_function(fname,  min_x, max_x, sample_number):
    ecg, v = import_td_text_file(fname)
    vx = v[:, 0]
    vy = v[:, 1]
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    total_time = vx[len(vx)-1] - vx[0]
    time_step = total_time / (len(vx)-1)
    exgy_avgd = []

    for ultra in vx:
        temp_val = 0
        temp_ecg_val = 0

        for spot, ecg_time in enumerate(ecgx, 0):

            if ecg_time < (ultra + (time_step/2)):
                if ecg_time > (ultra - (time_step/2)):
                    temp_val =+ 1
                    temp_ecg_val =+ ecgy[spot]

        temp_ecg_val = temp_ecg_val/temp_val
        exgy_avgd.append(temp_ecg_val)

    f_v = interp1d(vx, vy)
    f_ecg = interp1d(vx, exgy_avgd)

    vx = np.linspace(min_x, max_x, sample_number)

    return f_ecg, f_v, vx
    # return exgy_avgd, vy, vx


# Help function to trim array, creates an array of indexes for uniform removal of values
def create_index_remove_list(remove_quantity, remove_step):
    rmv_index = [0]
    value = 0

    while len(rmv_index) < remove_quantity:
        value = value + remove_step
        rmv_index.append(int(value))
    return rmv_index


# Function to reduce the size of an array by uniformly removing values
def trim_array(array, goal_size):
    size = len(array)
    rmv_quantity = size - goal_size
    rmv_step = size / rmv_quantity
    rmv_index = create_index_remove_list(rmv_quantity, rmv_step)
    trimmed_array = np.delete(array, obj=rmv_index, axis=0)
    return trimmed_array


def get_data(f_names):
    targets = []
    inputs = []

    for name in f_names:
        f_ecg, f_v, x = create_interpolation_function(name, 0.02, 2.31, 950)
        targets.append(f_ecg(x))
        inputs.append(f_v(x))

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x

#Turns video files into a picture per frame.
def vidToImg(video):
    # Read the video from specified path
    cam = cv2.VideoCapture(video)

    try:

        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    print(cv2.__version__)
    vidcap = cv2.VideoCapture('Pat7_Vi7.avi')
    success, image = vidcap.read()
    count = 0
    images = []
    while success:
        cv2.imwrite("data/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read frame:{} '.format(count), success)
        count += 1

    imageString = "data/frame%d.jpg"

    return imageString, count

def imgToList(imageString):
    imag = Image.open(imageString)
    # Convert the image te Greyscale if it is a .gif for example
    imag = imag.convert('L')

    pixelList = []

    # Get RGB
    img_width, img_height = imag.size
    for x in range(img_width):
        for y in range(img_height):
            pixelBrightness = imag.getpixel((x, y)) #Retreives the pixel value of the pixel.
            pixelList.append(pixelBrightness)

    return pixelList

def vidToNestedPixelList(video):
    imageString, amount = vidToImg(video)

    nestedPixelList = []

    for frame in range(amount):
        temp = imgToList("data/frame%d.jpg" %frame)
        nestedPixelList.append(temp)

    return nestedPixelList
