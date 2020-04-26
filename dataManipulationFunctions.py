import numpy as np
from scipy.interpolate import interp1d
import cv2
import os
from PIL import Image
from math import sqrt
from itertools import islice
import math


# Help function for import td txt file - finds where there are no longer 4 columns by throwing exception
def find_max_rows(fname):
    i = 1  # The smallest array size possible for TD data
    while 1:
        try:
            np.loadtxt(fname, skiprows=3, usecols=(2, 3), max_rows=i)
        except IndexError:
            break
        else:
            i = i + 1
    return i - 1


# Help function for import td txt file - finds where there are no longer 4 columns by throwing exception
def find_max_rows_3Trace(fname):
    i = 1  # The smallest array size possible for TD data
    while 1:
        try:
            np.loadtxt(fname, skiprows=3, usecols=(4, 5), max_rows=i)
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


# Function to import TD+ECG data from txt file
def import_td_text_file_ecg(fname):
    max_rows = find_max_rows_3Trace(fname)
    print("max rows:{}".format(max_rows))
    ecg1 = np.loadtxt(fname, skiprows=3, usecols=(4, 5), max_rows=max_rows)
    ecg2 = np.loadtxt(fname, skiprows=max_rows + 3)
    ecg = np.concatenate((ecg1, ecg2), axis=0)

    return ecg


def normalizeData(data):
    data_norm = np.linalg.norm(data, np.inf)
    data = np.divide(data, data_norm)
    data = np.add(data, 0.5)

    return data, data_norm


def reNormalizeData(data, data_norm):
    data = np.subtract(data, 0.5)
    data = np.multiply(data, data_norm)
    return data


# create an interpolation function - range of function is from lowest value in file to highest value in file
# for 2,3,4 & 5 max range is 0.02 to 2.31
def create_interpolation_function(fname, sample_number):
    ecg, v = import_td_text_file(fname)
    vx = v[:, 0]
    vy = v[:, 1]
    f_v = interp1d(vx, vy)
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    f_ecg = interp1d(ecgx, ecgy)
    # np.lins
    min_x = ecg[0]
    max_x = ecg[len(ecg)-1]
    x = np.linspace(min_x, max_x, sample_number)
    return f_ecg, f_v, x


# create an interpolation function - range of function is from lowest value in file to highest value in file
# automatically finds min and max
def create_interpolation_function_ecg(fname, deriveSize):
    ecg = import_td_text_file_ecg(fname)
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    f_ecg = interp1d(ecgx, ecgy)

    min_x = ecgx[0]
    max_x = ecgx[len(ecgx)-1]

    x_length = max_x - min_x

    #x = np.linspace(min_x, max_x, sample_number)
    x = np.linspace(min_x, max_x, int(256/deriveSize))
    return f_ecg, x


# Same as above only taking ecg, v directly instead of fname
def create_interpolation_function_ecg_v(ecg, v, min_x, max_x, sample_number):
    vx = v[:, 0]
    vy = v[:, 1]
    f_v = interp1d(vx, vy)
    ecgx = ecg[:, 0]
    ecgy = ecg[:, 1]
    f_ecg = interp1d(ecgx, ecgy)
    # np.lins
    x = np.linspace(min_x, max_x, sample_number)
    return f_ecg, f_v, x


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


def get_data(f_names, sample_number=0):
    targets = []
    inputs = []

    for name in f_names:
        if sample_number == 0:
            f_ecg, f_v, x = create_interpolation_function(name, 0.02, 2.31, 950)
        else:
            f_ecg, f_v, x = create_interpolation_function(name, 0.02, 2.31, sample_number)
        targets.append(f_ecg(x))
        inputs.append(f_v(x))

    targets = np.array(targets)
    inputs = np.array(inputs)
    return targets, inputs, x


def get_data_ecg(f_names, sample_number=0):
    targets = []

    for name in f_names:
        if sample_number == 0:
            f_ecg, x = create_interpolation_function_ecg(name, 950)
        else:
            f_ecg, x = create_interpolation_function_ecg(name, sample_number)
        temp_f_ecg = discretizeECG(f_ecg(x))
        targets.append(temp_f_ecg)

    targets = np.array(targets)
    return targets, x


def get_data_ecg2(f_name, deriveSize):

    f_ecg, x = create_interpolation_function_ecg(f_name, deriveSize)

    temp_f_ecg = discretizeECG(f_ecg(x))
    targets = temp_f_ecg

    targets = np.array(targets)
    return targets, x


# Turns video files into a picture per frame.
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
    vidcap = cv2.VideoCapture(video)


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


# Function returns true if pixel belongs to cone shape of ultrasound
def belongs_to_ultrasound(x, y):
    if y > -1.183*x + 672.79 and y > 1.183 * x - 529.14:
        if y < -(23 / 23814) * x * x + (11684 / 11907) * x + (4855163 / 11907):
            return True
        else:
            return False
    else:
        return False


def imgToList(imageString, divideSize = 1):
    imag = Image.open(imageString)
    # Convert the image te Greyscale if it is a .gif for example
    imag = imag.convert('L')

    pixelList = []
    TempPixelBrightness = 0
    #Changed this function so that it can reduce the size of orig image by averaging.
    # Get RGB
    img_width, img_height = imag.size
    for x in range(int(img_width/divideSize)):
        for y in range(int(img_height/divideSize)):
            pixelsInNewPixel = 0
            TempPixelBrightness = 0
            for z in range(divideSize):
                for u in range(divideSize):
                    if belongs_to_ultrasound(x*divideSize+z,y*divideSize+u):
                        TempPixelBrightness = TempPixelBrightness + int(imag.getpixel((x*divideSize+z, y*divideSize+u)))  # Retreives the pixel value of the pixel and adds it.
                        pixelsInNewPixel = pixelsInNewPixel + 1
            if pixelsInNewPixel > 0:
                TempPixelBrightness2 = TempPixelBrightness/pixelsInNewPixel
                pixelList.append(TempPixelBrightness2)



            #if belongs_to_ultrasound(x, y):
            #    pixelBrightness = int(imag.getpixel((x, y))) # Retreives the pixel value of the pixel.
            #    pixelList.append(pixelBrightness)

    return pixelList

def reduceImgSize(ListOfPixels, fraction):
    new_ListOfPixels = []

    if np.size(ListOfPixels) % fraction != 0:
        return ValueError

    for x in range(np.size(ListOfPixels)/fraction):
        for y in range(fraction):
            temp = temp + ListOfPixels[(x * fraction) + y]
        temp = temp/fraction
        new_ListOfPixels.extend(temp)


def vidToNestedPixelList(video, div, derivesize, x):
    imageString, amount = vidToImg(video)

    nestedPixelList = []

    for frame in range(amount):
        temp = imgToList("data/frame%d.jpg" %frame, div)
        nestedPixelList.append(temp)

    print("HERE")
    print(np.shape(nestedPixelList))
    temp2 = imgToDerivateOfImg(nestedPixelList, derivesize)
    temp2 = imgSortOfInterpolation(temp2, x)

    # Normalizing the ecg data
    input_mean = np.mean(temp2)
    input_std = np.std(temp2)

    for rownumber, rows in enumerate(temp2):
        temp2[rownumber] = (rows - input_mean) / input_std

    return temp2
    #return nestedPixelList



def imgSortOfInterpolation(imagelist, x):

    print("len x{}".format(len(x)))
    print("len imageList{}".format(len(imagelist)))




    timesBigger = math.floor(len(x)/len(imagelist))
    remainder = len(x)%len(imagelist)

    newImageList = []
    for img_number, img in enumerate(imagelist, 0):
        for times in range(timesBigger):
            newImageList.append(imagelist[img_number])

        if remainder > 0:
            newImageList.append(imagelist[img_number])
            remainder = remainder - 1

    print("John<3")
    print(np.shape(newImageList))
    return newImageList

'''

temp2 = imgToDerivateOfImg(temp)

# Normalizing the ecg data
input_mean = np.mean(temp2)
input_std = np.std(temp2)

for rownumber, rows in enumerate(temp2):
    temp2[rownumber] = (rows - input_mean) / input_std
'''

def listOfVidsToListOfNestedPixelList(videoList):
    video_list = []
    for video in videoList:
        temp = vidToNestedPixelList(video)
        video_list.append(temp)
    return video_list


def createVidInputsAndTargetEcgs(videoList, ecgList, div, deriveSize = 1):
    vid_list = []
    ecg_list = []
    time_list = []


    for i, video in enumerate(videoList):
        #temp2, tempx = get_data_ecg2(ecgList[i], len(temp) * deriveSize)
        temp2, tempx = get_data_ecg2(ecgList[i], deriveSize)
        temp = vidToNestedPixelList(video, div, deriveSize, tempx)
        vid_list.append(temp)
        ecg_list.append(temp2)
        time_list.append(tempx)

    return vid_list, ecg_list, time_list


# Gets the derivate of the pixel values in regards to next and previous frame
def imgToDerivateOfImg(imgList, deriveSize = 1):

    temp_list = imgList
    '''
    for img_number,img in enumerate(imgList, 1):
        for pixel_number, pixel in enumerate(img ,0):
            if img_number > len(temp_list)-2:
                temp_list[img_number-1][pixel_number] = temp_list[img_number-2][pixel_number]
                temp_list[0][pixel_number] = temp_list[1][pixel_number]
            else:
                temp_list[img_number][pixel_number] = imgList[img_number+1][pixel_number] -  imgList[img_number-1][pixel_number]
    '''

    count = 1
    for img_number, img in enumerate(imgList, 1):
        if(img_number%deriveSize == 0):
            for pixel_number, pixel in enumerate(img, 0):
                if img_number > len(temp_list) - 2:
                    temp_list[img_number - 1][pixel_number] = temp_list[img_number - 2][pixel_number]
                    temp_list[0][pixel_number] = temp_list[1][pixel_number]
                else:
                    temp_list[count][pixel_number] = imgList[count + deriveSize][pixel_number] - imgList[count - deriveSize][
                        pixel_number]
            count = count +1

    for times in range(len(temp_list)-count):
        temp_list.pop()

    return temp_list


def discretizeECG(ecg):
    temp = ecg

    for x, ecgVal in enumerate(ecg):
        temp[x] = int(round(ecgVal))

    return temp


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