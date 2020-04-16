import numpy as np
from scipy.interpolate import interp1d
import cv2
import os
from PIL import Image
from dataManipulationFunctions import create_interpolation_function_ecg


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


def imgToList(imageString):
    imag = Image.open(imageString)
    # Convert the image te Greyscale if it is a .gif for example
    imag = imag.convert('L')

    pixelList = []

    # Get RGB
    img_width, img_height = imag.size
    for x in range(img_width):
        for y in range(img_height):
            if belongs_to_ultrasound(x, y):
                pixelBrightness = int(imag.getpixel((x, y))) # Retreives the pixel value of the pixel.
                pixelList.append(pixelBrightness)

    return pixelList


# Function returns true if pixel belongs to cone shape of ultrasound
def belongs_to_ultrasound(x, y):
    if y > -1.183*x + 672.79 and y > 1.183 * x - 529.14:
        if y < -(23 / 23814) * x * x + (11684 / 11907) * x + (4855163 / 11907):
            return True
        else:
            return False
    else:
        return False


def reduceImgSize(ListOfPixels, fraction):
    new_ListOfPixels = []

    if np.size(ListOfPixels) % fraction != 0:
        return ValueError

    for x in range(np.size(ListOfPixels)/fraction):
        for y in range(fraction):
            temp = temp + ListOfPixels[(x * fraction) + y]
        temp = temp/fraction
        new_ListOfPixels.extend(temp)


def vidToNestedPixelList(video):
    imageString, amount = vidToImg(video)

    nestedPixelList = []

    for frame in range(amount):
        temp = imgToList("data/frame%d.jpg" %frame)
        nestedPixelList.append(temp)

    return nestedPixelList


# TODO: try this aswell, maybe it works better?
def vidToLongPixelList(video):
    imageString, amount = vidToImg(video)
    longPixelList = []

    for frame in range(amount):
        temp = imgToList("data/frame%d.jpg" % frame)
        longPixelList.extend(temp)

    return longPixelList


def listOfVidsToListOfNestedPixelList(videoList):
    video_list = []
    for video in videoList:
        temp = vidToNestedPixelList(video)
        video_list.append(temp)
    return video_list


def get_data_ecg2(f_name, sample_number=0):
    if sample_number == 0:
        f_ecg, x = create_interpolation_function_ecg(f_name, 950)
    else:
        f_ecg, x = create_interpolation_function_ecg(f_name, sample_number)

    temp_f_ecg = discretizeECG(f_ecg(x))
    targets = temp_f_ecg

    targets = np.array(targets)
    return targets, x


def createVidInputsAndTargetEcgs(videoList, ecgList):
    vid_list = []
    ecg_list = []
    time_list = []

    for i, video in enumerate(videoList):
        temp = vidToNestedPixelList(video)
        temp2, tempx = get_data_ecg2(ecgList[i], len(temp))
        vid_list.append(temp)
        ecg_list.append(temp2)
        time_list.append(tempx)

    return vid_list, ecg_list


# Gets the derivate of the pixel values in regards to next and previous frame
def imgToDerivateOfImg(imgList):

    temp_list = imgList

    for img_number,img in enumerate(imgList, 1):
        print(len(temp_list) - 2)
        for pixel_number, pixel in enumerate(img,0):
            if img_number > len(temp_list)-2:
                temp_list[img_number-1][pixel_number] = temp_list[img_number-2][pixel_number]
                temp_list[0][pixel_number] = temp_list[1][pixel_number]
            else:
                temp_list[img_number][pixel_number] = imgList[img_number+1][pixel_number] -  imgList[img_number-1][pixel_number]

    return temp_list


def discretizeECG(ecg):
    temp = ecg

    for x, ecgVal in enumerate(ecg):
        temp[x] = int(round(ecgVal))

    return temp