import cv2
import os

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
    success,image = vidcap.read()
    count = 0
    images = []
    while success:
        cv2.imwrite("data/frame%d.jpg" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        print ('Read frame:{} '.format(count), success)
        count += 1
