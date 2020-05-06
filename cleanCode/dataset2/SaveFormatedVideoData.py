import dataManipulationFunctions as dmf
import numpy as np
import os
import datetime

#Using this file/function you can save video files as arrays of important pixel derivatives
# And using image division value you can make the image smaller using div*div square averaging
# Reducing height and width of the frames by divz<s

div = 16
derivesize = 4


upsidedown_filenames = ['Pat9_3Trace.txt', 'Pat10_3Trace.txt', 'Pat11_3Trace.txt', 'Pat27_3Trace.txt',
                        'Pat33_3Trace.txt', 'Pat35_3Trace.txt', 'Pat40_3Trace.txt', 'Pat42_3Trace.txt']

high_quality_index = [2, 3, 5, 7, 12, 13, 15, 16, 17, 18, 20, 21, 24, 25, 29, 30, 32, 34, 36, 37, 38, 46, 47, 51,
                      52, 57, 58, 62, 63, 64, 66, 69, 70, 72, 73, 74, 75, 76, 78, 80, 84, 85, 87, 88, 89, 90, 94, 95,
                      97, 98, 100]
#41

high_and_medium_quality_index = [2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 51, 52, 53, 54, 56, 57,
                                 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 94, 95, 97, 98, 100]


for t in range(0, 15):
    print("Filepart {} started at {}".format(t+1, datetime.datetime.now()))


    file_folder = "Julia/ecg_folder/Pat"
    ecg_suffix = "_3Trace.txt"
    avi_suffix1 = "_Vi.avi"

    videoList = []
    ecgList = []
   # for x in range(t*5,t*5+5):
   #     videoList.append("Julia/ecg_folder/Pat{}_Vi.avi".format(x+1))
   #     ecgList.append("Julia/ecg_folder/Pat{}_3Trace.txt".format(x+1))

    for x in range(t*5,t*5+5):
        videoList.append("Julia/ecg_folder/Pat{}_Vi.avi".format(high_quality_index[x]))
        ecgList.append("Julia/ecg_folder/Pat{}_3Trace.txt".format(high_quality_index[x]))


    #videoList = ["Julia/ecg_folder/Pat3_Vi.avi", "Julia/ecg_folder/Pat4_Vi.avi",]
    #ecgList = ["Julia/ecg_folder/Pat3_3Trace.txt", "Julia/ecg_folder/Pat4_3Trace.txt"]



    #n_video_list = dmf.listOfVidsToListOfNestedPixelList(videoList)

    n_video_list, n_ecg_list, ecg_x_list = dmf.createVidInputsAndTargetEcgs(videoList, ecgList, div, derivesize)



    video_list = np.array(n_video_list)
    ecg_list = np.array(n_ecg_list)

   # print("ohnoo")
   # print(np.shape(n_video_list))
   # print(np.shape(n_ecg_list))
   # print(np.shape(ecg_x_list))



    rowlength = 0

    for i in range(int(len(n_video_list))):
        rowlength = rowlength + len(n_video_list[i])

    video_numpy_correct_array = np.empty([rowlength, int(len(n_video_list[0][0]))])
    ecg_numpy_correct_array = np.empty([int(5*256/derivesize), 1])
    x_numpy_correct_array = np.empty([int(5*256/derivesize), 1])

    count = 0

    for x in range(len(n_video_list)):
        for y in range(len(n_video_list[x])):
            for z in range(len(n_video_list[x][y])):
                video_numpy_correct_array [count][z] = n_video_list[x][y][z]
            count = count + 1

    count = 0
    for x in range(len(n_ecg_list)):
        for y in range(len(n_ecg_list[x])):
            ecg_numpy_correct_array[count][0] = n_ecg_list[x][y]
            count = count + 1

    count = 0
    for x in range(len(ecg_x_list)):
        for y in range(len(ecg_x_list[x])):
            x_numpy_correct_array[count][0] = ecg_x_list[x][y]
            count = count + 1

   # print("ohnoo")
   # print(video_numpy_correct_array .size)
  # print(video_numpy_correct_array .shape)



  #  print("ohnoo")
  #  print(ecg_numpy_correct_array .size)
  #  print(ecg_numpy_correct_array .shape)

  #  print("ohrooo")
   # print(ecg_numpy_correct_array.size)
  #  print(ecg_numpy_correct_array.shape)

    np.save("video_list{}".format(t+1), video_numpy_correct_array)
    np.save("ecg_list{}".format(t+1), ecg_numpy_correct_array)
    np.save("x_list{}".format(t+1), x_numpy_correct_array)

    print("Filepart {} saved at {}".format(t + 1, datetime.datetime.now()))