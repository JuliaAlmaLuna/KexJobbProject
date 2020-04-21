import dataManipulationFunctions as dmf
import numpy as np


#Using this file/function you can save video files as arrays of important pixel derivatives
# And using image division value you can make the image smaller using div*div square averaging
# Reducing height and width of the frames by div

div = 4

for t in range(0, 20):
    print(t)

    file_folder = "Julia/ecg_folder/Pat"
    ecg_suffix = "_3Trace.txt"
    avi_suffix1 = "_Vi.avi"

    videoList = []
    ecgList = []
    for x in range(t*5,t*5+5):
        videoList.append("Julia/ecg_folder/Pat{}_Vi.avi".format(x+1))
        ecgList.append("Julia/ecg_folder/Pat{}_3Trace.txt".format(x+1))


    #videoList = ["Julia/ecg_folder/Pat3_Vi.avi", "Julia/ecg_folder/Pat4_Vi.avi",]
    #ecgList = ["Julia/ecg_folder/Pat3_3Trace.txt", "Julia/ecg_folder/Pat4_3Trace.txt"]



    #n_video_list = dmf.listOfVidsToListOfNestedPixelList(videoList)

    n_video_list, n_ecg_list, ecg_x_list = dmf.createVidInputsAndTargetEcgs(videoList, ecgList, div)



    video_list = np.array(n_video_list)
    ecg_list = np.array(n_ecg_list)



    rowlength = 0

    for i in range(int(len(n_video_list))):
        rowlength = rowlength + len(n_video_list[i])

    video_numpy_correct_array = np.empty([rowlength, int(len(n_video_list[0][0]))])
    ecg_numpy_correct_array = np.empty([rowlength, 1])

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

    print("ohnoo")
    print(video_numpy_correct_array .size)
    print(video_numpy_correct_array .shape)



    print("ohnoo")
    print(ecg_numpy_correct_array .size)
    print(ecg_numpy_correct_array .shape)

    np.save("video_list{}".format(t+1), video_numpy_correct_array)
    np.save("ecg_list{}".format(t+1), ecg_numpy_correct_array)
