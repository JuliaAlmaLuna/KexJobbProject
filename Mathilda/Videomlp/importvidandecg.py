import dataManipulationFunctions as dmf
import numpy as np


videoList = ["video_folder/Pat2_Vi2.avi", "video_folder/Pat3_Vi3.avi", "video_folder/Pat4_Vi4.avi"]
ecgList = ["video_folder/Pat2_3Trace.txt", "video_folder/Pat3_3Trace.txt", "video_folder/Pat4_3Trace.txt"]

video_list, ecg_list = dmf.createVidInputsAndTargetEcgs(videoList, ecgList)
video_list = np.array(video_list)
ecg_list = np.array(ecg_list)

np.save("video_list", video_list)
np.save("ecg_list", ecg_list)


