import dataManipulationFunctions as dmf


file_folder = "Julia/ecg_folder/Pat"
ecg_suffix = "_3Trace.txt"
avi_suffix1 = "_Vi.avi"

videoList = ["Julia/ecg_folder/Pat3_Vi.avi", "Julia/ecg_folder/Pat4_Vi.avi"]
ecgList = ["Julia/ecg_folder/Pat3_3Trace.txt", "Julia/ecg_folder/Pat4_3Trace.txt"]

n_video_list, ecg_list = dmf.createVidInputsAndTargetEcgs(videoList, ecgList)


print(len(n_video_list))
print(len(n_video_list[0]))

print(len(ecg_list))
print(len(ecg_list[0]))




