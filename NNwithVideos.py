import dataManipulationFunctions as dmf


file_folder = "Julia/data/Pat"
ecg_suffix = "_3Trace.txt"
avi_suffix1 = "_Vi.avi"

videoList = ["Julia/data/Pat3_Vi.avi", "Julia/data/Pat4_Vi.avi"]
ecgList = ["Julia/data/Pat3_3Trace.txt", "Julia/data/Pat4_3Trace.txt"]

n_video_list, ecg_list = dmf.createVidInputsAndTargetEcgs(videoList, ecgList)


print(len(n_video_list))
print(len(n_video_list[0]))

print(len(ecg_list))
print(len(ecg_list[0]))




