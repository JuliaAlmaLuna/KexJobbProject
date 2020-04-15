import dataManipulationFunctions as dmf
import numpy as np

number_of_samples = 3 # TODO: set correct number of sampled
start_index = 2 # TODO: set the number on the first file (Eg. Pat1_Vi1 set start_index at 1)
step = 2 # TODO: set amount of samples to be saved in one file
video_prefix = "../Mathilda/Videomlp/video_folder/Pat" # TODO: Set right path
video_middlix = "_Vi"
video_suffix = ".avi"
ecg_prefix = "../Mathilda/Videomlp/video_folder/Pat"
ecg_suffix = "_3Trace.txt"


# Start index = number of first sample
def init_names(start_index, nmb_of_samples, prefix, suffix, middlix=0):
    names = []

    for index in range (start_index, nmb_of_samples+start_index):
        if middlix == 0:
            name = prefix + str(index) + suffix
        else:
            name = prefix + str(index) + middlix + str(index) + suffix
        names.append(name)
    names = np.array(names)
    return names


def create_arrays_and_save(videoList, ecgList, nmbr_of_samples, step):
    save_number = 1

    for index in range(0, nmbr_of_samples, step):
        video_list = []
        ecg_list = []

        for jindex in range(index, index+step):
            if jindex < len(videoList) :
                video_name = np.reshape(videoList[jindex], (1,))
                ecg_name = np.reshape(ecgList[jindex], (1,))
                video, ecg = dmf.createVidInputsAndTargetEcgs(video_name, ecg_name)
                video_list.extend(video[0])
                ecg_list.extend(ecg[0])
        video_list = np.array(video_list)
        ecg_list = np.array(ecg_list)
        name_vid = "video_list" + str(save_number)
        name_ecg = "ecg_list" + str(save_number)
        np.save(name_vid, video_list)
        np.save(name_ecg, ecg_list)
        save_number = save_number + 1


videoList = init_names(start_index= start_index, nmb_of_samples=number_of_samples, prefix=video_prefix, suffix=video_suffix, middlix=video_middlix)
ecgList = init_names(start_index, number_of_samples, ecg_prefix, ecg_suffix)
create_arrays_and_save(videoList, ecgList, number_of_samples, step=step)





