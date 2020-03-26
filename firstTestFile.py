import matplotlib.pyplot as plt
import numpy as np
import pprint
import neurolab as nl
import dataManipulationFunctions as imp

#Importing the file
ecg, v = imp.import_td_text_file("TD_+_EKG_data/3_ecg.txt")


#Creating temporary arrays for easier plotting
temp_list_jul = []
temp_list_jul2 = []

for rows in ecg:
    temp_list_jul.append(rows[0])
    temp_list_jul2.append(rows[1])

temp_list_v = []
temp_list_v2 = []

for rows in v:
    temp_list_v.append(rows[0])
    temp_list_v2.append(rows[1])

#Creating figure and plotting

fig = plt.figure(figsize=(18,4))

plt.subplot(121)
plt.plot(temp_list_jul,temp_list_jul2)
plt.subplot(122)
plt.plot(temp_list_v,temp_list_v2)

plt.show()

