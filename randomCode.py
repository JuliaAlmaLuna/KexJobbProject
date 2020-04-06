import dataManipulationFunctions as dmf

temp = dmf.vidToNestedPixelList("Pat7_Vi7.avi")
print(len(temp[0]))
print(len(temp[1]))
print(len(temp))