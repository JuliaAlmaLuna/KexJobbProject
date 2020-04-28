# Här är resultaten från klassificeringen av ekg för videodatan! Under mappen Videomlp/ecg finns alla .txt filer
# kategoriserade.

# För att använda specifika filer, ändra på init file names som i exemplet nedan.

# För att vänta rätt på dem som är upp och ner (enligt index nedan) lägg in if-sats i rimlig metod och om de är rätt
# filnamn gör ecg = -ecg på y-axeln :)


upsidedown_filenames = ['Pat9_3Trace.txt', 'Pat10_3Trace.txt', 'Pat11_3Trace.txt', 'Pat27_3Trace.txt',
                        'Pat33_3Trace.txt', 'Pat35_3Trace.txt', 'Pat40_3Trace.txt', 'Pat42_3Trace.txt']

high_quality_index = [2, 3, 5, 7, 12, 13, 15, 16, 17, 18, 20, 21, 24, 25, 29, 30, 32, 34, 36, 37, 38, 41, 46, 47, 51,
                      52, 57, 58, 62, 63, 64, 66, 69, 70, 72, 73, 74, 75, 76, 78, 80, 84, 85, 87, 88, 89, 90, 94, 95,
                      97, 98, 100]
high_and_medium_quality_index = [2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                 30, 32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 51, 52, 53, 54, 56, 57,
                                 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                                 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 94, 95, 97, 98, 100]


import numpy as np


def init_names(index_list, prefix, suffix, middlix=0):
    names = []

    for index in index_list:
        if middlix == 0:
            name = prefix + str(index) + suffix
        else:
            name = prefix + str(index) + middlix + str(index) + suffix
        names.append(name)
    names = np.array(names)
    return names
