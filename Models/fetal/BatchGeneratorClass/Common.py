import os
import csv
from pylab import *
import logging

def getlist(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default,
       split on a comma and strip whitespaces."""
    return [ chunk.strip(chars) for chunk in option.split(sep) ]

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        timeToReport = str(time.time() - startTime_for_tictoc)
        print ("Elapsed time is " + timeToReport + " seconds.")
        return timeToReport
    else:
        print ("Toc: start time not set")
        return None

def arrayToList(arr):
    if type(arr) == type(array([])):
        return arrayToList(arr.tolist())
    elif type(arr) == type([]):
        return [arrayToList(a) for a in arr]
    else:
        return arr

def ensureDir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def readCsv(fileName, separator = ','):
    rows = []
    with open(fileName, 'rb') as f:
        reader = csv.reader(f, delimiter=separator)
        for row in reader:
            print (row)
            rows.append([row.split(separator)])

    return rows

def logDictToInfoStream(d, prefix=""):

    for k in d.keys():
        logging.info(prefix + " " + k + " = " + str(d[k]))

def getMedicalImageBasename(filename):
    if filename.endswith(".nii.gz"):
        return os.path.splitext(os.path.splitext(os.path.split(filename)[1])[0])[0]
    else:
        return os.path.splitext(os.path.split(filename)[1])[0]

def getMedicalImageExtension(filename):
    if filename.endswith(".nii.gz"):
        return ".nii.gz"
    else:
        return os.path.splitext(filename)[1]

def loadFilenamesSingle(filenames):

    listFilenames = []

    with open(filenames) as f:
        listFilenames = [line.rstrip('\n') for line in f if line.rstrip('\n') != ""]

    return listFilenames

def loadFilenames(channels, gtLabels, roiMasks = None, namesForPredictionsPerCase = None):
    """
        Load the filenames files (for training, validation or testing)

        :param channels: list containing the path to the text files containing the path to the channels (this list
                        contains one file per channel).
        :param gtLabels: path of the text file containing the paths to gt label files
        :param roiMasks: [Optional] path of the text file containing the paths to ROI files
        :param roiMasks: [Only used for testing] path of the names that will be assigned to the predictions. If it is
                        not provided, then it is filled with the filenames from the first channel.

        :return allChannelsFilenames, gtFilenames, roiFilenames, namesForPredictionsPerCaseFilenames
                allChannelsFilenames is a list of lists where allChannelsFilenames[0] contains a list of filenames for channel 0 (size: numChannels x numFiles)
                gtFilenames is a list of GT filenames (size: numFiles)
                roiFilenames [Optional: it will be None if argument roiMasks is None]: is a list of filenames (size: numFiles)
                namesForPredictionsPerCaseFilenames [Optional: it will be filled with the basenames from allChannelsFilenames[0] if it s not provided]
                                                    is a list of base names that will be used to generate the output predictions
                                                    when testing (size: numFiles)
    """
    allChannelsFilenames = []
    gtFilenames = None
    roiFilenames = []
    namesForPredictionsPerCaseFilenames = None

    # Load training filenames
    if len(channels) > 0:
        # For every channel
        for filenames in channels:
            # Load the filenames for the given channel
            with open(filenames) as f:
                allChannelsFilenames.append([line.rstrip('\n') for line in f if line.rstrip('\n') != ""])

        for i in range(1, len(channels)):
            assert len(allChannelsFilenames[i]) == len(allChannelsFilenames[0]), "[ERROR] All the channels must contain same number of filenames"

        # Load the GT filenames [Required]
        with open(gtLabels) as f:
            gtFilenames = [line.rstrip('\n') for line in f if line.rstrip('\n') != ""]

        assert len(gtFilenames) == len(allChannelsFilenames[0]), "[ERROR] Number of GT filenames must be the same than image (channels) filenames"

        # Load the ROI filenames [Optional]
        if roiMasks is not None:
            if not(roiMasks == ""):
                with open(roiMasks) as f:
                    roiFilenames = [line.rstrip('\n') for line in f if line.rstrip('\n') != ""]

            assert len(roiFilenames) == len(allChannelsFilenames[0]), "[ERROR] Number of ROI filenames must be the same than image (channels) filenames"

        # Load the Names that will be used for the predictions [Optional]
        if namesForPredictionsPerCase is not None:
            with open(namesForPredictionsPerCase) as f:
                    namesForPredictionsPerCaseFilenames = [line.rstrip('\n') for line in f if line.rstrip('\n') != ""]

            assert len(namesForPredictionsPerCaseFilenames) == len(allChannelsFilenames[0]), "[ERROR] Number of ROI filenames must be the same than image (channels) filenames"
        else:

            with open(channels[0]) as f:
                namesForPredictionsPerCaseFilenames = ([getMedicalImageBasename(line.rstrip('\n')) + ".nii.gz" for line in f if line.rstrip('\n') != ""])

    else:
        raise Exception("channel files are missing")

    return allChannelsFilenames, gtFilenames, roiFilenames, namesForPredictionsPerCaseFilenames
