# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:12:13 2018

@author: jcerrola
"""


from queue import Queue
import random as rnd
import logging
import Common
from Common import loadFilenamesSingle
import threading
from threading import Thread
import nibabel as nib
import numpy as np
from skimage.transform import resize

import sys
sys.path.append("/homes/wt814/IndividualProject/code/")
import volumeTransformations
#imp.reload(BatchGenerator_v2)
from volumeTransformations import *
from volumeNoiseLibrary import *
import time

import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


#import datetime


#import gc
#import itertools
#from Common import tic, toc
#import cv2
#import scipy.ndimage.interpolation
#from scipy.ndimage.interpolation import map_coordinates
#from scipy.ndimage.filters import gaussian_filter

#from skimage.draw import line_aa
#import scipy.misc
#from scipy import ndimage



""" #################################################################################################################
                                               BatchGenerator Class (deffault class))
    ################################################################################################################# """

class BatchGenerator:
    """
        This class implements an interface for batch generators. Any batch generator deriving from this class
        should implement its own data augmentation strategy.

        In this class, we suppose that in one epoch we process several subepochs, every one formed by multiple batches
        (every batch is processed independently by the CNN). In every subepoch, new volumes are read from disk
        and used to sample segments for several batches.

        The method generateBatches( ) runs a thread that will call the abstract method generateBatchesForOneEpoch( ) as many
        times as confTrain['numEpochs'] indicates. The method generateBatchesForOneEpoch( ) runs several subepochs,
        reading new volumes in every subepoch. In every subepoch, this method generates several batches composed
        of different random segments extracted from the loaded volumes.
        The batches will be all queued in self.queue using self.queue.put(batch).

        Note that generateBatchesForOneEpoch( ) should read data only once, produce all the necessary batches for the
        given subepochs, insert them at the end of the queue and close the data files.

        The idea is that all the batches (corresponding to all the epochs) are stored (in order) in the same queue.

        The methods that must be implemented by any class inheriting BatchGenerator are:
        - generateBatchesForOneEpoch(self):
            This method will read some data, produce all the necessary batches for the subepochs, insert them at
            the end of the queue and close the data files.

        The class using any BatchGenerator will proceed as follows:

        ===================== Training Loop =====================

            batchGen = BatchGenerator(confTrain)
            batchGen.generateBatches()

            for e in range(0, confTrain['numEpochs']):
                for se in range (0, confTrain['numSubepochs']):
                    batchesPerSubepoch = confTrain['numTrainingSegmentsLoadedOnGpuPerSubep'] // confTrain['batchSizeTraining']

                    for bps in range(0, batchesPerSubepoch)
                        batch = batchGen.getBatch()
                        updateCNN(batch)

            assert (batchGen.emptyQueue()), "The training loop finished before the queue is empty".

        ==========================================

         ====== Pseudocode of the data generation loop =======
         for epoch in range(0, numEpochs):
           for subepoch in range (0, numSubepochs):
               read 'numOfCasesLoadedPerSubepoch' volumes
               batchesPerSubepoch = numberTrainingSegmentsLoadedOnGpuPerSubep // batchSizeTraining
               for batch in range(0, batchesPerSubepoch):
                   data = extractSegments(batchSizeTraining)
                   queue.put(data)

         ====== Pseudocode of the training loop (running in parallel with data generation) =======
         for epoch in range(0, numEpochs):
           for subepoch in range (0, numSubepochs):
               batchesPerSubepoch = numberTrainingSegmentsLoadedOnGpuPerSubep // batchSizeTraining
               for batch in range(0, batchesPerSubepoch):
                   data = queue.get()
                   updateCNNWeights(data)
    """
    def __init__(self, confTrain, maxQueueSize = 15, infiniteLoop = False):
        """
            Creates a batch generator.

            :param confTrain: a configuration dictionary containing the necessary training parameters
            :param maxQueueSize: maximum number of batches that will be inserted in the queue at the same time.
                           If this number is achieved, the batch generator will wait until one batch is
                           consumed to generate a new one.

                           The number of elements in the queue can be monitored using getNumBatchesInQueue. The queue
                           should never be empty so that the GPU is never idle. Note that the bigger maxQueueSize,
                           the more RAM will the program consume to store the batches in memory. You should find
                           a good balance between RAM consumption and keeping the GPU processing batches all the time.

            :param infiniteLoop: if it's True, then epochs are ignored and the batch generator itearates until it's
                                killed by the system.

            :return: self.queue.empty()
        """
        self.confTrain = confTrain
        #print "Creating Queue with size: " + str(maxQueueSize)
        self.queue = Queue(maxsize=maxQueueSize)
        self.queueFileNames = Queue(maxsize=maxQueueSize)
        self.currentEpoch = 0
        self.infiniteLoop = infiniteLoop

        self.rndSequence = rnd.Random()
        self.rndSequence.seed(1)
        self.keepOnRunning = True
        self.listCurrentFiles = []
        self.currentFile = []

        # ID used when printing LOG messages.
        self.id = "[BATCHGEN]"

    def emptyQueue(self):
        """
            Checks if the batch queue is empty or not.

            :return: self.queue.empty()
        """

        return self.queue.empty()

    def _generateBatches(self):
        """
            Private function that generates as many batches as epochs were specified
        """
        self.currentEpoch = 0

        while (self.infiniteLoop or (self.currentEpoch < self.confTrain['numEpochs'])) and self.keepOnRunning:
            self.generateBatchesForOneEpoch()
            self.currentEpoch += 1

        logging.info(self.id + " The batch generation process finished. Elements still in the queue before finishing: %s. The queue will be destroyed." % str(self.getNumBatchesInQueue()))

    def generateBatches(self):
        """
            This public interface lunches a thread that will start generating batches for the epochs/subepochs specified
            in the configuration file, and storing them in the self.queue.
            To extract these batches, use self.getBatch()
        """
        worker = Thread(target=self._generateBatches, args=())
        worker.setDaemon(False)
        worker.start()


    def getBatch(self):
        """
            It returns a batch and removes it from the front of the queue

            :return: a batch from the queue
        """
        print('getting batch')
        batch = self.queue.get()
        self.currentFile = self.queueFileNames.get()
        self.queue.task_done()
        return batch

    def getNumBatchesInQueue(self):
        """
            It returns the number of batches currently in the queue, which are ready to be processed

            :return: number of batches in the queue
        """
        return self.queue.qsize()

    def finish(self, delay = .5):
        """
            It will interrupt the batch generation process, even if there are still batches to be created.
            If the batch generator is currently producing a batch, then it will stop after finishing that batch.
            The queue will be destroyed together with the process.

            Note: if there is a process waiting for a batch in the queue, the behaviour is unpredictable. The process
                  that is waiting may wait forever.

            :param delay: the delay will be used after getting an element from the queue. If your batch generation
                        process is too time consuming, you should increase the delay to guarantee that once
                        the queue is empty, the batch generation process is done.
        """
        self.keepOnRunning = False
        logging.info(self.id + " Stopping batch generator. Cleaning the queue which currently contains %s elements ..." % str(self.queue.qsize()))

        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()

            time.sleep(delay)
            if not self.queue.empty():
                logging.info(self.id + " Still %s elements in the queue ..." % str(self.queue.qsize()))

        logging.info(self.id + " Done.")

    def generateBatchesForOneEpoch(self):
        """
            This abstract function must be implemented. It must generate all the batches corresponding to one epoch
            (one epoch is divided in subepochs where different data samples are read from disk, and every subepoch is
            composed by several batches, where every batch includes many segments.)

            Every batch must be queued using self.queue.put(batch) and encoded using lasagne-compatible format,
            i.e: a 5D tensor with size (batch_size, num_input_channels, input_depth, input_rows, input_columns)

        """
        raise NotImplementedError('users must define "generateBatches" to use this base class')


""" #################################################################################################################
                                      Additional functions
    ################################################################################################################# """

def printMessageVerb(FLAGverbosity, messageOut):
    """
        Function to display status messages if FLAGverbosity == 1
    """
    if FLAGverbosity == 1:
        print(messageOut)

def resizeData(array, imageSize, FLAGpreserveIntValues = True):
    """
        Function to resize iamges (numpy arrays)
             - array: input numpy.ndarray
             - imageSize: new size of the image
             - FLAGpreserveIntValues: Flag to preserve the original integer values of the input array (used for intensity masks and labels)
    """
    array = array.astype(float)
    arrayResize = (array - array.min()) / (array.max()-array.min()) # values need to be between 0 and 1 to use the resize function.
    arrayResize = resize(arrayResize,imageSize, preserve_range=True, mode = 'reflect')
    arrayResize = (arrayResize*(array.max()-array.min()))+array.min() # restore the original range of values.
    # If needed, restore the original set of integers values (used for masks and labels)
    if FLAGpreserveIntValues:
        arrayResize = np.around(arrayResize)
        #arrayResize[arrayResize < array.min()] = array.min()
        #arrayResize[arrayResize > array.max()] = array.max()
        rangeValues = np.unique(array)
        interLabelsTh = [ (x + rangeValues[i - 1])/2. for i, x in enumerate(rangeValues)][1:]
        LOWlimit = [arrayResize.min()-1] + interLabelsTh
        UPlimit = interLabelsTh + [arrayResize.max()+1]
        for label_i in range(0,len(rangeValues)):
            arrayResize[np.multiply(UPlimit[label_i] > arrayResize, arrayResize >= LOWlimit[label_i])] = rangeValues[label_i]
    return arrayResize



def intensityNormalization(array,arrayMask=None,intensityNormalizationMode='range',intNormParam1 = 0, intNormParam2 =1):
    '''
         Intensity normalization function
             - array: input numpy.ndarray
             - arrayMask=[] : numpy.ndarray defining the region of interest considered when normalizing the intensity
             - intensityNormalizationMode = 'range', 'meanStd' Normalization mode
             - intNormParam1: 1st parameter ('range': minimum value | 'meanStd': mean value)
             - intNormParam2: 2ndt parameter ('range': maximum value | 'meanStd': std value)
    '''
    if arrayMask==None: arrayMask = np.ones(array.shape)
    inMaskIDs = np.where(arrayMask>0)
    if (intensityNormalizationMode=='range'):       # normalization to the range [intNormParam1 intNormParam2]
         arrayParam1,arrayParam2 = (array[inMaskIDs].min(),array[inMaskIDs].max())
         array[inMaskIDs] = (((array[inMaskIDs] - arrayParam1)/(arrayParam2-arrayParam1))*(intNormParam2-intNormParam1)) + intNormParam1
    elif (intensityNormalizationMode=='meanStd'):   # normalization to mean = intNormParam1 and std = intNormParam2.
         arrayParam1,arrayParam2 = (array[inMaskIDs].mean(),array[inMaskIDs].std())
         array[inMaskIDs] = (((array[inMaskIDs] - arrayParam1) / arrayParam2)*intNormParam2)+intNormParam1
    return array



def preprocessIntensityData(array, FLAGresizeImages = False, imageSize = [], FLAGpreserveIntValues = True, arrayMask=None, intensityNormalizationMode=None,intNormParam1 = 0, intNormParam2 =1):
    '''
         Preprocessing input data
             - array: input numpy.ndarray
             - FLAGresizeImages: FLAG to resize the input image.
             - imageSize: new size of the image
             - FLAGpreserveIntValues: Flag to preserve the original integer values of the input array (used for intensity masks and labels)
             - arrayMask=[] : numpy.ndarray defining the region of interest considered when normalizing the intensity
             - intensityNormalizationMode = 'range', 'meanStd' Normalization mode
             - intNormParam1: 1st parameter ('range': minimum value | 'meanStd': mean value)
             - intNormParam2: 2ndt parameter ('range': maximum value | 'meanStd': std value)
    '''
    if FLAGresizeImages:
        array = resizeData(array, imageSize, FLAGpreserveIntValues)
    if not(intensityNormalizationMode is None):
        array = intensityNormalization(array,arrayMask,intensityNormalizationMode,intNormParam1,intNormParam2)
    return array


def preprocessIntensityData_2D(array, FLAGresizeImages = False, imageSize = [], FLAGpreserveIntValues = True, intensityNormalizationMode=None, intNormParam1 = 0, intNormParam2 =1):
    '''
         Preprocessing 2D input data
             - array: input numpy.ndarray
             - FLAGresizeImages: FLAG to resize the input image.
             - imageSize: new size of the image
             - FLAGpreserveIntValues: Flag to preserve the original integer values of the input array (used for intensity masks and labels)
             - intensityNormalizationMode = 'range', 'meanStd' Normalization mode
             - intNormParam1: 1st parameter ('range': minimum value | 'meanStd': mean value)
             - intNormParam2: 2ndt parameter ('range': maximum value | 'meanStd': std value)
    '''

    if FLAGresizeImages:
        array = resizeData(array, imageSize, FLAGpreserveIntValues)
    if not(intensityNormalizationMode is None):
        array = intensityNormalization_2D(array,intensityNormalizationMode,intNormParam1,intNormParam2)
    return array




def setOutMaskValue(array, arrayMask, voxelValue = 0):
    '''
         Set the background voxels (outside of the input mask) to a predefined value.
             - array: input numpy.ndarray
             - arrayMask=[] : numpy.ndarray defining the region of interest considered when normalizing the intensity
             - voxelValue = value of the background voxels
     '''
    if not(arrayMask is None):
        array[arrayMask == 0] = voxelValue
    return array


def normalizeLabels(array):
    '''
         Normalize or discretize the labels (e.g., if is a ground truth segmentation volume uses 255 as foreground value, the function map it to 1.
         Similarly, if the function uses any other random set label values, [0, 12, 56, 987], the function maps these values to [0, 1, 2, 3].
             - array: input numpy.ndarray
    '''
    arrayOut = array.copy()
    labelList = np.unique(array)
    for li in range(0,len(labelList)):
        arrayOut[array==labelList[li]]=li
    return arrayOut



""" #################################################################################################################
                                      BatchGeneratorBinVolDataAug Class
    ################################################################################################################# """

class BatchGeneratorBinVolDataAug(BatchGenerator):
    """
        Simple batch generator that takes 3D binary volumes as input,
        and generates batches.
        The class also includes data augmentation (anisotropic and isotropic)
    """

    def __init__(self, confTrain, mode = 'training',infiniteLoop = False, maxQueueSize = 15):
        self.object__ = """
            Initialize a 3D batch generator with a confTrain object.
            - mode (training/validation): Modes can be 'training' or 'validation'. Depending on the mode, the batch generator
            will load the training files (channelsTraining, gtLabelsTraining, roiMasksTraining)
            or the validation files (channelsValidation, gtLabelsValidation, roiMasksValidation) from
            the confTrain object.
        """
        BatchGenerator.__init__(self, confTrain, infiniteLoop = infiniteLoop, maxQueueSize = maxQueueSize)

        # List of currently loaded channel images (size:numOfCasesLoadedPerSubepochTraining or numOfCasesLoadedPerSubepochValidation X channels)
        self.currentChannelImages = []
        # List of currently loaded ROI images (size:numOfCasesLoadedPerSubepochTraining or numOfCasesLoadedPerSubepochValidation)
        self.currentRois = []
        # List of currently loaded GT images (size:numOfCasesLoadedPerSubepochTraining or numOfCasesLoadedPerSubepochValidation)
        self.currentGt = []
        # Names of the cases in the queue
        self.currentFile = []

        # queque of scaling factors (we are monitoring the corresponding scaling factor for each batch)
        self.queueScalingFactors = Queue(maxsize=maxQueueSize)

        self.batch = []         # this will be a numpy array storing the current batch
        self.gt = []            #this will be a numpy array storing the current batch

        self.indexCurrentImages = [];
        self.IDsCases = []      # order in which the images are used.

        self.FLAGverbose = (self.confTrain['FLAGverbose']) if ('FLAGverbose' in self.confTrain) else (0) # Verbosity flag.

        # Mode: can be training or validation
        self.mode = mode
        if mode == 'training':
            self.id = '[WHOLEVOL BATCHGEN TRAIN]'
            self.batchSize = (self.confTrain['batchSizeTraining']) if ('batchSizeTraining' in self.confTrain) else (1)
            self.batchesPerEpoch = 1 # This parameter will be updated latter
        elif mode == 'validation':
            self.id = '[WHOLEVOL BATCHGEN VAL]'
            self.batchSize = 1
            self.batchesPerEpoch = 1 # This parameter will be updated latter
        printMessageVerb(self.FLAGverbose, '-->> Initializing BatchGeneratorBinVolDataAug '+self.id)

        # Image preprocessing options
        # .....................................................................
        self.FLAGresizeImages = self.confTrain['FLAGresizeImages'] # list of flags (one per channel)
        self.imageSize,self.gtSize = (self.confTrain['imageSize'], self.confTrain['gtSize']) if self.FLAGresizeImages else (0,0)
        self.FLAGintensityNormalization = (self.confTrain['FLAGintensityNormalization']) if ('FLAGintensityNormalization' in self.confTrain) else (0)
        self.intensityNormalizationMode,self.intNormParam1,self.intNormParam2 = (self.confTrain['intensityNormalizationMode'], self.confTrain['intNormParam1'], self.confTrain['intNormParam2']) if ('intensityNormalizationMode' in self.confTrain) else ('none',0,0)
        self.isChannelBinary = self.confTrain['isChannelBinary']
        self.FLAGsetBkgrnd,self.bkgrndLabel = (self.confTrain['FLAGsetBkgrnd'], self.confTrain['bkgrndLabel']) if ('FLAGsetBkgrnd' in self.confTrain) else (False,0)

        # Data augmentation options / params
        # .....................................................................
        self.dataAugmentationRate = self.confTrain['dataAugmentationRate'] if (('dataAugmentationRate' in self.confTrain) and (mode == 'training')) else 0.0
        if (self.dataAugmentationRate > 0.0):
            self.translationRangeX = self.confTrain['translationRangeX']
            self.translationRangeY = self.confTrain['translationRangeY']
            self.translationRangeZ = self.confTrain['translationRangeZ']
            self.rotationRangeX = self.confTrain['rotationRangeX']
            self.rotationRangeY = self.confTrain['rotationRangeY']
            self.rotationRangeZ = self.confTrain['rotationRangeZ']
            self.FLAGholesNoise = (self.confTrain['FLAGholesNoise']) if ('FLAGholesNoise' in self.confTrain) else (0)
            self.holesRatio = self.confTrain['holesRatio']
            self.holesRadiiRange = self.confTrain['holesRadiiRange']
            self.FLAGsaltpepperNoise = (self.confTrain['FLAGsaltpepperNoise']) if ('FLAGsaltpepperNoise' in self.confTrain) else (0)
            self.ratioSaltPepperNoise = self.confTrain['ratioSaltPepperNoise']
            self.saltPepperNoiseSizeRange = self.confTrain['saltPepperNoiseSizeRange']

            textOut = '-->> data augmentation ON: trans x[%3.1f, %3.1f] y[%3.1f, %3.1f] z[%3.1f, %3.1f] - rot x[%3.1f, %3.1f] y[%3.1f, %3.1f] z[%3.1f, %3.1f]' % (self.translationRangeX[0], self.translationRangeX[1], self.translationRangeY[0], self.translationRangeY[1], self.translationRangeZ[0], self.translationRangeZ[1], self.rotationRangeX[0], self.rotationRangeX[1], self.rotationRangeY[0], self.rotationRangeY[1], self.rotationRangeZ[0], self.rotationRangeZ[1])
            printMessageVerb(self.FLAGverbose, textOut)

            self.isotropicScaleFLAG = (self.confTrain['isotropicScaleFLAG']) if ('isotropicScaleFLAG' in self.confTrain) else (1)
            if self.isotropicScaleFLAG:
                self.isotropicScaleRange = self.confTrain['isotropicScaleRange']
                textOut = '-->> isotropic scaling [%3.1f, %3.1f]' % (self.isotropicScaleRange[0], self.isotropicScaleRange[1])
            else:
                self.anisoScaleRangeX = self.confTrain['anisoScaleRangeX']
                self.anisoScaleRangeY = self.confTrain['anisoScaleRangeY']
                self.anisoScaleRangeZ = self.confTrain['anisoScaleRangeZ']
                textOut = '-->> anisotropic scaling x[%3.1f, %3.1f], y[%3.1f, %3.1f], z[%3.1f, %3.1f]' % (self.anisoScaleRangeX[0], self.anisoScaleRangeX[1], self.anisoScaleRangeY[0], self.anisoScaleRangeY[1], self.anisoScaleRangeZ[0], self.anisoScaleRangeZ[1])
            printMessageVerb(self.FLAGverbose, textOut)
        else:
            printMessageVerb(self.FLAGverbose, '-->> data augmentation OFF')


#        # Read files' paths
#        # .....................................................................
        if mode == 'training':
            logging.info('-- Initializing TRAINING Batch generator')
            self.numChannels = len(self.confTrain['channelsTraining']) # Number of channels per image
            self.numOfCasesLoadedPerEpoch = self.confTrain['numOfCasesLoadedPerEpochWhenTrainingWholeVolume']
            self.loadFilenames(self.confTrain['channelsTraining'], self.confTrain['gtLabelsTraining'],
                          self.confTrain['roiMasksTraining'] if ('roiMasksTraining' in self.confTrain) else None)
            # if <=0, load all the cases per epoch
            if (self.numOfCasesLoadedPerEpoch <= 0 or self.numOfCasesLoadedPerEpoch>self.numFiles): self.numOfCasesLoadedPerEpoch = self.numFiles

        elif mode == 'validation':
            logging.info('-- Initializing VALIDATION Batch generator')
            self.numChannels = len(self.confTrain['channelsValidation'])
            #self.numOfCasesLoadedPerEpoch = self.confTrain['numOfCasesLoadedPerEpochWhenValidatingWholeVolume']
            self.loadFilenames(self.confTrain['channelsValidation'], self.confTrain['gtLabelsValidation'],
                               self.confTrain['roiMasksValidation'] if ('roiMasksValidation' in self.confTrain) else None)
            self.numOfCasesLoadedPerEpoch =  len(self.allChannelsFilenames[0]) # load all the cases per epoch.

        else:
            raise Exception('ERROR: Batch generator mode is not valid. Valid options are training or validation')

        textOut = '-->> numChannels: '+str(self.numChannels) + ' - cases loaded per ecpoch: '+str(self.numOfCasesLoadedPerEpoch)
        printMessageVerb(self.FLAGverbose, textOut)


        # Check if, given the number of cases loaded per subepoch and the total number of samples, we
        # will need to load new data in every subepoch or not.

        self.loadNewFilesEveryEpoch = len(self.allChannelsFilenames[0]) > self.numOfCasesLoadedPerEpoch
        logging.info(self.id + " Loading new files every subepoch: " + str(self.loadNewFilesEveryEpoch))
        if self.loadNewFilesEveryEpoch: printMessageVerb(self.FLAGverbose, '-->> need to load new files every epoch')


    def getBatchAndScalingFactor(self):
        """
            It returns a batch and removes it from the front of the queue

            :return: a batch from the queue
        """
        batch = self.queue.get()
        self.currentFile = self.queueFileNames.get()
        self.currentScalingFactor = self.queueScalingFactors.get()
        self.queue.task_done()
        self.queueScalingFactors.task_done()
        self.queueFileNames.task_done()
        return batch


    def loadFilenames(self, channels, gtLabels, roiMasks = None):
        '''
            Load the filenames that will be used to generate the batches.

            :param channels: list containing the path to the text files containing the path to the channels (this list
                            contains one file per channel).
            :param gtLabels: path of the text file containing the paths to gt label files
            :param roiMasks: [Optional] path of the text file containing the paths to ROI files

        '''
        self.allChannelsFilenames, self.gtFilenames, self.roiFilenames, _ = Common.loadFilenames(channels, gtLabels, roiMasks)
        self.numFiles = len(self.gtFilenames)


    def unloadFiles(self):
        '''
            Unload the filenames if need to load new images each epoch.
        '''
        logging.debug(self.id + " Unloading Files")
        for image in self.currentChannelImages:
            for channelImage in image:
                del channelImage

        for image in self.currentGt:
            del image

        for image in self.currentRois:
            del image

        del self.currentChannelImages
        del self.currentGt
        del self.currentRois

        self.currentChannelImages = []
        self.currentGt = []
        self.currentRois = []


    def generateTransformParams(self):
        """
            Generate random transformation parameters for data augmentation.
        """
        tX,tY,tZ = (int(np.random.uniform(self.translationRangeX[0],self.translationRangeX[1],1)[0]),
                    int(np.random.uniform(self.translationRangeY[0],self.translationRangeY[1],1)[0]),
                    int(np.random.uniform(self.translationRangeZ[0],self.translationRangeZ[1],1)[0]))

        if self.isotropicScaleFLAG:
            scaleFactor = np.random.uniform(self.isotropicScaleRange[0],self.isotropicScaleRange[1],1)[0]
        else:
            anisoScaleX = np.random.uniform(self.anisoScaleRangeX[0],self.anisoScaleRangeX[1],1)[0]
            anisoScaleY = np.random.uniform(self.anisoScaleRangeY[0],self.anisoScaleRangeY[1],1)[0]
            anisoScaleZ = np.random.uniform(self.anisoScaleRangeZ[0],self.anisoScaleRangeZ[1],1)[0]
            scaleFactor = (anisoScaleX, anisoScaleY, anisoScaleZ)

        rX,rY,rZ = (np.random.uniform(self.rotationRangeX[0],self.rotationRangeX[1],1)[0],
                    np.random.uniform(self.rotationRangeY[0],self.rotationRangeY[1],1)[0],
                    np.random.uniform(self.rotationRangeZ[0],self.rotationRangeZ[1],1)[0])
        return tX,tY,tZ,scaleFactor,rX,rY,rZ



    def generateSingleBatch(self, numBatch):
        """
            It supposes that the images are already loaded in self.currentChannelImages / self.currentRois / self.currentGt

            :return: It returns the data and ground truth of a complete batch as data, gt. These structures are theano-compatible with shape:
                        np.ndarray(shape=(self.confTrain['batchSizeTraining'], self.numChannels, dim_1, dim_2, dim_3), dtype=np.float32)
        """

        if not(self.currentChannelImages == []): # If list of pre-loaded volumes is not empty...

            # define a default scaling factor. The scaling factor is provided as output value.
            if self.isotropicScaleFLAG:
                isoScale = 1.0
            else:
                isoScale = (1., 1., 1.)


            # Data augmentation
            # .....................................................................................
            if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):

                # generate random transformation parameters
                tX,tY,tZ,isoScale,rX,rY,rZ = self.generateTransformParams()

                # MASK (apply data augmentation to or auxRoi)
                if not(self.currentRois == []):
                    auxRoi = self.currentRois[numBatch]
                    rangeRoiValues = np.unique(auxRoi)
                    #auxRoi = applyTransformToVolume(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                    auxRoi = applyTransformToVolumeAnisoScale(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                    auxRoi = restoreIntValues (auxRoi, rangeRoiValues)
                    auxRoi = setOutMaskValue(auxRoi, auxRoi, voxelValue = self.bkgrndLabel)
                else:
                    auxRoi = np.ones((self.currentChannelImages[numBatch][0].shape))

               # GT
                auxGt = self.currentGt[numBatch]
                rangeGTValues = np.unique(auxGt)
                #auxGt = applyTransformToVolume(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                auxGt = applyTransformToVolumeAnisoScale(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                auxGt = restoreIntValues (auxGt, rangeGTValues)
                auxGt = setOutMaskValue(auxGt, auxRoi, voxelValue = self.bkgrndLabel)
                if auxGt.shape != self.gtSize:
                    auxGt = resizeData(auxGt, self.gtSize, preserveIntValues = True)
                self.gt[numBatch,0,:,:,:] = auxGt.astype(np.int16)


                # IMG channels
                for channel in range(self.numChannels):

                    auxImg = self.currentChannelImages[numBatch][channel].copy()

                    if self.isChannelBinary[channel] == 1: #if the channel is binary / or is like the GT (with integer discrete values)
                        rangeIMGValues = np.unique(auxImg)
                        #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                        auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                        auxImg = restoreIntValues (auxImg, rangeIMGValues)
                        auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)
                    else:
                        #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                        auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                        auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)

                    # Add noise if needed
                    if self.FLAGholesNoise == 1:
                        foreIDs = np.nonzero(auxGt)
                        volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.holesRatio, radiiRange = self.holesRadiiRange)
                        seedsIDs = np.nonzero(volSeeds)
                        auxImg[seedsIDs] = self.bkgrndLabel

                    if self.FLAGsaltpepperNoise == 1:
                        foreIDs = np.nonzero(np.ones(auxGt.shape))
                        volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                        saltIDs = np.nonzero(volSeeds)
                        volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                        pepperIDs = np.nonzero(volSeeds)
                        auxImg[saltIDs] = self.intNormParam1[channel]
                        auxImg[pepperIDs] = self.intNormParam2[channel]

                    self.batch[numBatch, channel, :, :, :] = auxImg.copy()


            # No data augmentation
            # .....................................................................................
            else:
                # GT
                self.gt[numBatch, 0, :, :, :] = self.currentGt[numBatch].astype(np.int16)

                # IMG channels
                for channel in range(self.numChannels):
                    if self.loadNewFilesEveryEpoch:
                        self.batch[numBatch, channel, :, :, :] = self.currentChannelImages[numBatch][channel].copy()
                    else:
                        self.batch[numBatch, channel, :, :, :] = self.currentChannelImages[numBatch][channel]

            return isoScale

        else:
            raise Exception(self.id + " No images loaded in self.currentVolumes." )



    def generateSingleBatchV2(self, IDs_i):
        """
            It supposes that the images are already loaded in self.currentChannelImages / self.currentRois / self.currentGt

            :return: It returns the data and ground truth of a complete batch as data, gt. These structures are theano-compatible with shape:
                        np.ndarray(shape=(self.confTrain['batchSizeTraining'], self.numChannels, dim_1, dim_2, dim_3), dtype=np.float32)

            IDs_i: IDs of the cases to use in the batch.
        """

        if not(self.currentChannelImages == []): # If list of pre-loaded volumes is not empty...

            # define a default scaling factor. The scaling factor is provided as output value.
            if self.isotropicScaleFLAG:
                isoScale_ALL = np.ones([len(IDs_i)])
            else:
                isoScale_ALL = np.ones([len(IDs_i),3])

            # Batch-size Loop
            for id_i in range(0,len(IDs_i)):

                # Data augmentation
                # .....................................................................................
                if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):

                    # generate random transformation parameters
                    tX,tY,tZ,isoScale,rX,rY,rZ = self.generateTransformParams()
                    if self.isotropicScaleFLAG:
                        isoScale_ALL[id_i] = isoScale
                    else:
                        isoScale_ALL[id_i,:] = np.array(isoScale)

                    # MASK (apply data augmentation to or auxRoi)
                    if not(self.currentRois == []):
                        auxRoi = self.currentRois[IDs_i[id_i]]
                        rangeRoiValues = np.unique(auxRoi)
                        #auxRoi = applyTransformToVolume(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                        auxRoi = applyTransformToVolumeAnisoScale(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                        auxRoi = restoreIntValues (auxRoi, rangeRoiValues)
                        auxRoi = setOutMaskValue(auxRoi, auxRoi, voxelValue = self.bkgrndLabel)
                    else:
                        auxRoi = np.ones((self.currentChannelImages[IDs_i[id_i]][0].shape))

                   # GT
                    auxGt = self.currentGt[IDs_i[id_i]]
                    rangeGTValues = np.unique(auxGt)
                    #auxGt = applyTransformToVolume(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                    auxGt = applyTransformToVolumeAnisoScale(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                    auxGt = restoreIntValues (auxGt, rangeGTValues)
                    auxGt = setOutMaskValue(auxGt, auxRoi, voxelValue = self.bkgrndLabel)
                    if auxGt.shape != self.gtSize:
                        auxGt = resizeData(auxGt, self.gtSize, preserveIntValues = True)
                    self.gt[id_i,0,:,:,:] = auxGt.astype(np.int16)


                    # IMG channels
                    for channel in range(self.numChannels):

                        auxImg = self.currentChannelImages[IDs_i[id_i]][channel].copy()

                        if self.isChannelBinary[channel] == 1: #if the channel is binary / or is like the GT (with integer discrete values)
                            rangeIMGValues = np.unique(auxImg)
                            #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                            auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                            auxImg = restoreIntValues (auxImg, rangeIMGValues)
                            auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)
                        else:
                            #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                            auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                            auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)

                        # Add noise if needed
                        if self.FLAGholesNoise == 1:
                            foreIDs = np.nonzero(auxGt)
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.holesRatio, radiiRange = self.holesRadiiRange)
                            seedsIDs = np.nonzero(volSeeds)
                            auxImg[seedsIDs] = self.bkgrndLabel

                        if self.FLAGsaltpepperNoise == 1:
                            foreIDs = np.nonzero(np.ones(auxGt.shape))
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                            saltIDs = np.nonzero(volSeeds)
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                            pepperIDs = np.nonzero(volSeeds)
                            auxImg[saltIDs] = self.intNormParam1[channel]
                            auxImg[pepperIDs] = self.intNormParam2[channel]

                        self.batch[id_i, channel, :, :, :] = auxImg.copy()


                # No data augmentation
                # .....................................................................................
                else:
                    # GT
                    self.gt[id_i, 0, :, :, :] = self.currentGt[IDs_i[id_i]].astype(np.int16)

                    # IMG channels
                    for channel in range(self.numChannels):
                        if self.loadNewFilesEveryEpoch:
                            self.batch[id_i, channel, :, :, :] = self.currentChannelImages[IDs_i[id_i]][channel].copy()
                        else:
                            self.batch[id_i, channel, :, :, :] = self.currentChannelImages[IDs_i[id_i]][channel]

            return isoScale_ALL

        else:
            raise Exception(self.id + " No images loaded in self.currentVolumes." )





    def generateBatchesForOneEpoch(self):
        """
            Reachable from 'generateBatches' method.
            generateBatches() -> _generateBatches -> generateBatchesForOneEpoch
        """

        #printMessageVerb(self.FLAGverbose, '-->> generating batches ...')


        # Load the files if needed
        if (self.currentEpoch == 0) or ((self.currentEpoch > 0) and self.loadNewFilesEveryEpoch):

            # Reset the list of images
            self.currentChannelImages = []
            self.currentGt = []
            self.currentRois = []


            # Choose the random images that will be sampled in this epoch
            # (*) ToDo -> make sure that new samples are used every epoch.
            self.indexCurrentImages = np.array(self.rndSequence.sample(range(0,self.numFiles), self.numOfCasesLoadedPerEpoch)) #it needs to be a numpy array so we can extract multiple elements with a list (the elements defined by IDsCases, which will be shuffled each epoch)
            self.IDsCases = list(range(0,self.numOfCasesLoadedPerEpoch)) #IDs to the cases in self.indexCurrentImages


            printMessageVerb(self.FLAGverbose, "Loading %d images for epoch %d" % (len(self.indexCurrentImages), self.currentEpoch))
            logging.debug(self.id + " Loading images number : %s" % self.indexCurrentImages )

            self.batchesPerEpoch = int(np.floor(self.numFiles / self.batchSize)) # number of batches per epoch

            #print(self.id + " Loading images number : %s" % indexCurrentImages )
            # Load the images for the epoch
            #i = 0
            self.listCurrentFiles = [] # reset the list of files loaded per epoch.
            for realImageIndex in self.indexCurrentImages:

                 loadedImageChannels = [] #list to store all the channels of the current image.
                 printMessageVerb(self.FLAGverbose, '-->> loading case %d'% (realImageIndex))
                 self.listCurrentFiles.append(self.allChannelsFilenames[0][realImageIndex])   #List of filenames in the order

                 # Load ROI if exists
                 if ('roiMasksTraining' in self.confTrain):
                     roi = nib.load(self.roiFilenames[realImageIndex]).get_data()
                     roi = preprocessIntensityData(roi, FLAGresizeImages=self.FLAGresizeImages, imageSize=self.imageSize, FLAGpreserveIntValues = True, arrayMask=[], intensityNormalizationMode=None)
                     self.currentRois.append(roi)
                 else:
                     roi = None

                 # Imgs channels ----------------------------------------------
                 # (*) ToDo --> incorporate the masks in the image normalization stage.!!
                 for channel in range(0, self.numChannels):
                      # Load the corresponding image for the corresponding channel and append it to the list of channels
                      # for the current imageIndex
                      # loadedImageChannels.append(nib.load(self.allChannelsFilenames[channel][realImageIndex]).get_data())

                      # Load, preprocess, and normalize the channel.
                      dataIn = nib.load(self.allChannelsFilenames[channel][realImageIndex]).get_data()
                      if self.isChannelBinary[channel] == 0: # not binary input
                          dataIn = preprocessIntensityData(dataIn,FLAGresizeImages=self.FLAGresizeImages,
                                    imageSize=self.imageSize,
                                    FLAGpreserveIntValues = False, # False
                                    arrayMask=[],
                                    intensityNormalizationMode=self.intensityNormalizationMode[channel],
                                    intNormParam1=self.intNormParam1[channel],
                                    intNormParam2=self.intNormParam2[channel]
                                    )
                          if self.FLAGsetBkgrnd == True:
                              dataIn = setOutMaskValue(dataIn, roi, voxelValue = self.bkgrndLabel)

                      elif self.isChannelBinary[channel] == 1: # binary input --> treat as gt.
                          dataIn = preprocessIntensityData(dataIn, FLAGresizeImages=self.FLAGresizeImages,
                                    imageSize=self.imageSize,
                                    FLAGpreserveIntValues = True,
                                    arrayMask=[],
                                    intensityNormalizationMode = None)
                          dataIn = normalizeLabels(dataIn)

                      # Add the image to the queue of channels
                      loadedImageChannels.append(dataIn)

                      # Check that all the channels have the same dimensions
                      if channel > 0:
                          assert loadedImageChannels[channel].shape == loadedImageChannels[0].shape, self.id + " Data size incompatibility when loading image channels for volume %s" % self.allChannelsFilenames[channel][realImageIndex]

                 # Append all the channels of the image to the list
                 self.currentChannelImages.append(loadedImageChannels)

                 # GT channel ----------------------------------------------
                 gt = nib.load(self.gtFilenames[realImageIndex]).get_data()
                 gt = preprocessIntensityData(gt, FLAGresizeImages=self.FLAGresizeImages, imageSize=self.imageSize, FLAGpreserveIntValues = True, arrayMask=[], intensityNormalizationMode=None)
                 gt = normalizeLabels(gt)
                 #assert gt.shape == loadedImageChannels[0].shape, self.id + " Data size incompatibility when loading GT %s" % self.gtFilenames[realImageIndex]
                 self.currentGt.append(gt)


            # Initialize the batch and gt variables (so we only need to declare them once)
            if self.FLAGresizeImages ==1:
                self.batch = np.ndarray(shape=(self.batchSize, self.numChannels, self.imageSize[0], self.imageSize[1], self.imageSize[2]), dtype=np.float32)
                #gt =    np.ndarray(shape=(1, self.numClasses, self.gtSize[0], self.gtSize[1], self.gtSize[2]),  dtype=np.float32)
                self.gt =    np.ndarray(shape=(self.batchSize, 1, self.gtSize[0], self.gtSize[1], self.gtSize[2]),  dtype=np.float32)
            else:
                dims = self.currentChannelImages[0][0].shape
                self.batch = np.ndarray(shape=(self.batchSize, self.numChannels, dims[0], dims[1], dims[2]), dtype=np.float32)
                #gt =    np.ndarray(shape=(1, self.numClasses, dims[0], dims[1], dims[2]),  dtype=np.float32)
                self.gt =    np.ndarray(shape=(self.batchSize, 1, dims[0], dims[1], dims[2]),  dtype=np.float32)

        else:
            # reshuffle the IDs of the cases so each epoch, the cases are presented in different order.
            rnd.shuffle(self.IDsCases)



#
##        #TODO FIX IT Generate only batches for the given validation batch size
#        for batch in range(0, len(self.indexCurrentImages)):
#            #print "generating batch %d" % (batch)
#            #logging.debug(self.id + " Generating batch: %d" % batch )
#            isoScale_i = self.generateSingleBatch(batch)
#            #self.batch, self.gt, isoScale_i = self.generateSingleBatch(batch)
#
#            self.queue.put((self.batch,self.gt))
#            self.queueFileNames.put(self.listCurrentFiles[batch])
#            self.queueScalingFactors.put(isoScale_i)
#

#        #TODO FIX IT Generate only batches for the given validation batch size
        for numbatch in range(0, self.batchesPerEpoch):
            #print "generating batch %d" % (batch)
            #logging.debug(self.id + " Generating batch: %d" % batch )
            IDs_aux_i = self.IDsCases[(numbatch*self.batchSize):((numbatch*self.batchSize)+self.batchSize)] #sample from the shuffled list of IDs (self.IDsCases)
            IDs_i = self.indexCurrentImages[IDs_aux_i] # recover the real IDs of the images to work with.

            isoScale_i = self.generateSingleBatchV2(IDs_i)
            #self.batch, self.gt, isoScale_i = self.generateSingleBatch(batch)

            self.queue.put((self.batch,self.gt))
            self.queueScalingFactors.put(isoScale_i)
            currentFilesNames = [];
            for IDs_j in IDs_i:
                currentFilesNames.append(self.listCurrentFiles[IDs_j])
            self.queueFileNames.put(currentFilesNames)



#         # Unload the files if we are loading new files every subpeoc
        if self.loadNewFilesEveryEpoch:
            self.unloadFiles()


##%% TESTING CLASS BINARY BATCH GENERATOR WITH US CUTS
##==============================================================================
#import sys
#sys.path.append("F:\\myPyLibrary")
#import logging
#
#import imp
#import BatchGenerator_v2
#imp.reload(BatchGenerator_v2)
#from BatchGenerator_v2 import *
#from GenerateNiftiFilesFromData import *
#
## LOADING CONFIG FILES
## -----------------------------------------------------------------------------
#confTrain = {}
#if sys.version_info[0] < 3:
#    execfile("F:\\myPyLibrary\\BatchGenerator_v2_configTemp.cfg", confTrain)
#else:
#    exec(open("F:\\myPyLibrary\\BatchGenerator_v2_configTemp.cfg").read(),confTrain)
#
#
### -----------------------------------------------------------------------------
### Create the batch generator
#
#batchGen = BatchGeneratorBinVolDataAug(confTrain, mode='training', infiniteLoop=False)
#batchGen.generateBatches()
#data_i, gt_i = batchGen.getBatch()
#



""" #################################################################################################################
                                      BatchGeneratorBinVolDataAug Class
    ################################################################################################################# """

class BatchGeneratorVolAnd2Dplanes(BatchGenerator):
    """
        Batch generator of volumes and associated 2D planes (e.g., useful when creating a Conditinal Variational Autoencoder, where the conditions are standard planes),
        The class also includes data augmentation (anisotropic and isotropic)
    """

    def __init__(self, confTrain, mode = 'training',infiniteLoop = False, maxQueueSize = 15):
        self.object__ = """
            Initialize a 3D batch generator with a confTrain object.
            - mode (training/validation): Modes can be 'training' or 'validation'. Depending on the mode, the batch generator
            will load the training files (channelsTraining, gtLabelsTraining, roiMasksTraining)
            or the validation files (channelsValidation, gtLabelsValidation, roiMasksValidation) from
            the confTrain object.
        """
        BatchGenerator.__init__(self, confTrain, infiniteLoop = infiniteLoop, maxQueueSize = maxQueueSize)

        # List of currently loaded channel images (size:numOfCasesLoadedPerSubepochTraining or numOfCasesLoadedPerSubepochValidation X channels)
        self.currentChannelImages = []
        # List of currently loaded ROI images (size:numOfCasesLoadedPerSubepochTraining or numOfCasesLoadedPerSubepochValidation)
        self.currentRois = []
        # List of currently loaded GT images (size:numOfCasesLoadedPerSubepochTraining or numOfCasesLoadedPerSubepochValidation)
        self.currentGt = []
        # Names of the cases in the queue
        self.currentFile = []

        self.current_2Dplane_1 = []
        self.current_2Dplane_2 = []
        self.current_2Dplane_3 = []


        # queque of scaling factors (we are monitoring the corresponding scaling factor for each batch)
        self.queueScalingFactors = Queue(maxsize=maxQueueSize)

        self.batch = []           # this will be a numpy array storing the current batch
        self.gt = []              # this will be a numpy array storing the current batch
        self.batch2D_1 = []       # batch of auxiliar 2D image 1
        self.batch2D_2 = []       # batch of auxiliar 2D image 2
        self.batch2D_3 = []       # batch of auxiliar 2D image 3

        self.indexCurrentImages = [];
        self.IDsCases = []      # order in which the images are used.

        self.FLAGverbose = (self.confTrain['FLAGverbose']) if ('FLAGverbose' in self.confTrain) else (0) # Verbosity flag.

        # Mode: can be training or validation
        self.mode = mode
        if mode == 'training':
            self.id = '[WHOLEVOL BATCHGEN TRAIN]'
            self.batchSize = (self.confTrain['batchSizeTraining']) if ('batchSizeTraining' in self.confTrain) else (1)
            self.batchesPerEpoch = 1 # This parameter will be updated latter
        elif mode == 'validation':
            self.id = '[WHOLEVOL BATCHGEN VAL]'
            self.batchSize = 1
            self.batchesPerEpoch = 1 # This parameter will be updated latter
        printMessageVerb(self.FLAGverbose, '-->> Initializing BatchGeneratorBinVolDataAug '+self.id)


        # Image preprocessing options
        # .....................................................................
        self.FLAGresizeImages = self.confTrain['FLAGresizeImages'] # list of flags (one per channel)
        self.imageSize,self.gtSize = (self.confTrain['imageSize'], self.confTrain['gtSize']) if self.FLAGresizeImages else (0,0)
        self.FLAGintensityNormalization = (self.confTrain['FLAGintensityNormalization']) if ('FLAGintensityNormalization' in self.confTrain) else (0)
        self.intensityNormalizationMode,self.intNormParam1,self.intNormParam2 = (self.confTrain['intensityNormalizationMode'], self.confTrain['intNormParam1'], self.confTrain['intNormParam2']) if ('intensityNormalizationMode' in self.confTrain) else ('none',0,0)
        self.isChannelBinary = self.confTrain['isChannelBinary']
        self.FLAGsetBkgrnd,self.bkgrndLabel = (self.confTrain['FLAGsetBkgrnd'], self.confTrain['bkgrndLabel']) if ('FLAGsetBkgrnd' in self.confTrain) else (False,0)

        self.FLAGresizeImages_2D = self.confTrain['FLAGresizeImages_2D'] # list of flags (one per channel)
        self.imageSize_2D = (self.confTrain['imageSize_2D']) if self.FLAGresizeImages_2D else (0,0)
        self.FLAGintensityNormalization_2D = (self.confTrain['FLAGintensityNormalization_2D']) if ('FLAGintensityNormalization_2D' in self.confTrain) else (0)
        self.intensityNormalizationMode_2D,self.intNormParam1_2D,self.intNormParam2_2D = (self.confTrain['intensityNormalizationMode_2D'], self.confTrain['intNormParam1_2D'], self.confTrain['intNormParam2_2D']) if ('intensityNormalizationMode_2D' in self.confTrain) else ('none',0,0)
        self.isChannelBinary_2D = self.confTrain['isChannelBinary_2D']


        # Data augmentation options / params
        # .....................................................................
        self.dataAugmentationRate = self.confTrain['dataAugmentationRate'] if (('dataAugmentationRate' in self.confTrain) and (mode == 'training')) else 0.0
        if (self.dataAugmentationRate > 0.0):
            self.translationRangeX = self.confTrain['translationRangeX']
            self.translationRangeY = self.confTrain['translationRangeY']
            self.translationRangeZ = self.confTrain['translationRangeZ']
            self.rotationRangeX = self.confTrain['rotationRangeX']
            self.rotationRangeY = self.confTrain['rotationRangeY']
            self.rotationRangeZ = self.confTrain['rotationRangeZ']
            self.FLAGholesNoise = (self.confTrain['FLAGholesNoise']) if ('FLAGholesNoise' in self.confTrain) else (0)
            self.holesRatio = self.confTrain['holesRatio']
            self.holesRadiiRange = self.confTrain['holesRadiiRange']
            self.FLAGsaltpepperNoise = (self.confTrain['FLAGsaltpepperNoise']) if ('FLAGsaltpepperNoise' in self.confTrain) else (0)
            self.ratioSaltPepperNoise = self.confTrain['ratioSaltPepperNoise']
            self.saltPepperNoiseSizeRange = self.confTrain['saltPepperNoiseSizeRange']

            textOut = '-->> data augmentation ON: trans x[%3.1f, %3.1f] y[%3.1f, %3.1f] z[%3.1f, %3.1f] - rot x[%3.1f, %3.1f] y[%3.1f, %3.1f] z[%3.1f, %3.1f]' % (self.translationRangeX[0], self.translationRangeX[1], self.translationRangeY[0], self.translationRangeY[1], self.translationRangeZ[0], self.translationRangeZ[1], self.rotationRangeX[0], self.rotationRangeX[1], self.rotationRangeY[0], self.rotationRangeY[1], self.rotationRangeZ[0], self.rotationRangeZ[1])
            printMessageVerb(self.FLAGverbose, textOut)

            self.isotropicScaleFLAG = (self.confTrain['isotropicScaleFLAG']) if ('isotropicScaleFLAG' in self.confTrain) else (1)
            if self.isotropicScaleFLAG:
                self.isotropicScaleRange = self.confTrain['isotropicScaleRange']
                textOut = '-->> isotropic scaling [%3.1f, %3.1f]' % (self.isotropicScaleRange[0], self.isotropicScaleRange[1])
            else:
                self.anisoScaleRangeX = self.confTrain['anisoScaleRangeX']
                self.anisoScaleRangeY = self.confTrain['anisoScaleRangeY']
                self.anisoScaleRangeZ = self.confTrain['anisoScaleRangeZ']
                textOut = '-->> anisotropic scaling x[%3.1f, %3.1f], y[%3.1f, %3.1f], z[%3.1f, %3.1f]' % (self.anisoScaleRangeX[0], self.anisoScaleRangeX[1], self.anisoScaleRangeY[0], self.anisoScaleRangeY[1], self.anisoScaleRangeZ[0], self.anisoScaleRangeZ[1])
            printMessageVerb(self.FLAGverbose, textOut)
        else:
            printMessageVerb(self.FLAGverbose, '-->> data augmentation OFF')


        # Data augmentation 2D planes options / params
        # .....................................................................
        self.dataAugmentationRate_2D = self.confTrain['dataAugmentationRate_2D'] if (('dataAugmentationRate_2D' in self.confTrain) and (mode == 'training')) else 0.0
        if (self.dataAugmentationRate_2D > 0.0):
            self.translationRangeX_2D = self.confTrain['translationRangeX_2D']
            self.translationRangeY_2D = self.confTrain['translationRangeY_2D']
            self.rotationRange_2D = self.confTrain['rotationRange_2D']

            textOut = '-->> 2D data augmentation ON: trans x[%3.1f, %3.1f] y[%3.1f, %3.1f]  - rot [%3.1f, %3.1f]' % (self.translationRangeX_2D[0], self.translationRangeX_2D[1], self.translationRangeY_2D[0], self.translationRangeY_2D[1], self.rotationRange_2D[0], self.rotationRange_2D[1])
            printMessageVerb(self.FLAGverbose, textOut)

            self.isotropicScaleFLAG_2D = (self.confTrain['isotropicScaleFLAG_2D']) if ('isotropicScaleFLAG_2D' in self.confTrain) else (1)
            if self.isotropicScaleFLAG_2D:
                self.isotropicScaleRange_2D = self.confTrain['isotropicScaleRange_2D']
                textOut = '-->> isotropic scaling [%3.1f, %3.1f]' % (self.isotropicScaleRange_2D[0], self.isotropicScaleRange_2D[1])
            else:
                self.anisoScaleRangeX_2D = self.confTrain['anisoScaleRangeX_2D']
                self.anisoScaleRangeY_2D = self.confTrain['anisoScaleRangeY_2D']
                textOut = '-->> anisotropic scaling x[%3.1f, %3.1f], y[%3.1f, %3.1f]' % (self.anisoScaleRangeX_2D[0], self.anisoScaleRangeX_2D[1], self.anisoScaleRangeY_2D[0], self.anisoScaleRangeY_2D[1])
            printMessageVerb(self.FLAGverbose, textOut)
        else:
            printMessageVerb(self.FLAGverbose, '-->> 2D data augmentation OFF')


#        # Read files' paths
#        # .....................................................................
        if mode == 'training':
            logging.info('-- Initializing TRAINING Batch generator')
            self.numChannels = len(self.confTrain['channelsTraining']) # Number of channels per image
            self.numOfCasesLoadedPerEpoch = self.confTrain['numOfCasesLoadedPerEpochWhenTrainingWholeVolume']

            self.loadOrigFilenames(self.confTrain['channelsTraining'], self.confTrain['gtLabelsTraining'],
                          self.confTrain['roiMasksTraining'] if ('roiMasksTraining' in self.confTrain) else None)
            self.allFilenamesIDs = np.array(loadFilenamesSingle(self.confTrain['channelsTraining_ID']),dtype=int)

            if ('channelsTraining2D_cor' in self.confTrain):
                self.allFilenames2Dplane_1 =  loadFilenamesSingle(self.confTrain['channelsTraining2D_cor'])
                self.allFilenames2Dplane_1_ID =  np.array(loadFilenamesSingle(self.confTrain['channelsTraining2D_cor_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_1 = []
                self.allFilenames2Dplane_1_ID = []
            if ('channelsTraining2D_sag' in self.confTrain):
                self.allFilenames2Dplane_2 =  loadFilenamesSingle(self.confTrain['channelsTraining2D_sag'])
                self.allFilenames2Dplane_2_ID =  np.array(loadFilenamesSingle(self.confTrain['channelsTraining2D_sag_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_2 = []
                self.allFilenames2Dplane_2_ID = []
            if ('channelsTraining2D_trvent' in self.confTrain):
                self.allFilenames2Dplane_3 =  loadFilenamesSingle(self.confTrain['channelsTraining2D_trvent'])
                self.allFilenames2Dplane_3_ID =  np.array(loadFilenamesSingle(self.confTrain['channelsTraining2D_trvent_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_3 = []
                self.allFilenames2Dplane_3_ID = []

            # Load all the trainig files:

            # if <=0, load all the cases per epoch
            if (self.numOfCasesLoadedPerEpoch <= 0 or self.numOfCasesLoadedPerEpoch>self.numFiles): self.numOfCasesLoadedPerEpoch = self.numFiles

        elif mode == 'validation':
            logging.info('-- Initializing VALIDATION Batch generator')
            self.numChannels = len(self.confTrain['channelsValidation'])
            #self.numOfCasesLoadedPerEpoch = self.confTrain['numOfCasesLoadedPerEpochWhenValidatingWholeVolume']
            self.loadOrigFilenames(self.confTrain['channelsValidation'], self.confTrain['gtLabelsValidation'],
                               self.confTrain['roiMasksValidation'] if ('roiMasksValidation' in self.confTrain) else None)
            self.numOfCasesLoadedPerEpoch =  len(self.allChannelsFilenames[0]) # load all the cases per epoch.
            self.allFilenamesIDs = np.array(loadFilenamesSingle(self.confTrain['channelsValidation_ID']),dtype=int)
            self.dataAugmentationRate = 0.0

            if ('channelsValidation2D_cor' in self.confTrain):
                self.allFilenames2Dplane_1 =  loadFilenamesSingle(self.confTrain['channelsValidation2D_cor'])
                self.allFilenames2Dplane_1_ID =   np.array(loadFilenamesSingle(self.confTrain['channelsValidation2D_cor_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_1 = []
                self.allFilenames2Dplane_1_ID = []
            if ('channelsValidation2D_sag' in self.confTrain):
                self.allFilenames2Dplane_2 =  loadFilenamesSingle(self.confTrain['channelsValidation2D_sag'])
                self.allFilenames2Dplane_2_ID =   np.array(loadFilenamesSingle(self.confTrain['channelsValidation2D_sag_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_2 = []
                self.allFilenames2Dplane_2_ID = []
            if ('channelsValidation2D_trvent' in self.confTrain):
                self.allFilenames2Dplane_3 =  loadFilenamesSingle(self.confTrain['channelsValidation2D_trvent'])
                self.allFilenames2Dplane_3_ID =   np.array(loadFilenamesSingle(self.confTrain['channelsValidation2D_trvent_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_3 =  []
                self.allFilenames2Dplane_3_ID = []

        else:
            raise Exception('ERROR: Batch generator mode is not valid. Valid options are training or validation')

        textOut = '-->> numChannels: '+str(self.numChannels) + ' - cases loaded per ecpoch: '+str(self.numOfCasesLoadedPerEpoch)
        printMessageVerb(self.FLAGverbose, textOut)

        # Check if, given the number of cases loaded per subepoch and the total number of samples, we
        # will need to load new data in every subepoch or not.

        self.loadNewFilesEveryEpoch = len(self.allChannelsFilenames[0]) > self.numOfCasesLoadedPerEpoch
        logging.info(self.id + " Loading new files every subepoch: " + str(self.loadNewFilesEveryEpoch))
        if self.loadNewFilesEveryEpoch: printMessageVerb(self.FLAGverbose, '-->> need to load new files every epoch')

        # Time controller
        self.timePerEpoch = np.array([])

    def getBatchAndScalingFactor(self):
        """
            It returns a batch and removes it from the front of the queue

            :return: a batch from the queue
        """
        batch = self.queue.get()
        self.currentFile = self.queueFileNames.get()
        self.currentScalingFactor = self.queueScalingFactors.get()
        self.queue.task_done()
        self.queueScalingFactors.task_done()
        self.queueFileNames.task_done()
        return batch


    def loadOrigFilenames(self, channels, gtLabels, roiMasks = None):
        '''
            Load the filenames that will be used to generate the batches.

            :param channels: list containing the path to the text files containing the path to the channels (this list
                            contains one file per channel).
            :param gtLabels: path of the text file containing the paths to gt label files
            :param roiMasks: [Optional] path of the text file containing the paths to ROI files

        '''
        self.allChannelsFilenames, self.gtFilenames, self.roiFilenames, _ = Common.loadFilenames(channels, gtLabels, roiMasks)
        self.numFiles = len(self.gtFilenames)


    def unloadFiles(self):
        '''
            Unload the filenames if need to load new images each epoch.
        '''
        logging.debug(self.id + " Unloading Files")
        for image in self.currentChannelImages:
            for channelImage in image:
                del channelImage

        for image in self.currentGt:
            del image

        for image in self.currentRois:
            del image

        del self.currentChannelImages
        del self.currentGt
        del self.currentRois
        del self.current_2Dplane_1
        del self.current_2Dplane_2
        del self.current_2Dplane_3

        self.currentChannelImages = []
        self.currentGt = []
        self.currentRois = []
        self.current_2Dplane_1 = []
        self.current_2Dplane_2 = []
        self.current_2Dplane_3 = []


    def generateTransformParams2D(self):
        """
            Generate random transformation parameters for data augmentation.
        """
        tX,tY = (int(np.random.uniform(self.translationRangeX_2D[0],self.translationRangeX_2D[1],1)[0]),
                    int(np.random.uniform(self.translationRangeY_2D[0],self.translationRangeY_2D[1],1)[0]))

        if self.isotropicScaleFLAG_2D:
            scaleFactor = np.random.uniform(self.isotropicScaleRange_2D[0],self.isotropicScaleRange_2D[1],1)[0]
        else:
            anisoScaleX_2D = np.random.uniform(self.anisoScaleRangeX_2D[0],self.anisoScaleRangeX_2D[1],1)[0]
            anisoScaleY_2D = np.random.uniform(self.anisoScaleRangeY_2D[0],self.anisoScaleRangeY_2D[1],1)[0]
            scaleFactor = (anisoScaleX_2D, anisoScaleY_2D)

        r = (np.random.uniform(self.rotationRange_2D[0],self.rotationRange_2D[1],1)[0])
        return tX,tY,scaleFactor,r

    def generateTransformParams(self):
        """
            Generate random transformation parameters for data augmentation.
        """
        tX,tY,tZ = (int(np.random.uniform(self.translationRangeX[0],self.translationRangeX[1],1)[0]),
                    int(np.random.uniform(self.translationRangeY[0],self.translationRangeY[1],1)[0]),
                    int(np.random.uniform(self.translationRangeZ[0],self.translationRangeZ[1],1)[0]))

        if self.isotropicScaleFLAG:
            scaleFactor = np.random.uniform(self.isotropicScaleRange[0],self.isotropicScaleRange[1],1)[0]
        else:
            anisoScaleX = np.random.uniform(self.anisoScaleRangeX[0],self.anisoScaleRangeX[1],1)[0]
            anisoScaleY = np.random.uniform(self.anisoScaleRangeY[0],self.anisoScaleRangeY[1],1)[0]
            anisoScaleZ = np.random.uniform(self.anisoScaleRangeZ[0],self.anisoScaleRangeZ[1],1)[0]
            scaleFactor = (anisoScaleX, anisoScaleY, anisoScaleZ)

        rX,rY,rZ = (np.random.uniform(self.rotationRangeX[0],self.rotationRangeX[1],1)[0],
                    np.random.uniform(self.rotationRangeY[0],self.rotationRangeY[1],1)[0],
                    np.random.uniform(self.rotationRangeZ[0],self.rotationRangeZ[1],1)[0])
        return tX,tY,tZ,scaleFactor,rX,rY,rZ


    def generateSingleBatch_VolAnd2DPlanes(self, IDs_i):
        """
            It supposes that the images are already loaded in self.currentChannelImages / self.currentRois / self.currentGt
            / self.batchGen.current_2Dplane_1 / self.batchGen.current_2Dplane_2 / self.batchGen.current_2Dplane_3

            :return: It returns the data and ground truth of a complete batch as data, gt. These structures are theano-compatible with shape:
                        np.ndarray(shape=(self.confTrain['batchSizeTraining'], self.numChannels, dim_1, dim_2, dim_3), dtype=np.float32)
                        It also return the information of the auxiliar 2D planes.

            IDs_i: IDs of the cases to use in the batch.
        """

        if not(self.currentChannelImages == []): # If list of pre-loaded volumes is not empty...

            # define a default scaling factor. The scaling factor is provided as output value.
            if self.isotropicScaleFLAG:
                isoScale_ALL = np.ones([len(IDs_i)])
            else:
                isoScale_ALL = np.ones([len(IDs_i),3])

            # Batch-size Loop
            for id_i in range(0,len(IDs_i)):

                #print('batch-size loop %d %d' %(id_i,len(IDs_i)))
                # Data augmentation Volumes
                # .....................................................................................
                if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):

                    # generate random transformation parameters
                    tX,tY,tZ,isoScale,rX,rY,rZ = self.generateTransformParams()
                    if self.isotropicScaleFLAG:
                        isoScale_ALL[id_i] = isoScale
                    else:
                        isoScale_ALL[id_i,:] = np.array(isoScale)

                    # MASK (apply data augmentation to or auxRoi)
                    if not(self.currentRois == []):
                        auxRoi = self.currentRois[IDs_i[id_i]]
                        rangeRoiValues = np.unique(auxRoi)
                        #auxRoi = applyTransformToVolume(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                        auxRoi = applyTransformToVolumeAnisoScale(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                        auxRoi = restoreIntValues (auxRoi, rangeRoiValues)
                        auxRoi = setOutMaskValue(auxRoi, auxRoi, voxelValue = self.bkgrndLabel)
                    else:
                        auxRoi = np.ones((self.currentChannelImages[IDs_i[id_i]][0].shape))

                   # GT
                    auxGt = self.currentGt[IDs_i[id_i]]
                    rangeGTValues = np.unique(auxGt)
                    #auxGt = applyTransformToVolume(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                    auxGt = applyTransformToVolumeAnisoScale(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                    auxGt = restoreIntValues (auxGt, rangeGTValues)
                    auxGt = setOutMaskValue(auxGt, auxRoi, voxelValue = self.bkgrndLabel)
                    if auxGt.shape != self.gtSize:
                        auxGt = resizeData(auxGt, self.gtSize, preserveIntValues = True)
                    self.gt[id_i,0,:,:,:] = auxGt.astype(np.int16)


                    # IMG channels
                    for channel in range(self.numChannels):

                        auxImg = self.currentChannelImages[IDs_i[id_i]][channel].copy()

                        if self.isChannelBinary[channel] == 1: #if the channel is binary / or is like the GT (with integer discrete values)
                            rangeIMGValues = np.unique(auxImg)
                            #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                            auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                            auxImg = restoreIntValues (auxImg, rangeIMGValues)
                            auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)
                        else:
                            #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                            auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                            auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)

                        # Add noise if needed
                        if self.FLAGholesNoise == 1:
                            foreIDs = np.nonzero(auxGt)
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.holesRatio, radiiRange = self.holesRadiiRange)
                            seedsIDs = np.nonzero(volSeeds)
                            auxImg[seedsIDs] = self.bkgrndLabel

                        if self.FLAGsaltpepperNoise == 1:
                            foreIDs = np.nonzero(np.ones(auxGt.shape))
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                            saltIDs = np.nonzero(volSeeds)
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                            pepperIDs = np.nonzero(volSeeds)
                            auxImg[saltIDs] = self.intNormParam1[channel]
                            auxImg[pepperIDs] = self.intNormParam2[channel]

                        self.batch[id_i, channel, :, :, :] = auxImg.copy()


                # No data augmentation
                # .....................................................................................
                else:
                    isoScale = (1., 1., 1.)
                    # GT
                    self.gt[id_i, 0, :, :, :] = self.currentGt[IDs_i[id_i]].astype(np.int16)

                    # IMG channels
                    for channel in range(self.numChannels):
                        if self.loadNewFilesEveryEpoch:
                            self.batch[id_i, channel, :, :, :] = self.currentChannelImages[IDs_i[id_i]][channel].copy()
                        else:
                            self.batch[id_i, channel, :, :, :] = self.currentChannelImages[IDs_i[id_i]][channel]


                # Data augmentation 2D planes
                # .....................................................................................
                case_ID_i = self.listCurrentIDs[IDs_i[id_i]]

                # 1st aux plane (coronal)
                aux_plane_i = 0
                if len(self.allFilenames2Dplane_1_ID)>0:
                     pos_ID_i = np.where(self.listCurrent2Dplane_1_ID == case_ID_i)[0]
                     rnd.shuffle(pos_ID_i)
                     pos_ID_i = pos_ID_i[0]
                     #pos_ID_i = rnd.shuffle(np.where(self.listCurrent2Dplane_1_ID == case_ID_i)[0])[0] # pick one randomly
                     auxImg = self.current_2Dplane_1[pos_ID_i]
                     self.auxImg_1 = auxImg
                     #1st) apply the corresponding scaling applied to the volume
                     if self.isotropicScaleFLAG:
                         scale_2D_i = np.array([isoScale[0],isoScale[0]])
                     else:
                         scale_2D_i = np.array([isoScale[2],isoScale[1]]) # coronal is [2,1]
                     auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = scale_2D_i, rotAngle = 0., transOffset = (0,0))

                     #2nd) if needed, apply additional data augmentation strategies
                     if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):
                         tX,tY,isoScale2D,r = self.generateTransformParams2D()
                         auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = isoScale2D, rotAngle = r, transOffset = (tX,tY))

                     if self.isChannelBinary_2D[aux_plane_i] == 0:
                         auxImg = preprocessIntensityData(auxImg,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                          imageSize=self.imageSize_2D,
                                                          FLAGpreserveIntValues = False,
                                                          arrayMask=None,
                                                          intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                          intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                          intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                          )

                     elif self.isChannelBinary_2D[aux_plane_i] == 1:
                         auxImg = preprocessIntensityData(auxImg, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                        imageSize=self.imageSize_2D,
                                                        FLAGpreserveIntValues = True,
                                                        arrayMask=None,
                                                        intensityNormalizationMode = None)
                         auxImg = normalizeLabels(auxImg)

                     self.batch2D_1[id_i, 0, :, :] = auxImg.copy()


                # 2st aux plane (sagittal)
                aux_plane_i = 1
                if len(self.allFilenames2Dplane_2_ID)>0:
                     pos_ID_i = np.where(self.listCurrent2Dplane_2_ID == case_ID_i)[0]
                     rnd.shuffle(pos_ID_i)
                     pos_ID_i = pos_ID_i[0]
                     auxImg = self.current_2Dplane_2[pos_ID_i]
                     self.auxImg_2 = auxImg

                     #1st) apply the corresponding scaling applied to the volume
                     if self.isotropicScaleFLAG:
                         scale_2D_i = np.array([isoScale[0],isoScale[0]])
                     else:
                         scale_2D_i = np.array([isoScale[2],isoScale[0]]) # sagittal is [2,0]

                     auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = scale_2D_i, rotAngle = 0., transOffset = (0,0))

                     #2nd) if needed, apply additional data augmentation strategies
                     if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):
                         tX,tY,isoScale2D,r = self.generateTransformParams2D()
                         auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = isoScale2D, rotAngle = r, transOffset = (tX,tY))

                     if self.isChannelBinary_2D[aux_plane_i] == 0:
                         auxImg = preprocessIntensityData(auxImg,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                          imageSize=self.imageSize_2D,
                                                          FLAGpreserveIntValues = False,
                                                          arrayMask=None,
                                                          intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                          intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                          intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                          )

                     elif self.isChannelBinary_2D[aux_plane_i] == 1:
                         auxImg = preprocessIntensityData(auxImg, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                        imageSize=self.imageSize_2D,
                                                        FLAGpreserveIntValues = True,
                                                        arrayMask=None,
                                                        intensityNormalizationMode = None)
                         auxImg = normalizeLabels(auxImg)

                     self.batch2D_2[id_i, 0, :, :] = auxImg.copy()


                # 3rd aux plane (sagittal)
                aux_plane_i = 2
                if len(self.allFilenames2Dplane_3_ID)>0:
                     pos_ID_i = np.where(self.listCurrent2Dplane_3_ID == case_ID_i)[0]
                     rnd.shuffle(pos_ID_i)
                     pos_ID_i = pos_ID_i[0]
                     auxImg = self.current_2Dplane_3[pos_ID_i]

                     #1st) apply the corresponding scaling applied to the volume
                     if self.isotropicScaleFLAG:
                         scale_2D_i = np.array([isoScale[0],isoScale[0]])
                     else:
                         scale_2D_i = np.array([isoScale[1],isoScale[0]]) # axial is [1,0]
                     auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = scale_2D_i, rotAngle = 0., transOffset = (0,0))

                     #2nd) if needed, apply additional data augmentation strategies
                     if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):
                         tX,tY,isoScale2D,r = self.generateTransformParams2D()
                         auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = isoScale2D, rotAngle = r, transOffset = (tX,tY))

                     if self.isChannelBinary_2D[aux_plane_i] == 0:
                         auxImg = preprocessIntensityData(auxImg,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                          imageSize=self.imageSize_2D,
                                                          FLAGpreserveIntValues = False,
                                                          arrayMask=None,
                                                          intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                          intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                          intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                          )

                     elif self.isChannelBinary_2D[aux_plane_i] == 1:
                         auxImg = preprocessIntensityData(auxImg, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                        imageSize=self.imageSize_2D,
                                                        FLAGpreserveIntValues = True,
                                                        arrayMask=None,
                                                        intensityNormalizationMode = None)
                         auxImg = normalizeLabels(auxImg)

                     self.batch2D_3[id_i, 0, :, :] = auxImg.copy()

            return isoScale_ALL

        else:
            raise Exception(self.id + " No images loaded in self.currentVolumes." )


    def generateBatchesForOneEpoch(self):
        """
            Volumes + 2D aux planes.
            Reachable from 'generateBatches' method.
            generateBatches() -> _generateBatches -> generateBatchesForOneEpoch
        """
        # Load the files if needed
        # =====================================================================
        if (self.currentEpoch == 0) or ((self.currentEpoch > 0) and self.loadNewFilesEveryEpoch):

            # Choose the random images that will be sampled in this epoch
            # (*) ToDo -> make sure that new samples are used every epoch.
            self.indexCurrentImages = np.array(self.rndSequence.sample(range(0,self.numFiles), self.numOfCasesLoadedPerEpoch)) #it needs to be a numpy array so we can extract multiple elements with a list (the elements defined by IDsCases, which will be shuffled each epoch)
            self.IDsCases = list(range(0,self.numOfCasesLoadedPerEpoch)) #IDs to the cases in self.indexCurrentImages

            printMessageVerb(self.FLAGverbose, "Loading %d images for epoch %d" % (len(self.indexCurrentImages), self.currentEpoch))
            logging.debug(self.id + " Loading images number : %s" % self.indexCurrentImages )

            self.batchesPerEpoch = int(np.floor(self.numFiles / self.batchSize)) # number of batches per epoch

            self.currentChannelImages = []  # reset the list of images volumes (actual volumes)
            self.currentGt = []             # reset the list of gt volumes (actual volumes)
            self.currentRois = []           # reset the list of ROIs volumes (actual volumes)
            self.listCurrentFiles = []          # reset the list of files loaded per epoch.
            self.listCurrentIDs = np.array([])            # reset the list of IDs

            self.listCurrent2Dplane_1 = []      # reset the list of 2D planes loaded per epoch.
            self.listCurrent2Dplane_1_ID = np.array([])   # reset the list of IDs of 2D planes loaded per epoch.
            self.current_2Dplane_1 = []     # reset the auxiliar 2D planes 1 (actual images)

            self.listCurrent2Dplane_2 = []      # reset the list of 2D planes loaded per epoch.
            self.listCurrent2Dplane_2_ID = np.array([])   # reset the list of IDs of 2D planes loaded per epoch.
            self.current_2Dplane_2 = []     # reset the auxiliar 2D planes 2 (actual images)

            self.listCurrent2Dplane_3 = []      # reset the list of 2D planes loaded per epoch.
            self.listCurrent2Dplane_3_ID = np.array([])   # reset the list of IDs of 2D planes loaded per epoch.
            self.current_2Dplane_3 = []     # reset the auxiliar 2D planes 3 (actual images)

            for realImageIndex in self.indexCurrentImages:
                 loadedImageChannels = [] #list to store all the channels of the current image.
                 self.listCurrentFiles.append(self.allChannelsFilenames[0][realImageIndex])   # List of filenames in the order
                 #self.listCurrentIDs.append(self.allFilenamesIDs[realImageIndex])             # Real patient ID of each case. Used to match the image with the auuxiliar 2D planes
                 case_ID_i = self.allFilenamesIDs[realImageIndex]
                 self.listCurrentIDs = np.append(self.listCurrentIDs, case_ID_i)
                 printMessageVerb(self.FLAGverbose, '-->> loading %d - ID %d'% (realImageIndex,case_ID_i))

                 # Load ROI if exists
                 if ('roiMasksTraining' in self.confTrain):
                     roi = nib.load(self.roiFilenames[realImageIndex]).get_data()
                     roi = preprocessIntensityData(roi, FLAGresizeImages=self.FLAGresizeImages, imageSize=self.imageSize, FLAGpreserveIntValues = True, arrayMask=[], intensityNormalizationMode=None)
                     self.currentRois.append(roi)
                 else:
                     roi = None

                 # Imgs channels ----------------------------------------------
                 # (*) ToDo --> incorporate the masks in the image normalization stage.!!
                 for channel in range(0, self.numChannels):
                      # Load the corresponding image for the corresponding channel and append it to the list of channels
                      # for the current imageIndex
                      # loadedImageChannels.append(nib.load(self.allChannelsFilenames[channel][realImageIndex]).get_data())

                      # Load, preprocess, and normalize the channel.
                      dataIn = nib.load(self.allChannelsFilenames[channel][realImageIndex]).get_data()
                      if self.isChannelBinary[channel] == 0: # not binary input
                          dataIn = preprocessIntensityData(dataIn,FLAGresizeImages=self.FLAGresizeImages,
                                    imageSize=self.imageSize,
                                    FLAGpreserveIntValues = False,
                                    arrayMask=None,
                                    intensityNormalizationMode=self.intensityNormalizationMode[channel],
                                    intNormParam1=self.intNormParam1[channel],
                                    intNormParam2=self.intNormParam2[channel]
                                    )
                          if self.FLAGsetBkgrnd == True:
                              dataIn = setOutMaskValue(dataIn, roi, voxelValue = self.bkgrndLabel)

                      elif self.isChannelBinary[channel] == 1: # binary input --> treat as gt.
                          dataIn = preprocessIntensityData(dataIn, FLAGresizeImages=self.FLAGresizeImages,
                                    imageSize=self.imageSize,
                                    FLAGpreserveIntValues = True,
                                    arrayMask=None,
                                    intensityNormalizationMode = None)
                          dataIn = normalizeLabels(dataIn)

                      # Add the image to the queue of channels
                      loadedImageChannels.append(dataIn)

                      # Check that all the channels have the same dimensions
                      if channel > 0:
                          assert loadedImageChannels[channel].shape == loadedImageChannels[0].shape, self.id + " Data size incompatibility when loading image channels for volume %s" % self.allChannelsFilenames[channel][realImageIndex]

                 # Append all the channels of the image to the list
                 self.currentChannelImages.append(loadedImageChannels)

                 # GT channel ----------------------------------------------
                 gt = nib.load(self.gtFilenames[realImageIndex]).get_data()
                 gt = preprocessIntensityData(gt, FLAGresizeImages=self.FLAGresizeImages, imageSize=self.imageSize, FLAGpreserveIntValues = True, arrayMask=[], intensityNormalizationMode=None)
                 gt = normalizeLabels(gt)
                 #assert gt.shape == loadedImageChannels[0].shape, self.id + " Data size incompatibility when loading GT %s" % self.gtFilenames[realImageIndex]
                 self.currentGt.append(gt)


#                 # Aux 2D planes  ---------------------------------------------

                 # Additional 2D image 1
                 aux_plane_i = 0
                 if len(self.allFilenames2Dplane_1_ID)>0:
                     pos_ID_i = np.where(self.allFilenames2Dplane_1_ID == case_ID_i)[0]
                     assert len(pos_ID_i)>0, self.id + " 2D plane 1 missing for case ID %d" % case_ID_i
                     if len(pos_ID_i)>0:
                         for p_i in pos_ID_i:
                             dataIn = nib.load(self.allFilenames2Dplane_1[p_i]).get_data()[:,:,0]
                             if self.isChannelBinary_2D[aux_plane_i] == 0: # not binary input

                                 dataIn = preprocessIntensityData(dataIn,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                  imageSize=self.imageSize_2D,
                                                                  FLAGpreserveIntValues = False,
                                                                  arrayMask=None,
                                                                  intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                                  intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                                  intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                                  )
                             elif self.isChannelBinary_2D[aux_plane_i] == 1:
                                 dataIn = preprocessIntensityData(dataIn, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                imageSize=self.imageSize_2D,
                                                                FLAGpreserveIntValues = True,
                                                                arrayMask=None,
                                                                intensityNormalizationMode = None)
                                 dataIn = normalizeLabels(dataIn)

                             # Append all the channels of the image to the list
                             self.current_2Dplane_1.append(dataIn)
                             self.listCurrent2Dplane_1.append(self.allFilenames2Dplane_1[p_i])
                             self.listCurrent2Dplane_1_ID = np.append(self.listCurrent2Dplane_1_ID,case_ID_i)
                             printMessageVerb(self.FLAGverbose, '-->>2D plane 1 found: %s'% (self.allFilenames2Dplane_1[p_i]))

                 # Additional 2D image 2
                 aux_plane_i = 1
                 if len(self.allFilenames2Dplane_2_ID)>0:
                     pos_ID_i = np.where(self.allFilenames2Dplane_2_ID == case_ID_i)[0]
                     assert len(pos_ID_i)>0, self.id + " 2D plane 2 missing for case ID %d" % case_ID_i
                     if len(pos_ID_i)>0:
                         for p_i in pos_ID_i:
                             dataIn = nib.load(self.allFilenames2Dplane_2[p_i]).get_data()[:,:,0]
                             if self.isChannelBinary_2D[aux_plane_i] == 0: # not binary input
                                 dataIn = preprocessIntensityData(dataIn,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                  imageSize=self.imageSize_2D,
                                                                  FLAGpreserveIntValues = False,
                                                                  arrayMask=None,
                                                                  intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                                  intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                                  intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                                  )
                             elif self.isChannelBinary_2D[aux_plane_i] == 1:
                                 dataIn = preprocessIntensityData(dataIn, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                imageSize=self.imageSize_2D,
                                                                FLAGpreserveIntValues = True,
                                                                arrayMask=None,
                                                                intensityNormalizationMode = None)
                                 dataIn = normalizeLabels(dataIn)

                             # Append all the channels of the image to the list
                             self.current_2Dplane_2.append(dataIn)
                             self.listCurrent2Dplane_2.append(self.allFilenames2Dplane_2[p_i])
                             self.listCurrent2Dplane_2_ID = np.append(self.listCurrent2Dplane_2_ID,case_ID_i)
                             printMessageVerb(self.FLAGverbose, '-->>2D plane 2 found: %s'% (self.allFilenames2Dplane_2[p_i]))

                 # Additional 2D image 3
                 aux_plane_i = 2
                 if len(self.allFilenames2Dplane_3_ID)>0:
                     pos_ID_i = np.where(self.allFilenames2Dplane_3_ID == case_ID_i)[0]
                     assert len(pos_ID_i)>0, self.id + " 2D plane 2 missing for case ID %d" % case_ID_i
                     if len(pos_ID_i)>0:
                         for p_i in pos_ID_i:
                             dataIn = nib.load(self.allFilenames2Dplane_3[p_i]).get_data()[:,:,0]
                             if self.isChannelBinary_2D[aux_plane_i] == 0: # not binary input
                                 dataIn = preprocessIntensityData(dataIn,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                  imageSize=self.imageSize_2D,
                                                                  FLAGpreserveIntValues = False,
                                                                  arrayMask=None,
                                                                  intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                                  intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                                  intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                                  )
                             elif self.isChannelBinary_2D[aux_plane_i] == 1:
                                 dataIn = preprocessIntensityData(dataIn, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                imageSize=self.imageSize_2D,
                                                                FLAGpreserveIntValues = True,
                                                                arrayMask=None,
                                                                intensityNormalizationMode = None)
                                 dataIn = normalizeLabels(dataIn)

                             # Append all the channels of the image to the list
                             self.current_2Dplane_3.append(dataIn)
                             self.listCurrent2Dplane_3.append(self.allFilenames2Dplane_2[p_i])
                             self.listCurrent2Dplane_3_ID = np.append(self.listCurrent2Dplane_3_ID,case_ID_i)
                             printMessageVerb(self.FLAGverbose, '-->>2D plane 3 found: %s'% (self.allFilenames2Dplane_3[p_i]))



            # Initialize the batch and gt variables (so we only need to declare them once)
            if self.FLAGresizeImages ==1:
                self.batch = np.ndarray(shape=(self.batchSize, self.numChannels, self.imageSize[0], self.imageSize[1], self.imageSize[2]), dtype=np.float32)
                self.batch2D_1 = np.ndarray(shape=(self.batchSize, 1, self.imageSize_2D[0], self.imageSize_2D[1]), dtype=np.float32)
                self.batch2D_2 = np.ndarray(shape=(self.batchSize, 1, self.imageSize_2D[0], self.imageSize_2D[1]), dtype=np.float32)
                self.batch2D_3 = np.ndarray(shape=(self.batchSize, 1, self.imageSize_2D[0], self.imageSize_2D[1]), dtype=np.float32)
                #gt =    np.ndarray(shape=(1, self.numClasses, self.gtSize[0], self.gtSize[1], self.gtSize[2]),  dtype=np.float32)
                self.gt = np.ndarray(shape=(self.batchSize, 1, self.gtSize[0], self.gtSize[1], self.gtSize[2]),  dtype=np.float32)
            else:
                dims = self.currentChannelImages[0][0].shape
                self.batch = np.ndarray(shape=(self.batchSize, self.numChannels, dims[0], dims[1], dims[2]), dtype=np.float32)
                #gt =    np.ndarray(shape=(1, self.numClasses, dims[0], dims[1], dims[2]),  dtype=np.float32)
                dims_2D = self.current_2Dplane_1[1].shape
                self.batch2D_1 = np.ndarray(shape=(self.batchSize, 1, dims_2D[0], dims_2D[1]), dtype=np.float32)
                self.batch2D_2 = np.ndarray(shape=(self.batchSize, 1, dims_2D[0], dims_2D[1]), dtype=np.float32)
                self.batch2D_3 = np.ndarray(shape=(self.batchSize, 1, dims_2D[0], dims_2D[1]), dtype=np.float32)
                self.gt = np.ndarray(shape=(self.batchSize, 1, dims[0], dims[1], dims[2]),  dtype=np.float32)
        else:
            # reshuffle the IDs of the cases so each epoch, the cases are presented in different order.
            rnd.shuffle(self.IDsCases)


        # Create the batches
        # =====================================================================
        t_1 = time.time()
        for numbatch in range(0, self.batchesPerEpoch):
            #print "generating batch %d" % (batch)
            #logging.debug(self.id + " Generating batch: %d" % batch )
            #print('generating batch %d' %(numbatch))
            IDs_aux_i = self.IDsCases[(numbatch*self.batchSize):((numbatch*self.batchSize)+self.batchSize)] #sample from the shuffled list of IDs (self.IDsCases)
            IDs_i = self.indexCurrentImages[IDs_aux_i] # recover the real IDs of the images to work with.
            #            self.isoScale_i = self.generateSingleBatchV2(IDs_i)
            isoScale_i = self.generateSingleBatch_VolAnd2DPlanes(IDs_i)

#            self.aux = IDs_i
#            #self.batch, self.gt, isoScale_i = self.generateSingleBatch(batch)

            self.queue.put((self.batch,self.gt, self.batch2D_1, self.batch2D_2, self.batch2D_3))
            self.queueScalingFactors.put(isoScale_i)
            currentFilesNames = [];
            for IDs_j in IDs_i:
                currentFilesNames.append(self.listCurrentFiles[IDs_j])
            self.queueFileNames.put(currentFilesNames)

            t_2 = time.time()
            self.timePerEpoch = np.append(self.timePerEpoch,t_2-t_1)
            t_1 = t_2
#         # Unload the files if we are loading new files every subpeoc
        if self.loadNewFilesEveryEpoch:
            self.unloadFiles()






""" #################################################################################################################
                                      BatchGeneratorBinVolDataAug Class
    ################################################################################################################# """

class BatchGeneratorVolAnd2DplanesMultiThread(BatchGenerator):
    """
        Batch generator of volumes and associated 2D planes (e.g., useful when creating a Conditinal Variational Autoencoder, where the conditions are standard planes),
        The class also includes data augmentation (anisotropic and isotropic)
        In this case we will use multiple Threads to generate the batches so we can speed it up if needed.
    """

    def __init__(self, confTrain, mode = 'training',infiniteLoop = False, maxQueueSize = 15):
        self.object__ = """
            Initialize a 3D batch generator with a confTrain object.
            - mode (training/validation): Modes can be 'training' or 'validation'. Depending on the mode, the batch generator
            will load the training files (channelsTraining, gtLabelsTraining, roiMasksTraining)
            or the validation files (channelsValidation, gtLabelsValidation, roiMasksValidation) from
            the confTrain object.
        """
        BatchGenerator.__init__(self, confTrain, infiniteLoop = infiniteLoop, maxQueueSize = maxQueueSize)

        # List of currently loaded channel images (size:numOfCasesLoadedPerSubepochTraining or numOfCasesLoadedPerSubepochValidation X channels)
        self.currentChannelImages = []
        # List of currently loaded ROI images (size:numOfCasesLoadedPerSubepochTraining or numOfCasesLoadedPerSubepochValidation)
        self.currentRois = []
        # List of currently loaded GT images (size:numOfCasesLoadedPerSubepochTraining or numOfCasesLoadedPerSubepochValidation)
        self.currentGt = []
        # Names of the cases in the queue
        self.currentFile = []

        self.numThreads = (self.confTrain['numThreads']) if ('numThreads' in self.confTrain) else (1) # Number of Threads used to generate the batches.

        self.current_2Dplane_1 = []
        self.current_2Dplane_2 = []
        self.current_2Dplane_3 = []


        # queque of scaling factors (we are monitoring the corresponding scaling factor for each batch)
        self.queueScalingFactors = Queue(maxsize=maxQueueSize)

        self.batch = []           # this will be a numpy array storing the current batch
        self.gt = []              # this will be a numpy array storing the current batch
        self.batch2D_1 = []       # batch of auxiliar 2D image 1
        self.batch2D_2 = []       # batch of auxiliar 2D image 2
        self.batch2D_3 = []       # batch of auxiliar 2D image 3

        self.lock = threading.Lock() # Lock to synchronize the multiple threads

        self.indexCurrentImages = [];
        self.IDsCases = []      # order in which the images are used.

        self.FLAGverbose = (self.confTrain['FLAGverbose']) if ('FLAGverbose' in self.confTrain) else (0) # Verbosity flag.

        # Mode: can be training or validation
        self.mode = mode
        if mode == 'training':
            self.id = '[WHOLEVOL BATCHGEN TRAIN]'
            self.batchSize = (self.confTrain['batchSizeTraining']) if ('batchSizeTraining' in self.confTrain) else (1)
            self.batchesPerEpoch = 1 # This parameter will be updated latter
        elif mode == 'validation':
            self.id = '[WHOLEVOL BATCHGEN VAL]'
            self.batchSize = 1
            self.batchesPerEpoch = 1 # This parameter will be updated latter
        printMessageVerb(self.FLAGverbose, '-->> Initializing BatchGeneratorBinVolDataAug '+self.id)


        # Image preprocessing options
        # .....................................................................
        self.FLAGresizeImages = self.confTrain['FLAGresizeImages'] # list of flags (one per channel)
        self.imageSize,self.gtSize = (self.confTrain['imageSize'], self.confTrain['gtSize']) if self.FLAGresizeImages else (0,0)
        self.FLAGintensityNormalization = (self.confTrain['FLAGintensityNormalization']) if ('FLAGintensityNormalization' in self.confTrain) else (0)
        self.intensityNormalizationMode,self.intNormParam1,self.intNormParam2 = (self.confTrain['intensityNormalizationMode'], self.confTrain['intNormParam1'], self.confTrain['intNormParam2']) if ('intensityNormalizationMode' in self.confTrain) else ('none',0,0)
        self.isChannelBinary = self.confTrain['isChannelBinary']
        self.FLAGsetBkgrnd,self.bkgrndLabel = (self.confTrain['FLAGsetBkgrnd'], self.confTrain['bkgrndLabel']) if ('FLAGsetBkgrnd' in self.confTrain) else (False,0)

        self.FLAGresizeImages_2D = self.confTrain['FLAGresizeImages_2D'] # list of flags (one per channel)
        self.imageSize_2D = (self.confTrain['imageSize_2D']) if self.FLAGresizeImages_2D else (0,0)
        self.FLAGintensityNormalization_2D = (self.confTrain['FLAGintensityNormalization_2D']) if ('FLAGintensityNormalization_2D' in self.confTrain) else (0)
        self.intensityNormalizationMode_2D,self.intNormParam1_2D,self.intNormParam2_2D = (self.confTrain['intensityNormalizationMode_2D'], self.confTrain['intNormParam1_2D'], self.confTrain['intNormParam2_2D']) if ('intensityNormalizationMode_2D' in self.confTrain) else ('none',0,0)
        self.isChannelBinary_2D = self.confTrain['isChannelBinary_2D']


        # Data augmentation options / params
        # .....................................................................
        self.dataAugmentationRate = self.confTrain['dataAugmentationRate'] if (('dataAugmentationRate' in self.confTrain) and (mode == 'training')) else 0.0
        if (self.dataAugmentationRate > 0.0):
            self.translationRangeX = self.confTrain['translationRangeX']
            self.translationRangeY = self.confTrain['translationRangeY']
            self.translationRangeZ = self.confTrain['translationRangeZ']
            self.rotationRangeX = self.confTrain['rotationRangeX']
            self.rotationRangeY = self.confTrain['rotationRangeY']
            self.rotationRangeZ = self.confTrain['rotationRangeZ']
            self.FLAGholesNoise = (self.confTrain['FLAGholesNoise']) if ('FLAGholesNoise' in self.confTrain) else (0)
            self.holesRatio = self.confTrain['holesRatio']
            self.holesRadiiRange = self.confTrain['holesRadiiRange']
            self.FLAGsaltpepperNoise = (self.confTrain['FLAGsaltpepperNoise']) if ('FLAGsaltpepperNoise' in self.confTrain) else (0)
            self.ratioSaltPepperNoise = self.confTrain['ratioSaltPepperNoise']
            self.saltPepperNoiseSizeRange = self.confTrain['saltPepperNoiseSizeRange']

            textOut = '-->> data augmentation ON: trans x[%3.1f, %3.1f] y[%3.1f, %3.1f] z[%3.1f, %3.1f] - rot x[%3.1f, %3.1f] y[%3.1f, %3.1f] z[%3.1f, %3.1f]' % (self.translationRangeX[0], self.translationRangeX[1], self.translationRangeY[0], self.translationRangeY[1], self.translationRangeZ[0], self.translationRangeZ[1], self.rotationRangeX[0], self.rotationRangeX[1], self.rotationRangeY[0], self.rotationRangeY[1], self.rotationRangeZ[0], self.rotationRangeZ[1])
            printMessageVerb(self.FLAGverbose, textOut)

            self.isotropicScaleFLAG = (self.confTrain['isotropicScaleFLAG']) if ('isotropicScaleFLAG' in self.confTrain) else (1)
            if self.isotropicScaleFLAG:
                self.isotropicScaleRange = self.confTrain['isotropicScaleRange']
                textOut = '-->> isotropic scaling [%3.1f, %3.1f]' % (self.isotropicScaleRange[0], self.isotropicScaleRange[1])
            else:
                self.anisoScaleRangeX = self.confTrain['anisoScaleRangeX']
                self.anisoScaleRangeY = self.confTrain['anisoScaleRangeY']
                self.anisoScaleRangeZ = self.confTrain['anisoScaleRangeZ']
                textOut = '-->> anisotropic scaling x[%3.1f, %3.1f], y[%3.1f, %3.1f], z[%3.1f, %3.1f]' % (self.anisoScaleRangeX[0], self.anisoScaleRangeX[1], self.anisoScaleRangeY[0], self.anisoScaleRangeY[1], self.anisoScaleRangeZ[0], self.anisoScaleRangeZ[1])
            printMessageVerb(self.FLAGverbose, textOut)
        else:
            self.isotropicScaleFLAG = 1
            printMessageVerb(self.FLAGverbose, '-->> data augmentation OFF')


        # Data augmentation 2D planes options / params
        # .....................................................................
        self.dataAugmentationRate_2D = self.confTrain['dataAugmentationRate_2D'] if (('dataAugmentationRate_2D' in self.confTrain) and (mode == 'training')) else 0.0
        if (self.dataAugmentationRate_2D > 0.0):
            self.translationRangeX_2D = self.confTrain['translationRangeX_2D']
            self.translationRangeY_2D = self.confTrain['translationRangeY_2D']
            self.rotationRange_2D = self.confTrain['rotationRange_2D']

            textOut = '-->> 2D data augmentation ON: trans x[%3.1f, %3.1f] y[%3.1f, %3.1f]  - rot [%3.1f, %3.1f]' % (self.translationRangeX_2D[0], self.translationRangeX_2D[1], self.translationRangeY_2D[0], self.translationRangeY_2D[1], self.rotationRange_2D[0], self.rotationRange_2D[1])
            printMessageVerb(self.FLAGverbose, textOut)

            self.isotropicScaleFLAG_2D = (self.confTrain['isotropicScaleFLAG_2D']) if ('isotropicScaleFLAG_2D' in self.confTrain) else (1)
            if self.isotropicScaleFLAG_2D:
                self.isotropicScaleRange_2D = self.confTrain['isotropicScaleRange_2D']
                textOut = '-->> isotropic scaling [%3.1f, %3.1f]' % (self.isotropicScaleRange_2D[0], self.isotropicScaleRange_2D[1])
            else:
                self.anisoScaleRangeX_2D = self.confTrain['anisoScaleRangeX_2D']
                self.anisoScaleRangeY_2D = self.confTrain['anisoScaleRangeY_2D']
                textOut = '-->> anisotropic scaling x[%3.1f, %3.1f], y[%3.1f, %3.1f]' % (self.anisoScaleRangeX_2D[0], self.anisoScaleRangeX_2D[1], self.anisoScaleRangeY_2D[0], self.anisoScaleRangeY_2D[1])
            printMessageVerb(self.FLAGverbose, textOut)
        else:
            printMessageVerb(self.FLAGverbose, '-->> 2D data augmentation OFF')


#        # Read files' paths
#        # .....................................................................
        if mode == 'training':
            logging.info('-- Initializing TRAINING Batch generator')
            self.numChannels = len(self.confTrain['channelsTraining']) # Number of channels per image
            self.numOfCasesLoadedPerEpoch = self.confTrain['numOfCasesLoadedPerEpochWhenTrainingWholeVolume']

            self.loadOrigFilenames(self.confTrain['channelsTraining'], self.confTrain['gtLabelsTraining'],
                          self.confTrain['roiMasksTraining'] if ('roiMasksTraining' in self.confTrain) else None)
            self.allFilenamesIDs = np.array(loadFilenamesSingle(self.confTrain['channelsTraining_ID']),dtype=int)

            if ('channelsTraining2D_cor' in self.confTrain):
                self.allFilenames2Dplane_1 =  loadFilenamesSingle(self.confTrain['channelsTraining2D_cor'])
                self.allFilenames2Dplane_1_ID =  np.array(loadFilenamesSingle(self.confTrain['channelsTraining2D_cor_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_1 = []
                self.allFilenames2Dplane_1_ID = []
            if ('channelsTraining2D_sag' in self.confTrain):
                self.allFilenames2Dplane_2 =  loadFilenamesSingle(self.confTrain['channelsTraining2D_sag'])
                self.allFilenames2Dplane_2_ID =  np.array(loadFilenamesSingle(self.confTrain['channelsTraining2D_sag_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_2 = []
                self.allFilenames2Dplane_2_ID = []
            if ('channelsTraining2D_trvent' in self.confTrain):
                self.allFilenames2Dplane_3 =  loadFilenamesSingle(self.confTrain['channelsTraining2D_trvent'])
                self.allFilenames2Dplane_3_ID =  np.array(loadFilenamesSingle(self.confTrain['channelsTraining2D_trvent_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_3 = []
                self.allFilenames2Dplane_3_ID = []

            # Load all the trainig files:

            # if <=0, load all the cases per epoch
            if (self.numOfCasesLoadedPerEpoch <= 0 or self.numOfCasesLoadedPerEpoch>self.numFiles): self.numOfCasesLoadedPerEpoch = self.numFiles

        elif mode == 'validation':
            logging.info('-- Initializing VALIDATION Batch generator')
            self.numChannels = len(self.confTrain['channelsValidation'])
            #self.numOfCasesLoadedPerEpoch = self.confTrain['numOfCasesLoadedPerEpochWhenValidatingWholeVolume']
            self.loadOrigFilenames(self.confTrain['channelsValidation'], self.confTrain['gtLabelsValidation'],
                               self.confTrain['roiMasksValidation'] if ('roiMasksValidation' in self.confTrain) else None)
            self.numOfCasesLoadedPerEpoch =  len(self.allChannelsFilenames[0]) # load all the cases per epoch.
            self.allFilenamesIDs = np.array(loadFilenamesSingle(self.confTrain['channelsValidation_ID']),dtype=int)
            self.dataAugmentationRate = 0.0

            if ('channelsValidation2D_cor' in self.confTrain):
                self.allFilenames2Dplane_1 =  loadFilenamesSingle(self.confTrain['channelsValidation2D_cor'])
                self.allFilenames2Dplane_1_ID =   np.array(loadFilenamesSingle(self.confTrain['channelsValidation2D_cor_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_1 = []
                self.allFilenames2Dplane_1_ID = []
            if ('channelsValidation2D_sag' in self.confTrain):
                self.allFilenames2Dplane_2 =  loadFilenamesSingle(self.confTrain['channelsValidation2D_sag'])
                self.allFilenames2Dplane_2_ID =   np.array(loadFilenamesSingle(self.confTrain['channelsValidation2D_sag_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_2 = []
                self.allFilenames2Dplane_2_ID = []
            if ('channelsValidation2D_trvent' in self.confTrain):
                self.allFilenames2Dplane_3 =  loadFilenamesSingle(self.confTrain['channelsValidation2D_trvent'])
                self.allFilenames2Dplane_3_ID =   np.array(loadFilenamesSingle(self.confTrain['channelsValidation2D_trvent_ID']),dtype=int)
            else:
                self.allFilenames2Dplane_3 =  []
                self.allFilenames2Dplane_3_ID = []

        else:
            raise Exception('ERROR: Batch generator mode is not valid. Valid options are training or validation')

        textOut = '-->> numChannels: '+str(self.numChannels) + ' - cases loaded per ecpoch: '+str(self.numOfCasesLoadedPerEpoch)
        printMessageVerb(self.FLAGverbose, textOut)

        # Check if, given the number of cases loaded per subepoch and the total number of samples, we
        # will need to load new data in every subepoch or not.

        self.loadNewFilesEveryEpoch = len(self.allChannelsFilenames[0]) > self.numOfCasesLoadedPerEpoch
        logging.info(self.id + " Loading new files every subepoch: " + str(self.loadNewFilesEveryEpoch))
        if self.loadNewFilesEveryEpoch: printMessageVerb(self.FLAGverbose, '-->> need to load new files every epoch')

        # Time controller
        self.timePerEpoch = np.array([])

    def getBatchAndScalingFactor(self):
        """
            It returns a batch and removes it from the front of the queue

            :return: a batch from the queue
        """
        batch = self.queue.get()
        self.currentFile = self.queueFileNames.get()
        self.currentScalingFactor = self.queueScalingFactors.get()
        self.queue.task_done()
        self.queueScalingFactors.task_done()
        self.queueFileNames.task_done()
        return batch


    def loadOrigFilenames(self, channels, gtLabels, roiMasks = None):
        '''
            Load the filenames that will be used to generate the batches.

            :param channels: list containing the path to the text files containing the path to the channels (this list
                            contains one file per channel).
            :param gtLabels: path of the text file containing the paths to gt label files
            :param roiMasks: [Optional] path of the text file containing the paths to ROI files

        '''
        self.allChannelsFilenames, self.gtFilenames, self.roiFilenames, _ = Common.loadFilenames(channels, gtLabels, roiMasks)
        self.numFiles = len(self.gtFilenames)


    def unloadFiles(self):
        '''
            Unload the filenames if need to load new images each epoch.
        '''
        logging.debug(self.id + " Unloading Files")
        for image in self.currentChannelImages:
            for channelImage in image:
                del channelImage

        for image in self.currentGt:
            del image

        for image in self.currentRois:
            del image

        del self.currentChannelImages
        del self.currentGt
        del self.currentRois
        del self.current_2Dplane_1
        del self.current_2Dplane_2
        del self.current_2Dplane_3

        self.currentChannelImages = []
        self.currentGt = []
        self.currentRois = []
        self.current_2Dplane_1 = []
        self.current_2Dplane_2 = []
        self.current_2Dplane_3 = []


    def generateTransformParams2D(self):
        """
            Generate random transformation parameters for data augmentation.
        """
        tX,tY = (int(np.random.uniform(self.translationRangeX_2D[0],self.translationRangeX_2D[1],1)[0]),
                    int(np.random.uniform(self.translationRangeY_2D[0],self.translationRangeY_2D[1],1)[0]))

        if self.isotropicScaleFLAG_2D:
            scaleFactor = np.random.uniform(self.isotropicScaleRange_2D[0],self.isotropicScaleRange_2D[1],1)[0]
        else:
            anisoScaleX_2D = np.random.uniform(self.anisoScaleRangeX_2D[0],self.anisoScaleRangeX_2D[1],1)[0]
            anisoScaleY_2D = np.random.uniform(self.anisoScaleRangeY_2D[0],self.anisoScaleRangeY_2D[1],1)[0]
            scaleFactor = (anisoScaleX_2D, anisoScaleY_2D)

        r = (np.random.uniform(self.rotationRange_2D[0],self.rotationRange_2D[1],1)[0])
        return tX,tY,scaleFactor,r

    def generateTransformParams(self):
        """
            Generate random transformation parameters for data augmentation.
        """
        tX,tY,tZ = (int(np.random.uniform(self.translationRangeX[0],self.translationRangeX[1],1)[0]),
                    int(np.random.uniform(self.translationRangeY[0],self.translationRangeY[1],1)[0]),
                    int(np.random.uniform(self.translationRangeZ[0],self.translationRangeZ[1],1)[0]))

        if self.isotropicScaleFLAG:
            scaleFactor = np.random.uniform(self.isotropicScaleRange[0],self.isotropicScaleRange[1],1)[0]
        else:
            anisoScaleX = np.random.uniform(self.anisoScaleRangeX[0],self.anisoScaleRangeX[1],1)[0]
            anisoScaleY = np.random.uniform(self.anisoScaleRangeY[0],self.anisoScaleRangeY[1],1)[0]
            anisoScaleZ = np.random.uniform(self.anisoScaleRangeZ[0],self.anisoScaleRangeZ[1],1)[0]
            scaleFactor = (anisoScaleX, anisoScaleY, anisoScaleZ)

        rX,rY,rZ = (np.random.uniform(self.rotationRangeX[0],self.rotationRangeX[1],1)[0],
                    np.random.uniform(self.rotationRangeY[0],self.rotationRangeY[1],1)[0],
                    np.random.uniform(self.rotationRangeZ[0],self.rotationRangeZ[1],1)[0])
        return tX,tY,tZ,scaleFactor,rX,rY,rZ



    def batchGeneratorThread(self, threadID = 0, IDsList = []):

        numBatches = len(IDsList)//self.batchSize
        printMessageVerb(self.FLAGverbose, 'Thr_id: %d  - Num Batches: %d '% (threadID, numBatches))

        for b_i in range(0,numBatches):
            printMessageVerb(self.FLAGverbose, 'Thr_id: %d - generating batch: %d' %(threadID, b_i+1))
            IDs_aux_i = IDsList[(b_i*self.batchSize):((b_i*self.batchSize)+self.batchSize)]
            IDs_i = self.indexCurrentImages[IDs_aux_i] # recover the real IDs of the images to work with.
            (isoScale_i, batch_i,gt_i,batch2D_1_i, batch2D_2_i, batch2D_3_i) = self.generateSingleBatch_VolAnd2DPlanesThread(IDs_i)

            printMessageVerb(self.FLAGverbose, 'Thr_id: %d - waiting for lock...' %(threadID))
            self.lock.acquire()
            try:
                printMessageVerb(self.FLAGverbose, 'Thr_id: %d - Acquired lock!' %(threadID))
                self.queue.put((batch_i,gt_i, batch2D_1_i, batch2D_2_i, batch2D_3_i))
                printMessageVerb(self.FLAGverbose, 'Thr_id: %d - Element in queue!' %(threadID))
                self.queueScalingFactors.put(isoScale_i)
                currentFilesNames = [];
                for IDs_j in IDs_i:
                    currentFilesNames.append(self.listCurrentFiles[IDs_j])
                self.queueFileNames.put(currentFilesNames)
            finally:
                printMessageVerb(self.FLAGverbose, 'Thr_id: %d - Releasing lock' %(threadID))
                self.lock.release()


    def generateSingleBatch_VolAnd2DPlanesThread(self, IDs_i):
        """
            MultiThread Version...
            --> !!! Make sure this method doesn't access or change any global variable.

            It supposes that the images are already loaded in self.currentChannelImages / self.currentRois / self.currentGt
            / self.batchGen.current_2Dplane_1 / self.batchGen.current_2Dplane_2 / self.batchGen.current_2Dplane_3

            :return: It returns the data and ground truth of a complete batch as data, gt. These structures are theano-compatible with shape:
                        np.ndarray(shape=(self.confTrain['batchSizeTraining'], self.numChannels, dim_1, dim_2, dim_3), dtype=np.float32)
                        It also return the information of the auxiliar 2D planes.

            IDs_i: IDs of the cases to use in the batch.
        """

        if not(self.currentChannelImages == []): # If list of pre-loaded volumes is not empty...


            # define local variables
            if self.FLAGresizeImages ==1:
                batch = np.ndarray(shape=(self.batchSize, self.numChannels, self.imageSize[0], self.imageSize[1], self.imageSize[2]), dtype=np.float32)
                batch2D_1 = np.ndarray(shape=(self.batchSize, 1, self.imageSize_2D[0], self.imageSize_2D[1]), dtype=np.float32)
                batch2D_2 = np.ndarray(shape=(self.batchSize, 1, self.imageSize_2D[0], self.imageSize_2D[1]), dtype=np.float32)
                batch2D_3 = np.ndarray(shape=(self.batchSize, 1, self.imageSize_2D[0], self.imageSize_2D[1]), dtype=np.float32)
                #gt =    np.ndarray(shape=(1, self.numClasses, self.gtSize[0], self.gtSize[1], self.gtSize[2]),  dtype=np.float32)
                gt = np.ndarray(shape=(self.batchSize, 1, self.gtSize[0], self.gtSize[1], self.gtSize[2]),  dtype=np.float32)
            else:
                dims = self.currentChannelImages[0][0].shape
                batch = np.ndarray(shape=(self.batchSize, self.numChannels, dims[0], dims[1], dims[2]), dtype=np.float32)
                #gt =    np.ndarray(shape=(1, self.numClasses, dims[0], dims[1], dims[2]),  dtype=np.float32)
                dims_2D = self.current_2Dplane_1[1].shape
                batch2D_1 = np.ndarray(shape=(self.batchSize, 1, dims_2D[0], dims_2D[1]), dtype=np.float32)
                batch2D_2 = np.ndarray(shape=(self.batchSize, 1, dims_2D[0], dims_2D[1]), dtype=np.float32)
                batch2D_3 = np.ndarray(shape=(self.batchSize, 1, dims_2D[0], dims_2D[1]), dtype=np.float32)
                gt = np.ndarray(shape=(self.batchSize, 1, dims[0], dims[1], dims[2]),  dtype=np.float32)


            # define a default scaling factor. The scaling factor is provided as output value.
            if (self.mode == 'training') and (self.isotropicScaleFLAG):
                isoScale_ALL = np.ones([len(IDs_i)])
            else:
                isoScale_ALL = np.ones([len(IDs_i),3])

            # Batch-size Loop
            for id_i in range(0,len(IDs_i)):

                #print('batch-size loop %d %d' %(id_i,len(IDs_i)))
                # Data augmentation Volumes
                # .....................................................................................
                if (self.mode == 'training') and (np.random.uniform(size=1)[0] < self.dataAugmentationRate):

                    # generate random transformation parameters
                    tX,tY,tZ,isoScale,rX,rY,rZ = self.generateTransformParams()
                    if self.isotropicScaleFLAG:
                        isoScale_ALL[id_i] = isoScale
                    else:
                        isoScale_ALL[id_i,:] = np.array(isoScale)

                    # MASK (apply data augmentation to or auxRoi)
                    if not(self.currentRois == []):
                        auxRoi = self.currentRois[IDs_i[id_i]]
                        rangeRoiValues = np.unique(auxRoi)
                        #auxRoi = applyTransformToVolume(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                        auxRoi = applyTransformToVolumeAnisoScale(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                        auxRoi = restoreIntValues (auxRoi, rangeRoiValues)
                        auxRoi = setOutMaskValue(auxRoi, auxRoi, voxelValue = self.bkgrndLabel)
                    else:
                        auxRoi = np.ones((self.currentChannelImages[IDs_i[id_i]][0].shape))

                   # GT
                    auxGt = self.currentGt[IDs_i[id_i]]
                    rangeGTValues = np.unique(auxGt)
                    #auxGt = applyTransformToVolume(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                    auxGt = applyTransformToVolumeAnisoScale(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                    auxGt = restoreIntValues (auxGt, rangeGTValues)
                    auxGt = setOutMaskValue(auxGt, auxRoi, voxelValue = self.bkgrndLabel)
                    if auxGt.shape != self.gtSize:
                        auxGt = resizeData(auxGt, self.gtSize, preserveIntValues = True)
                    gt[id_i,0,:,:,:] = auxGt.astype(np.int16)


                    # IMG channels
                    for channel in range(self.numChannels):

                        auxImg = self.currentChannelImages[IDs_i[id_i]][channel].copy()

                        if self.isChannelBinary[channel] == 1: #if the channel is binary / or is like the GT (with integer discrete values)
                            rangeIMGValues = np.unique(auxImg)
                            #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                            auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                            auxImg = restoreIntValues (auxImg, rangeIMGValues)
                            auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)
                        else:
                            #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                            auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                            auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)

                        # Add noise if needed
                        if self.FLAGholesNoise == 1:
                            foreIDs = np.nonzero(auxGt)
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.holesRatio, radiiRange = self.holesRadiiRange)
                            seedsIDs = np.nonzero(volSeeds)
                            auxImg[seedsIDs] = self.bkgrndLabel

                        if self.FLAGsaltpepperNoise == 1:
                            foreIDs = np.nonzero(np.ones(auxGt.shape))
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                            saltIDs = np.nonzero(volSeeds)
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                            pepperIDs = np.nonzero(volSeeds)
                            auxImg[saltIDs] = self.intNormParam1[channel]
                            auxImg[pepperIDs] = self.intNormParam2[channel]

                        batch[id_i, channel, :, :, :] = auxImg.copy()


                # No data augmentation
                # .....................................................................................
                else:
                    isoScale = (1., 1., 1.)
                    self.isotropicScaleFLAG = 1
                    # GT
                    gt[id_i, 0, :, :, :] = self.currentGt[IDs_i[id_i]].astype(np.int16)

                    # IMG channels
                    for channel in range(self.numChannels):
                        if self.loadNewFilesEveryEpoch:
                            batch[id_i, channel, :, :, :] = self.currentChannelImages[IDs_i[id_i]][channel].copy()
                        else:
                            batch[id_i, channel, :, :, :] = self.currentChannelImages[IDs_i[id_i]][channel]


                # Data augmentation 2D planes
                # .....................................................................................
                case_ID_i = self.listCurrentIDs[IDs_i[id_i]]

                # 1st aux plane (coronal)
                aux_plane_i = 0
                if len(self.allFilenames2Dplane_1_ID)>0:
                     pos_ID_i = np.where(self.listCurrent2Dplane_1_ID == case_ID_i)[0]
                     rnd.shuffle(pos_ID_i)
                     pos_ID_i = pos_ID_i[0]
                     #pos_ID_i = rnd.shuffle(np.where(self.listCurrent2Dplane_1_ID == case_ID_i)[0])[0] # pick one randomly
                     auxImg = self.current_2Dplane_1[pos_ID_i]
                     #1st) apply the corresponding scaling applied to the volume
                     if self.isotropicScaleFLAG:
                         scale_2D_i = np.array([isoScale[0],isoScale[0]])
                     else:
                         scale_2D_i = np.array([isoScale[2],isoScale[1]]) # coronal is [2,1]
                     auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = scale_2D_i, rotAngle = 0., transOffset = (0,0))

                     #2nd) if needed, apply additional data augmentation strategies
                     if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):
                         tX,tY,isoScale2D,r = self.generateTransformParams2D()
                         auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = isoScale2D, rotAngle = r, transOffset = (tX,tY))

                     if self.isChannelBinary_2D[aux_plane_i] == 0:
                         auxImg = preprocessIntensityData(auxImg,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                          imageSize=self.imageSize_2D,
                                                          FLAGpreserveIntValues = False,
                                                          arrayMask=None,
                                                          intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                          intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                          intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                          )

                     elif self.isChannelBinary_2D[aux_plane_i] == 1:
                         auxImg = preprocessIntensityData(auxImg, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                        imageSize=self.imageSize_2D,
                                                        FLAGpreserveIntValues = True,
                                                        arrayMask=None,
                                                        intensityNormalizationMode = None)
                         auxImg = normalizeLabels(auxImg)

                     batch2D_1[id_i, 0, :, :] = auxImg.copy()


                # 2st aux plane (sagittal)
                aux_plane_i = 1
                if len(self.allFilenames2Dplane_2_ID)>0:
                     pos_ID_i = np.where(self.listCurrent2Dplane_2_ID == case_ID_i)[0]
                     rnd.shuffle(pos_ID_i)
                     pos_ID_i = pos_ID_i[0]
                     auxImg = self.current_2Dplane_2[pos_ID_i]

                     #1st) apply the corresponding scaling applied to the volume
                     if self.isotropicScaleFLAG:
                         scale_2D_i = np.array([isoScale[0],isoScale[0]])
                     else:
                         scale_2D_i = np.array([isoScale[2],isoScale[0]]) # sagittal is [2,0]

                     auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = scale_2D_i, rotAngle = 0., transOffset = (0,0))

                     #2nd) if needed, apply additional data augmentation strategies
                     if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):
                         tX,tY,isoScale2D,r = self.generateTransformParams2D()
                         auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = isoScale2D, rotAngle = r, transOffset = (tX,tY))

                     if self.isChannelBinary_2D[aux_plane_i] == 0:
                         auxImg = preprocessIntensityData(auxImg,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                          imageSize=self.imageSize_2D,
                                                          FLAGpreserveIntValues = False,
                                                          arrayMask=None,
                                                          intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                          intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                          intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                          )

                     elif self.isChannelBinary_2D[aux_plane_i] == 1:
                         auxImg = preprocessIntensityData(auxImg, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                        imageSize=self.imageSize_2D,
                                                        FLAGpreserveIntValues = True,
                                                        arrayMask=None,
                                                        intensityNormalizationMode = None)
                         auxImg = normalizeLabels(auxImg)

                     batch2D_2[id_i, 0, :, :] = auxImg.copy()


                # 3rd aux plane (sagittal)
                aux_plane_i = 2
                if len(self.allFilenames2Dplane_3_ID)>0:
                     pos_ID_i = np.where(self.listCurrent2Dplane_3_ID == case_ID_i)[0]
                     rnd.shuffle(pos_ID_i)
                     pos_ID_i = pos_ID_i[0]
                     auxImg = self.current_2Dplane_3[pos_ID_i]

                     #1st) apply the corresponding scaling applied to the volume
                     if self.isotropicScaleFLAG:
                         scale_2D_i = np.array([isoScale[0],isoScale[0]])
                     else:
                         scale_2D_i = np.array([isoScale[1],isoScale[0]]) # axial is [1,0]
                     auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = scale_2D_i, rotAngle = 0., transOffset = (0,0))

                     #2nd) if needed, apply additional data augmentation strategies
                     if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):
                         tX,tY,isoScale2D,r = self.generateTransformParams2D()
                         auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = isoScale2D, rotAngle = r, transOffset = (tX,tY))

                     if self.isChannelBinary_2D[aux_plane_i] == 0:
                         auxImg = preprocessIntensityData(auxImg,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                          imageSize=self.imageSize_2D,
                                                          FLAGpreserveIntValues = False,
                                                          arrayMask=None,
                                                          intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                          intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                          intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                          )

                     elif self.isChannelBinary_2D[aux_plane_i] == 1:
                         auxImg = preprocessIntensityData(auxImg, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                        imageSize=self.imageSize_2D,
                                                        FLAGpreserveIntValues = True,
                                                        arrayMask=None,
                                                        intensityNormalizationMode = None)
                         auxImg = normalizeLabels(auxImg)

                     batch2D_3[id_i, 0, :, :] = auxImg.copy()

            return isoScale_ALL, batch, gt, batch2D_1, batch2D_2, batch2D_3

        else:
            raise Exception(self.id + " No images loaded in self.currentVolumes." )


    def generateSingleBatch_VolAnd2DPlanes(self, IDs_i):
        """
            It supposes that the images are already loaded in self.currentChannelImages / self.currentRois / self.currentGt
            / self.batchGen.current_2Dplane_1 / self.batchGen.current_2Dplane_2 / self.batchGen.current_2Dplane_3

            :return: It returns the data and ground truth of a complete batch as data, gt. These structures are theano-compatible with shape:
                        np.ndarray(shape=(self.confTrain['batchSizeTraining'], self.numChannels, dim_1, dim_2, dim_3), dtype=np.float32)
                        It also return the information of the auxiliar 2D planes.

            IDs_i: IDs of the cases to use in the batch.
        """

        if not(self.currentChannelImages == []): # If list of pre-loaded volumes is not empty...

            # define a default scaling factor. The scaling factor is provided as output value.
            if self.isotropicScaleFLAG:
                isoScale_ALL = np.ones([len(IDs_i)])
            else:
                isoScale_ALL = np.ones([len(IDs_i),3])

            # Batch-size Loop
            for id_i in range(0,len(IDs_i)):

                #print('batch-size loop %d %d' %(id_i,len(IDs_i)))
                # Data augmentation Volumes
                # .....................................................................................
                if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):

                    # generate random transformation parameters
                    tX,tY,tZ,isoScale,rX,rY,rZ = self.generateTransformParams()
                    if self.isotropicScaleFLAG:
                        isoScale_ALL[id_i] = isoScale
                    else:
                        isoScale_ALL[id_i,:] = np.array(isoScale)

                    # MASK (apply data augmentation to or auxRoi)
                    if not(self.currentRois == []):
                        auxRoi = self.currentRois[IDs_i[id_i]]
                        rangeRoiValues = np.unique(auxRoi)
                        #auxRoi = applyTransformToVolume(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                        auxRoi = applyTransformToVolumeAnisoScale(auxRoi, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxRoi.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxRoi.min())
                        auxRoi = restoreIntValues (auxRoi, rangeRoiValues)
                        auxRoi = setOutMaskValue(auxRoi, auxRoi, voxelValue = self.bkgrndLabel)
                    else:
                        auxRoi = np.ones((self.currentChannelImages[IDs_i[id_i]][0].shape))

                   # GT
                    auxGt = self.currentGt[IDs_i[id_i]]
                    rangeGTValues = np.unique(auxGt)
                    #auxGt = applyTransformToVolume(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                    auxGt = applyTransformToVolumeAnisoScale(auxGt, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                    auxGt = restoreIntValues (auxGt, rangeGTValues)
                    auxGt = setOutMaskValue(auxGt, auxRoi, voxelValue = self.bkgrndLabel)
                    if auxGt.shape != self.gtSize:
                        auxGt = resizeData(auxGt, self.gtSize, preserveIntValues = True)
                    self.gt[id_i,0,:,:,:] = auxGt.astype(np.int16)


                    # IMG channels
                    for channel in range(self.numChannels):

                        auxImg = self.currentChannelImages[IDs_i[id_i]][channel].copy()

                        if self.isChannelBinary[channel] == 1: #if the channel is binary / or is like the GT (with integer discrete values)
                            rangeIMGValues = np.unique(auxImg)
                            #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                            auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxGt.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxGt.min())
                            auxImg = restoreIntValues (auxImg, rangeIMGValues)
                            auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)
                        else:
                            #auxImg = applyTransformToVolume(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                            auxImg = applyTransformToVolumeAnisoScale(auxImg, rotAngles = (rX,rY,rZ), transOffset = (tX,tY,tZ), scaleFactor = isoScale, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = auxImg.min(), FLAGscalePreserveSize =True, scalePaddingValue = auxImg.min())
                            auxImg = setOutMaskValue(auxImg, auxRoi, voxelValue = self.bkgrndLabel)

                        # Add noise if needed
                        if self.FLAGholesNoise == 1:
                            foreIDs = np.nonzero(auxGt)
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.holesRatio, radiiRange = self.holesRadiiRange)
                            seedsIDs = np.nonzero(volSeeds)
                            auxImg[seedsIDs] = self.bkgrndLabel

                        if self.FLAGsaltpepperNoise == 1:
                            foreIDs = np.nonzero(np.ones(auxGt.shape))
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                            saltIDs = np.nonzero(volSeeds)
                            volSeeds = generateVolSeedsIDs(volSize = self.imageSize, foreIDs = foreIDs, numSeedsRange = self.ratioSaltPepperNoise, radiiRange = self.saltPepperNoiseSizeRange)
                            pepperIDs = np.nonzero(volSeeds)
                            auxImg[saltIDs] = self.intNormParam1[channel]
                            auxImg[pepperIDs] = self.intNormParam2[channel]

                        self.batch[id_i, channel, :, :, :] = auxImg.copy()


                # No data augmentation
                # .....................................................................................
                else:
                    isoScale = (1., 1., 1.)
                    # GT
                    self.gt[id_i, 0, :, :, :] = self.currentGt[IDs_i[id_i]].astype(np.int16)

                    # IMG channels
                    for channel in range(self.numChannels):
                        if self.loadNewFilesEveryEpoch:
                            self.batch[id_i, channel, :, :, :] = self.currentChannelImages[IDs_i[id_i]][channel].copy()
                        else:
                            self.batch[id_i, channel, :, :, :] = self.currentChannelImages[IDs_i[id_i]][channel]


                # Data augmentation 2D planes
                # .....................................................................................
                case_ID_i = self.listCurrentIDs[IDs_i[id_i]]

                # 1st aux plane (coronal)
                aux_plane_i = 0
                if len(self.allFilenames2Dplane_1_ID)>0:
                     pos_ID_i = np.where(self.listCurrent2Dplane_1_ID == case_ID_i)[0]
                     rnd.shuffle(pos_ID_i)
                     pos_ID_i = pos_ID_i[0]
                     #pos_ID_i = rnd.shuffle(np.where(self.listCurrent2Dplane_1_ID == case_ID_i)[0])[0] # pick one randomly
                     auxImg = self.current_2Dplane_1[pos_ID_i]
                     self.auxImg_1 = auxImg
                     #1st) apply the corresponding scaling applied to the volume
                     if self.isotropicScaleFLAG:
                         scale_2D_i = np.array([isoScale[0],isoScale[0]])
                     else:
                         scale_2D_i = np.array([isoScale[2],isoScale[1]]) # coronal is [2,1]
                     auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = scale_2D_i, rotAngle = 0., transOffset = (0,0))

                     #2nd) if needed, apply additional data augmentation strategies
                     if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):
                         tX,tY,isoScale2D,r = self.generateTransformParams2D()
                         auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = isoScale2D, rotAngle = r, transOffset = (tX,tY))

                     if self.isChannelBinary_2D[aux_plane_i] == 0:
                         auxImg = preprocessIntensityData(auxImg,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                          imageSize=self.imageSize_2D,
                                                          FLAGpreserveIntValues = False,
                                                          arrayMask=None,
                                                          intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                          intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                          intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                          )

                     elif self.isChannelBinary_2D[aux_plane_i] == 1:
                         auxImg = preprocessIntensityData(auxImg, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                        imageSize=self.imageSize_2D,
                                                        FLAGpreserveIntValues = True,
                                                        arrayMask=None,
                                                        intensityNormalizationMode = None)
                         auxImg = normalizeLabels(auxImg)

                     self.batch2D_1[id_i, 0, :, :] = auxImg.copy()


                # 2st aux plane (sagittal)
                aux_plane_i = 1
                if len(self.allFilenames2Dplane_2_ID)>0:
                     pos_ID_i = np.where(self.listCurrent2Dplane_2_ID == case_ID_i)[0]
                     rnd.shuffle(pos_ID_i)
                     pos_ID_i = pos_ID_i[0]
                     auxImg = self.current_2Dplane_2[pos_ID_i]
                     self.auxImg_2 = auxImg

                     #1st) apply the corresponding scaling applied to the volume
                     if self.isotropicScaleFLAG:
                         scale_2D_i = np.array([isoScale[0],isoScale[0]])
                     else:
                         scale_2D_i = np.array([isoScale[2],isoScale[0]]) # sagittal is [2,0]

                     auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = scale_2D_i, rotAngle = 0., transOffset = (0,0))

                     #2nd) if needed, apply additional data augmentation strategies
                     if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):
                         tX,tY,isoScale2D,r = self.generateTransformParams2D()
                         auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = isoScale2D, rotAngle = r, transOffset = (tX,tY))

                     if self.isChannelBinary_2D[aux_plane_i] == 0:
                         auxImg = preprocessIntensityData(auxImg,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                          imageSize=self.imageSize_2D,
                                                          FLAGpreserveIntValues = False,
                                                          arrayMask=None,
                                                          intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                          intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                          intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                          )

                     elif self.isChannelBinary_2D[aux_plane_i] == 1:
                         auxImg = preprocessIntensityData(auxImg, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                        imageSize=self.imageSize_2D,
                                                        FLAGpreserveIntValues = True,
                                                        arrayMask=None,
                                                        intensityNormalizationMode = None)
                         auxImg = normalizeLabels(auxImg)

                     self.batch2D_2[id_i, 0, :, :] = auxImg.copy()


                # 3rd aux plane (sagittal)
                aux_plane_i = 2
                if len(self.allFilenames2Dplane_3_ID)>0:
                     pos_ID_i = np.where(self.listCurrent2Dplane_3_ID == case_ID_i)[0]
                     rnd.shuffle(pos_ID_i)
                     pos_ID_i = pos_ID_i[0]
                     auxImg = self.current_2Dplane_3[pos_ID_i]

                     #1st) apply the corresponding scaling applied to the volume
                     if self.isotropicScaleFLAG:
                         scale_2D_i = np.array([isoScale[0],isoScale[0]])
                     else:
                         scale_2D_i = np.array([isoScale[1],isoScale[0]]) # axial is [1,0]
                     auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = scale_2D_i, rotAngle = 0., transOffset = (0,0))

                     #2nd) if needed, apply additional data augmentation strategies
                     if (np.random.uniform(size=1)[0] < self.dataAugmentationRate):
                         tX,tY,isoScale2D,r = self.generateTransformParams2D()
                         auxImg= applyTransformTo2Dplane(auxImg, scaleFactor = isoScale2D, rotAngle = r, transOffset = (tX,tY))

                     if self.isChannelBinary_2D[aux_plane_i] == 0:
                         auxImg = preprocessIntensityData(auxImg,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                          imageSize=self.imageSize_2D,
                                                          FLAGpreserveIntValues = False,
                                                          arrayMask=None,
                                                          intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                          intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                          intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                          )

                     elif self.isChannelBinary_2D[aux_plane_i] == 1:
                         auxImg = preprocessIntensityData(auxImg, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                        imageSize=self.imageSize_2D,
                                                        FLAGpreserveIntValues = True,
                                                        arrayMask=None,
                                                        intensityNormalizationMode = None)
                         auxImg = normalizeLabels(auxImg)

                     self.batch2D_3[id_i, 0, :, :] = auxImg.copy()

            return isoScale_ALL

        else:
            raise Exception(self.id + " No images loaded in self.currentVolumes." )



    def generateBatchesForOneEpoch(self):
        """
            Volumes + 2D aux planes.
            Reachable from 'generateBatches' method.
            generateBatches() -> _generateBatches -> generateBatchesForOneEpoch
        """
        # Load the files if needed
        # =====================================================================
        if (self.currentEpoch == 0) or ((self.currentEpoch > 0) and self.loadNewFilesEveryEpoch):

            printMessageVerb(self.FLAGverbose, '-->> Epoch %d: LOADING IMAGES... '% (self.currentEpoch))

            # Choose the random images that will be sampled in this epoch
            # (*) ToDo -> make sure that new samples are used every epoch.
            self.indexCurrentImages = np.array(self.rndSequence.sample(range(0,self.numFiles), self.numOfCasesLoadedPerEpoch)) #it needs to be a numpy array so we can extract multiple elements with a list (the elements defined by IDsCases, which will be shuffled each epoch)
            self.IDsCases = list(range(0,self.numOfCasesLoadedPerEpoch)) #IDs to the cases in self.indexCurrentImages

            printMessageVerb(self.FLAGverbose, "Loading %d images for epoch %d" % (len(self.indexCurrentImages), self.currentEpoch))
            logging.debug(self.id + " Loading images number : %s" % self.indexCurrentImages )

            self.batchesPerEpoch = int(np.floor(self.numFiles / self.batchSize)) # number of batches per epoch

            self.currentChannelImages = []  # reset the list of images volumes (actual volumes)
            self.currentGt = []             # reset the list of gt volumes (actual volumes)
            self.currentRois = []           # reset the list of ROIs volumes (actual volumes)
            self.listCurrentFiles = []          # reset the list of files loaded per epoch.
            self.listCurrentIDs = np.array([])            # reset the list of IDs

            self.listCurrent2Dplane_1 = []      # reset the list of 2D planes loaded per epoch.
            self.listCurrent2Dplane_1_ID = np.array([])   # reset the list of IDs of 2D planes loaded per epoch.
            self.current_2Dplane_1 = []     # reset the auxiliar 2D planes 1 (actual images)

            self.listCurrent2Dplane_2 = []      # reset the list of 2D planes loaded per epoch.
            self.listCurrent2Dplane_2_ID = np.array([])   # reset the list of IDs of 2D planes loaded per epoch.
            self.current_2Dplane_2 = []     # reset the auxiliar 2D planes 2 (actual images)

            self.listCurrent2Dplane_3 = []      # reset the list of 2D planes loaded per epoch.
            self.listCurrent2Dplane_3_ID = np.array([])   # reset the list of IDs of 2D planes loaded per epoch.
            self.current_2Dplane_3 = []     # reset the auxiliar 2D planes 3 (actual images)

            for realImageIndex in self.indexCurrentImages:
                 loadedImageChannels = [] #list to store all the channels of the current image.
                 self.listCurrentFiles.append(self.allChannelsFilenames[0][realImageIndex])   # List of filenames in the order
                 #self.listCurrentIDs.append(self.allFilenamesIDs[realImageIndex])             # Real patient ID of each case. Used to match the image with the auuxiliar 2D planes
                 case_ID_i = self.allFilenamesIDs[realImageIndex]
                 self.listCurrentIDs = np.append(self.listCurrentIDs, case_ID_i)
                 printMessageVerb(self.FLAGverbose, '-->> loading %d - ID %d'% (realImageIndex,case_ID_i))

                 # Load ROI if exists
                 if ('roiMasksTraining' in self.confTrain):
                     roi = nib.load(self.roiFilenames[realImageIndex]).get_data()
                     roi = preprocessIntensityData(roi, FLAGresizeImages=self.FLAGresizeImages, imageSize=self.imageSize, FLAGpreserveIntValues = True, arrayMask=[], intensityNormalizationMode=None)
                     self.currentRois.append(roi)
                 else:
                     roi = None

                 # Imgs channels ----------------------------------------------
                 # (*) ToDo --> incorporate the masks in the image normalization stage.!!
                 for channel in range(0, self.numChannels):
                      # Load the corresponding image for the corresponding channel and append it to the list of channels
                      # for the current imageIndex
                      # loadedImageChannels.append(nib.load(self.allChannelsFilenames[channel][realImageIndex]).get_data())

                      # Load, preprocess, and normalize the channel.
                      dataIn = nib.load(self.allChannelsFilenames[channel][realImageIndex]).get_data()
                      if self.isChannelBinary[channel] == 0: # not binary input
                          dataIn = preprocessIntensityData(dataIn,FLAGresizeImages=self.FLAGresizeImages,
                                    imageSize=self.imageSize,
                                    FLAGpreserveIntValues = False,
                                    arrayMask=None,
                                    intensityNormalizationMode=self.intensityNormalizationMode[channel],
                                    intNormParam1=self.intNormParam1[channel],
                                    intNormParam2=self.intNormParam2[channel]
                                    )
                          if self.FLAGsetBkgrnd == True:
                              dataIn = setOutMaskValue(dataIn, roi, voxelValue = self.bkgrndLabel)

                      elif self.isChannelBinary[channel] == 1: # binary input --> treat as gt.
                          dataIn = preprocessIntensityData(dataIn, FLAGresizeImages=self.FLAGresizeImages,
                                    imageSize=self.imageSize,
                                    FLAGpreserveIntValues = True,
                                    arrayMask=None,
                                    intensityNormalizationMode = None)
                          dataIn = normalizeLabels(dataIn)

                      # Add the image to the queue of channels
                      loadedImageChannels.append(dataIn)

                      # Check that all the channels have the same dimensions
                      if channel > 0:
                          assert loadedImageChannels[channel].shape == loadedImageChannels[0].shape, self.id + " Data size incompatibility when loading image channels for volume %s" % self.allChannelsFilenames[channel][realImageIndex]

                 # Append all the channels of the image to the list
                 self.currentChannelImages.append(loadedImageChannels)

                 # GT channel ----------------------------------------------
                 gt = nib.load(self.gtFilenames[realImageIndex]).get_data()
                 gt = preprocessIntensityData(gt, FLAGresizeImages=self.FLAGresizeImages, imageSize=self.imageSize, FLAGpreserveIntValues = True, arrayMask=[], intensityNormalizationMode=None)
                 gt = normalizeLabels(gt)
                 #assert gt.shape == loadedImageChannels[0].shape, self.id + " Data size incompatibility when loading GT %s" % self.gtFilenames[realImageIndex]
                 self.currentGt.append(gt)


#                 # Aux 2D planes  ---------------------------------------------

                 # Additional 2D image 1
                 aux_plane_i = 0
                 if len(self.allFilenames2Dplane_1_ID)>0:
                     pos_ID_i = np.where(self.allFilenames2Dplane_1_ID == case_ID_i)[0]
                     assert len(pos_ID_i)>0, self.id + " 2D plane 1 missing for case ID %d" % case_ID_i
                     if len(pos_ID_i)>0:
                         for p_i in pos_ID_i:
                             dataIn = nib.load(self.allFilenames2Dplane_1[p_i]).get_data()[:,:,0]
                             if self.isChannelBinary_2D[aux_plane_i] == 0: # not binary input

                                 dataIn = preprocessIntensityData(dataIn,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                  imageSize=self.imageSize_2D,
                                                                  FLAGpreserveIntValues = False,
                                                                  arrayMask=None,
                                                                  intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                                  intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                                  intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                                  )
                             elif self.isChannelBinary_2D[aux_plane_i] == 1:
                                 dataIn = preprocessIntensityData(dataIn, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                imageSize=self.imageSize_2D,
                                                                FLAGpreserveIntValues = True,
                                                                arrayMask=None,
                                                                intensityNormalizationMode = None)
                                 dataIn = normalizeLabels(dataIn)

                             # Append all the channels of the image to the list
                             self.current_2Dplane_1.append(dataIn)
                             self.listCurrent2Dplane_1.append(self.allFilenames2Dplane_1[p_i])
                             self.listCurrent2Dplane_1_ID = np.append(self.listCurrent2Dplane_1_ID,case_ID_i)
                             printMessageVerb(self.FLAGverbose, '-->>2D plane 1 found: %s'% (self.allFilenames2Dplane_1[p_i]))

                 # Additional 2D image 2
                 aux_plane_i = 1
                 if len(self.allFilenames2Dplane_2_ID)>0:
                     pos_ID_i = np.where(self.allFilenames2Dplane_2_ID == case_ID_i)[0]
                     assert len(pos_ID_i)>0, self.id + " 2D plane 2 missing for case ID %d" % case_ID_i
                     if len(pos_ID_i)>0:
                         for p_i in pos_ID_i:
                             dataIn = nib.load(self.allFilenames2Dplane_2[p_i]).get_data()[:,:,0]
                             if self.isChannelBinary_2D[aux_plane_i] == 0: # not binary input
                                 dataIn = preprocessIntensityData(dataIn,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                  imageSize=self.imageSize_2D,
                                                                  FLAGpreserveIntValues = False,
                                                                  arrayMask=None,
                                                                  intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                                  intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                                  intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                                  )
                             elif self.isChannelBinary_2D[aux_plane_i] == 1:
                                 dataIn = preprocessIntensityData(dataIn, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                imageSize=self.imageSize_2D,
                                                                FLAGpreserveIntValues = True,
                                                                arrayMask=None,
                                                                intensityNormalizationMode = None)
                                 dataIn = normalizeLabels(dataIn)

                             # Append all the channels of the image to the list
                             self.current_2Dplane_2.append(dataIn)
                             self.listCurrent2Dplane_2.append(self.allFilenames2Dplane_2[p_i])
                             self.listCurrent2Dplane_2_ID = np.append(self.listCurrent2Dplane_2_ID,case_ID_i)
                             printMessageVerb(self.FLAGverbose, '-->>2D plane 2 found: %s'% (self.allFilenames2Dplane_2[p_i]))

                 # Additional 2D image 3
                 aux_plane_i = 2
                 if len(self.allFilenames2Dplane_3_ID)>0:
                     pos_ID_i = np.where(self.allFilenames2Dplane_3_ID == case_ID_i)[0]
                     assert len(pos_ID_i)>0, self.id + " 2D plane 2 missing for case ID %d" % case_ID_i
                     if len(pos_ID_i)>0:
                         for p_i in pos_ID_i:
                             dataIn = nib.load(self.allFilenames2Dplane_3[p_i]).get_data()[:,:,0]
                             if self.isChannelBinary_2D[aux_plane_i] == 0: # not binary input
                                 dataIn = preprocessIntensityData(dataIn,FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                  imageSize=self.imageSize_2D,
                                                                  FLAGpreserveIntValues = False,
                                                                  arrayMask=None,
                                                                  intensityNormalizationMode=self.intensityNormalizationMode_2D[aux_plane_i],
                                                                  intNormParam1=self.intNormParam1_2D[aux_plane_i],
                                                                  intNormParam2=self.intNormParam2_2D[aux_plane_i]
                                                                  )
                             elif self.isChannelBinary_2D[aux_plane_i] == 1:
                                 dataIn = preprocessIntensityData(dataIn, FLAGresizeImages=self.FLAGresizeImages_2D,
                                                                imageSize=self.imageSize_2D,
                                                                FLAGpreserveIntValues = True,
                                                                arrayMask=None,
                                                                intensityNormalizationMode = None)
                                 dataIn = normalizeLabels(dataIn)

                             # Append all the channels of the image to the list
                             self.current_2Dplane_3.append(dataIn)
                             self.listCurrent2Dplane_3.append(self.allFilenames2Dplane_2[p_i])
                             self.listCurrent2Dplane_3_ID = np.append(self.listCurrent2Dplane_3_ID,case_ID_i)
                             printMessageVerb(self.FLAGverbose, '-->>2D plane 3 found: %s'% (self.allFilenames2Dplane_3[p_i]))



            # Initialize the batch and gt variables (so we only need to declare them once)
            if self.FLAGresizeImages ==1:
                self.batch = np.ndarray(shape=(self.batchSize, self.numChannels, self.imageSize[0], self.imageSize[1], self.imageSize[2]), dtype=np.float32)
                self.batch2D_1 = np.ndarray(shape=(self.batchSize, 1, self.imageSize_2D[0], self.imageSize_2D[1]), dtype=np.float32)
                self.batch2D_2 = np.ndarray(shape=(self.batchSize, 1, self.imageSize_2D[0], self.imageSize_2D[1]), dtype=np.float32)
                self.batch2D_3 = np.ndarray(shape=(self.batchSize, 1, self.imageSize_2D[0], self.imageSize_2D[1]), dtype=np.float32)
                #gt =    np.ndarray(shape=(1, self.numClasses, self.gtSize[0], self.gtSize[1], self.gtSize[2]),  dtype=np.float32)
                self.gt = np.ndarray(shape=(self.batchSize, 1, self.gtSize[0], self.gtSize[1], self.gtSize[2]),  dtype=np.float32)
            else:
                dims = self.currentChannelImages[0][0].shape
                self.batch = np.ndarray(shape=(self.batchSize, self.numChannels, dims[0], dims[1], dims[2]), dtype=np.float32)
                #gt =    np.ndarray(shape=(1, self.numClasses, dims[0], dims[1], dims[2]),  dtype=np.float32)
                dims_2D = self.current_2Dplane_1[1].shape
                self.batch2D_1 = np.ndarray(shape=(self.batchSize, 1, dims_2D[0], dims_2D[1]), dtype=np.float32)
                self.batch2D_2 = np.ndarray(shape=(self.batchSize, 1, dims_2D[0], dims_2D[1]), dtype=np.float32)
                self.batch2D_3 = np.ndarray(shape=(self.batchSize, 1, dims_2D[0], dims_2D[1]), dtype=np.float32)
                self.gt = np.ndarray(shape=(self.batchSize, 1, dims[0], dims[1], dims[2]),  dtype=np.float32)
        else:
            # reshuffle the IDs of the cases so each epoch, the cases are presented in different order.
            rnd.shuffle(self.IDsCases)


        # Create the batches
        # =====================================================================
        printMessageVerb(self.FLAGverbose, '-->> Creating batches .................................')
        t_1 = time.time()

        # 1) Split the batches between all the threads
        #   i) split the number of batches

        batchesPerEpochPerThread = [self.batchesPerEpoch // self.numThreads] * self.numThreads
        batchesPerEpochPerThread[0] += self.batchesPerEpoch % self.numThreads #batches per thread
        self.IDsCases_Threads = [self.IDsCases[sum(batchesPerEpochPerThread[0:x])*self.batchSize:sum(batchesPerEpochPerThread[0:x+1])*self.batchSize] for x in range(1, len(batchesPerEpochPerThread))]
        self.IDsCases_Threads.append(self.IDsCases[0:batchesPerEpochPerThread[0]*self.batchSize]) #list of list with the IDs to use on each epoch

        listBatchGenTh = []
        for th_i in range(0,self.numThreads):

            batchGenTh_i = Thread(target = self.batchGeneratorThread, args=(th_i, self.IDsCases_Threads[th_i]))
            batchGenTh_i.setDaemon(False)
            batchGenTh_i.start()
            listBatchGenTh.append(batchGenTh_i)

        for th_i in range(0,self.numThreads): # Necessary to prevent collapsing
            listBatchGenTh[th_i].join()


        t_2 = time.time()
        self.timePerEpoch = np.append(self.timePerEpoch,t_2-t_1)
        t_1 = t_2

            # Lunch a thread with its corresponding list of IDs  IDsCases_Threads[th_i]

            #IDs_aux_i = self.IDsCases[(numbatch*self.batchSize):((numbatch*self.batchSize)+self.batchSize)] #sample from the shuffled list of IDs (self.IDsCases)
            #IDs_i = self.indexCurrentImages[IDs_aux_i] # recover the real IDs of the images to work with.
            #isoScale_i = self.generateSingleBatch_VolAnd2DPlanes(IDs_i)




        # 2) Each thread has to take care of a certain number of batches

        #   - check the new multi-thread function 'generateSingleBatch_VolAnd2DPlanes' doesn't change or modify common / global resurces.
        #   - control simultaneous access to common variables (use a lock...)
        #       - put the batch and filenames in the queue


#        for numbatch in range(0, self.batchesPerEpoch):
#            #print "generating batch %d" % (batch)
#            #logging.debug(self.id + " Generating batch: %d" % batch )
#            #print('generating batch %d' %(numbatch))
#            IDs_aux_i = self.IDsCases[(numbatch*self.batchSize):((numbatch*self.batchSize)+self.batchSize)] #sample from the shuffled list of IDs (self.IDsCases)
#            IDs_i = self.indexCurrentImages[IDs_aux_i] # recover the real IDs of the images to work with.
#            #            self.isoScale_i = self.generateSingleBatchV2(IDs_i)
#            isoScale_i = self.generateSingleBatch_VolAnd2DPlanes(IDs_i)
#
##            self.aux = IDs_i
##            #self.batch, self.gt, isoScale_i = self.generateSingleBatch(batch)
#
#            self.queue.put((self.batch,self.gt, self.batch2D_1, self.batch2D_2, self.batch2D_3))
#            self.queueScalingFactors.put(isoScale_i)
#            currentFilesNames = [];
#            for IDs_j in IDs_i:
#                currentFilesNames.append(self.listCurrentFiles[IDs_j])
#            self.queueFileNames.put(currentFilesNames)

#            t_2 = time.time()
#            self.timePerEpoch = np.append(self.timePerEpoch,t_2-t_1)
#            t_1 = t_2
#         # Unload the files if we are loading new files every subpeoc
        if self.loadNewFilesEveryEpoch:
            self.unloadFiles()
