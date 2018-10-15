# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:01:29 2018

@author: jcerrola
"""

##%% TESTING CLASS BINARY BATCH GENERATOR WITH US CUTS
##==============================================================================
import sys
sys.path.append("/homes/wt814/IndividualProject/code/")
import logging

import imp
import BatchGenerator_v2
imp.reload(BatchGenerator_v2)
from BatchGenerator_v2 import *
from GenerateNiftiFilesFromData import *

# LOADING CONFIG FILES
# -----------------------------------------------------------------------------
confTrain = {}
print("start")
if sys.version_info[0] < 3:
    execfile("/homes/wt814/IndividualProject/code/BatchGeneratorClass/BatchGenerator_v2_configTemp_v2.cfg", confTrain)
else:
    exec(open("/homes/wt814/IndividualProject/code/BatchGeneratorClass/BatchGenerator_v2_configTemp_v2.cfg").read(),confTrain)


## -----------------------------------------------------------------------------
## Create the batch generator

#batchGen = BatchGeneratorVolAnd2Dplanes(confTrain, mode='training', infiniteLoop=False, maxQueueSize = 5)

# batchGen = BatchGeneratorVolAnd2DplanesMultiThread(confTrain, mode='training', infiniteLoop=False, maxQueueSize = 1)


# batchGen.generateBatches()
#
# data_i, gt_i, plane_2D_1, plane_2D_2, plane_2D_3 = batchGen.getBatchAndScalingFactor()
# print(plane_2D_2.shape, "hello")
# print(plane_2D_1.shape, "hi")
# print(plane_2D_3.shape, "working")
# num = batchGen.getNumBatchesInQueue()
# print(num)
# batchGen.finish()


batchGen = BatchGeneratorVolAnd2Dplanes(confTrain, mode='training', infiniteLoop=False, maxQueueSize = 5)
batchGen.generateBatches()

for e in range(0, confTrain['numEpochs']):

    batchesPerSubepoch = confTrain['batchSizeTraining']

    for bps in range(0, batchesPerSubepoch):
        print("threadlocked")
        batch = batchGen.getBatchAndScalingFactor()
        print("hope not")
        #updateCNN(batch)
print("threadlocked?")
print(batchGen.emptyQueue())
print("threadlocked?2")
assert batchGen.emptyQueue() , "The training loop finished before the queue is empty"
batchGen.finish()
