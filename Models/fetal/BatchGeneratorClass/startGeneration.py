import sys
sys.path.append("/homes/wt814/IndividualProject/code/")
import logging

import imp
import BatchGenerator_v2
import threading
imp.reload(BatchGenerator_v2)
from BatchGenerator_v2 import *
from GenerateNiftiFilesFromData import *



def startBatch(epochs, batch_size, fold_no):
    confTrain = {}
    print("start")
    if sys.version_info[0] < 3:
        execfile("/homes/wt814/IndividualProject/code/BatchGeneratorClass/BatchGenerator_v2_config_fold_%d.cfg" %(fold_no), confTrain)
    else:
        exec(open("/homes/wt814/IndividualProject/code/BatchGeneratorClass/BatchGenerator_v2_config_fold_%d.cfg" %(fold_no)).read(),confTrain)

    try:
        epochs <= confTrain['numEpochs']
    except:
        sys.exit(1)

    try:
        batch_size == confTrain['batchSizeTraining']
    except:
        sys.exit(1)

    batchGen = BatchGeneratorVolAnd2DplanesMultiThread(confTrain, mode='training', infiniteLoop=False, maxQueueSize = 4)
    #batchGenV = BatchGeneratorVolAnd2DplanesMultiThread(confTrain, mode='validation', infiniteLoop=False, maxQueueSize = 4)

    batchGen.generateBatches()
    #batchGenV.generateBatches()
    for i in range(epochs):
        for batch_no in range(1):
            data_i, gt_i, plane_2D_1, plane_2D_2, plane_2D_3 = batchGen.getBatchAndScalingFactor()
            print("Epoch %d, Batch No: %d" % (i , batch_no) )
            np.save("./Plane1.npy", plane_2D_1)
            np.save("./Plane2.npy", plane_2D_2)
            np.save("./Plane3.npy", plane_2D_3)

        #for batch_n in range(30):
                #data_i, gt_i, plane_2D_1, plane_2D_2, plane_2D_3 = batchGenV.getBatchAndScalingFactor()
                #print("Val %d, Batch No: %d" % (i , batch_n) )

    #return data_i, gt_i, plane_2D_1, plane_2D_2, plane_2D_3
    print("done")
    batchGen.finish()
    #batchGenV.finish()
if __name__ == '__main__':
    #for i in range (3):
    startBatch(1, 2, fold_no=1)
