

import numpy as np

def generateSphericStruct(strRadii = 3):
    
    strRadii_ = int(strRadii)
    x_ = np.linspace(-1*strRadii_, strRadii_, 2*strRadii+1)
    y_ = np.linspace(-1*strRadii_, strRadii_, 2*strRadii+1)
    z_ = np.linspace(-1*strRadii_, strRadii_, 2*strRadii+1)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    distMat = np.sqrt(x**2 + y**2 + z**2)
    strBin = np.zeros(distMat.shape)
    strBin[distMat <= strRadii] = 1
    return strBin


def generateVolSeeds(volSize = (96,96,96), numSeedsRange = (5,10), radiiRange = (2,7)):
    volSeeds = np.zeros(volSize)
    numSeeds = int(np.random.uniform(low = numSeedsRange[0], high = numSeedsRange[1], size = 1))
    for s_i in range(0,numSeeds):
        radii_i = int(np.random.uniform(low = radiiRange[0], high = radiiRange[1]))
        cent_x =int(np.random.uniform(low = 0, high = volSize[0]))
        cent_y =int(np.random.uniform(low = 0, high = volSize[1]))
        cent_z =int(np.random.uniform(low = 0, high = volSize[2]))
        seed_i = generateSphericStruct(strRadii = radii_i)
        seedL = seed_i.shape
        if (seedL[0]%2 == 0): L_i = seedL[0]/2; R_i = (seedL[0]/2)-1; C_i = L_i+1
        else: L_i = (seedL[0]-1)/2; R_i = (seedL[0]-1)/2
        C_i = L_i+1;
        range_x = (max(0,cent_x-L_i), min(volSize[0]-1, cent_x + R_i))
        range_y = (max(0,cent_y-L_i), min(volSize[1]-1, cent_y + R_i))
        range_z = (max(0,cent_z-L_i), min(volSize[2]-1, cent_z + R_i))        
        SLx = C_i - L_i + abs(min(0, cent_x - L_i)) - 1
        SRx = C_i + L_i - abs(min(0, (volSize[0]-1) - (cent_x + R_i)))-1
        SLy = C_i - L_i + abs(min(0, cent_y - L_i)) - 1
        SRy = C_i + L_i - abs(min(0, (volSize[0]-1) - (cent_y + R_i)))-1
        SLz = C_i - L_i + abs(min(0, cent_z - L_i)) - 1
        SRz = C_i + L_i - abs(min(0, (volSize[0]-1) - (cent_z + R_i)))-1
        volSeeds[range_x[0]:range_x[1]+1,range_y[0]:range_y[1]+1,range_z[0]:range_z[1]+1]=seed_i[SLx:SRx+1,SLy:SRy+1,SLz:SRz+1]
    return volSeeds
        #generateNiiFileFromData(volSeeds, fileName = 'temp_1.nii')

def generateVolSeedsIDs(volSize = (96,96,96), foreIDs = (range(0,96**3),range(0,96**3),range(0,96**3)), numSeedsRange = (5,10), radiiRange = (2,7)):
    volSeeds = np.zeros(volSize)
    numSeeds = int(np.random.uniform(low = numSeedsRange[0], high = numSeedsRange[1], size = 1))
    numFore = len(foreIDs[0])
    for s_i in range(0,numSeeds):
        radii_i = int(np.random.uniform(low = radiiRange[0], high = radiiRange[1]))
        pos_id = np.random.randint(low=0,high=numFore-1)
        cent_x =foreIDs[0][pos_id]
        cent_y =foreIDs[1][pos_id]
        cent_z =foreIDs[2][pos_id]
        seed_i = generateSphericStruct(strRadii = radii_i)
        seedL = seed_i.shape
        if (seedL[0]%2 == 0): L_i = seedL[0]/2; R_i = (seedL[0]/2)-1; C_i = L_i+1
        else: L_i = (seedL[0]-1)/2; R_i = (seedL[0]-1)/2
        C_i = L_i+1;
        range_x = (max(0,cent_x-L_i), min(volSize[0]-1, cent_x + R_i))
        range_y = (max(0,cent_y-L_i), min(volSize[1]-1, cent_y + R_i))
        range_z = (max(0,cent_z-L_i), min(volSize[2]-1, cent_z + R_i))        
        SLx = C_i - L_i + abs(min(0, cent_x - L_i)) - 1
        SRx = C_i + L_i - abs(min(0, (volSize[0]-1) - (cent_x + R_i)))-1
        SLy = C_i - L_i + abs(min(0, cent_y - L_i)) - 1
        SRy = C_i + L_i - abs(min(0, (volSize[0]-1) - (cent_y + R_i)))-1
        SLz = C_i - L_i + abs(min(0, cent_z - L_i)) - 1
        SRz = C_i + L_i - abs(min(0, (volSize[0]-1) - (cent_z + R_i)))-1
        volSeeds[range_x[0]:range_x[1]+1,range_y[0]:range_y[1]+1,range_z[0]:range_z[1]+1]=seed_i[SLx:SRx+1,SLy:SRy+1,SLz:SRz+1]
    return volSeeds
        #generateNiiFileFromData(volSeeds, fileName = 'temp_1.nii')
        
def createNoisiVolume(vIn = np.zeros((96,96,96)), radiiRange = (2,4), numSeedsRange = (5,10)):
    foreIDs = np.nonzero(vIn)
    volSeeds = generateVolSeedsIDs(volSize = vIn.shape, foreIDs = foreIDs, numSeedsRange = numSeedsRange, radiiRange = radiiRange)
    seedsIDs = np.nonzero(volSeeds)
    dataNoise = np.zeros(vIn.shape)
    dataNoise[foreIDs] = 1     
    dataNoise[seedsIDs] = 0
    return dataNoise



    
    