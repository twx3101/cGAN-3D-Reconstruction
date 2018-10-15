# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:45:48 2017

@author: jcerrola
"""

import cv2
import numpy as np
import scipy
from skimage.transform import resize


def restoreIntValues (imgIn, rangeValues):
    # Function to restore the original range of values of an nDarray. 
    # Typically, additional 'fake' intensity values are generated after many image transformations,
    # due to the use of interpolation methods (e.g., rotation, scaling, etc). 
    # This funciton allows to restore the original set of discrete intensity values.
    #
    # Input: 
    #  - imgIn: 3D input ndarray
    #  - rangeValues: 1D ndarray with the oritinal list of discrete values
    imgOut = np.around(imgIn)
    interLabelsTh = [ (x + rangeValues[i - 1])/2. for i, x in enumerate(rangeValues)][1:]
    LOWlimit = [imgOut.min()-1] + interLabelsTh
    UPlimit = interLabelsTh + [imgOut.max()+1]
    for label_i in range(0,len(rangeValues)):
        imgOut[np.multiply(UPlimit[label_i] > imgOut, imgOut >= LOWlimit[label_i])] = rangeValues[label_i]
    return imgOut


def translate3Dvolume(imgIn, transOffset = (0,0,0), paddingValue = 0 ):
    # Translation of a 3D volume through the three axes.
    # Input: 
    #  - imgIn: 3D ndarray
    #  - translateOffset: list of translation in the three axes.
    #  - paddingValue: value for padding
    
    #transOffset = map(int,transOffset)
    imgOut = imgIn.copy()
    padLx,padRx,padLy,padRy,padLz,padRz = (0,0,0,0,0,0)
    
    # X axis)
    if transOffset[0]>0:
        imgOut = imgOut[transOffset[0]:,:,:]
        padLx = abs(transOffset[0])
        #imgOut = np.lib.pad(imgOut,((pad_i, 0),(0, 0),(0,0)),'constant', constant_values=paddingValue)
    elif transOffset[0]<0:
        imgOut = imgOut[:transOffset[0],:,:]
        padRx = abs(transOffset[0])
        #imgOut = np.lib.pad(imgOut,((pad_i, 0),(0, 0),(0,0)),'constant', constant_values=paddingValue)

    # Y axis)
    if transOffset[1]>0:
        imgOut = imgOut[:,transOffset[1]:,:]
        padLy = abs(transOffset[1])
        #imgOut = np.lib.pad(imgOut,((pad_i, 0),(0, 0),(0,0)),'constant', constant_values=paddingValue)
    elif transOffset[1]<0:
        imgOut = imgOut[:,:transOffset[1],:]
        padRy = abs(transOffset[1])

    # Z axis)       
    if transOffset[2]>0:
        imgOut = imgOut[:,:,transOffset[2]:]
        padLz = abs(transOffset[2])
        #imgOut = np.lib.pad(imgOut,((pad_i, 0),(0, 0),(0,0)),'constant', constant_values=paddingValue)
    elif transOffset[2]<0:
        imgOut = imgOut[:,:,:transOffset[2]]
        padRz = abs(transOffset[2])
         
    imgOut = np.lib.pad(imgOut,((padLx, padRx),(padLy, padRy),(padLz,padRz)),'constant', constant_values=paddingValue)   
    return imgOut
    
    

def scale3DVolume(imgIn, scaleFactor = 1, FLAG_sameSize = True, paddingValue = 0):

    # 1.- Resize image
    shapeOut = list(map(int, np.multiply(imgIn.shape,scaleFactor)))
    #shapeOut = int(np.multiply(imgIn.shape,scaleFactor))
    imgOut = (imgIn - imgIn.min()) / (imgIn.max()-imgIn.min()) # values need to be between 0 and 1 to use the resize function.
    imgOut = resize(imgOut,shapeOut, preserve_range=True, mode = 'reflect')
    imgOut = (imgOut*(imgIn.max()-imgIn.min()))+imgIn.min() # restore the original range of values.

    if FLAG_sameSize:
        # 2.- Crop/Padd
        diffSizes = list(map(int, tuple(abs(x-y) for x,y in zip(imgIn.shape,imgOut.shape))))
        diffSizesLeft = list(map(int, np.divide(diffSizes,2)))
        diffSizesRight = tuple(x-y for x,y in zip(diffSizes,diffSizesLeft))
        
        if scaleFactor > 1: # crop image
            imgOut = imgOut[diffSizesLeft[0]:(imgOut.shape[0]-diffSizesRight[0]),
                            diffSizesLeft[1]:(imgOut.shape[1]-diffSizesRight[1]),
                            diffSizesLeft[2]:(imgOut.shape[2]-diffSizesRight[2])]
        elif scaleFactor < 1: # padd imag
            imgOut = np.lib.pad(imgOut,((diffSizesLeft[0], diffSizesRight[0]),
                                        (diffSizesLeft[1], diffSizesRight[1]),
                                        (diffSizesLeft[2], diffSizesRight[2])),'constant', constant_values=paddingValue)   
    return imgOut 


def anisoScale3DVolume(imgIn, scaleFactor = (1.0, 1.0, 1.0), FLAG_sameSize = True, paddingValue = 0):
    # scaleFactor = (1.2, 1.0, 0.5)
    # 1.- Resize image
    shapeOut = list(map(int, np.multiply(imgIn.shape,scaleFactor)))
    #shapeOut = int(np.multiply(imgIn.shape,scaleFactor))
    imgOut = (imgIn - imgIn.min()) / (imgIn.max()-imgIn.min()) # values need to be between 0 and 1 to use the resize function.
    imgOut = resize(imgOut,shapeOut, preserve_range=True, mode = 'reflect')
    imgOut = (imgOut*(imgIn.max()-imgIn.min()))+imgIn.min() # restore the original range of values.

    if FLAG_sameSize:
        # 2.- Crop/Padd
        diffSizes = list(map(int, tuple(abs(x-y) for x,y in zip(imgIn.shape,imgOut.shape))))
        diffSizesLeft = list(map(int, np.divide(diffSizes,2)))
        diffSizesRight = tuple(x-y for x,y in zip(diffSizes,diffSizesLeft))
        
        #        if scaleFactor > 1: # crop image
        #            imgOut = imgOut[diffSizesLeft[0]:(imgOut.shape[0]-diffSizesRight[0]),
        #                            diffSizesLeft[1]:(imgOut.shape[1]-diffSizesRight[1]),
        #                            diffSizesLeft[2]:(imgOut.shape[2]-diffSizesRight[2])]
        #        elif scaleFactor < 1: # padd imag
        #            imgOut = np.lib.pad(imgOut,((diffSizesLeft[0], diffSizesRight[0]),
        #                                        (diffSizesLeft[1], diffSizesRight[1]),
        #                                        (diffSizesLeft[2], diffSizesRight[2])),'constant', constant_values=paddingValue)   
            
        imgOut_aux = np.ones(imgIn.shape)*paddingValue
        # x_scale
        if scaleFactor[0]<1:
            outXrange = (diffSizesLeft[0],(imgOut_aux.shape[0]-diffSizesRight[0]))
            inXrange = (0,imgOut.shape[0])
        else:
            inXrange = (diffSizesLeft[0],(imgOut.shape[0]-diffSizesRight[0]))
            outXrange = (0,imgOut_aux.shape[0])
        # y_scale
        if scaleFactor[1]<1:
            outYrange = (diffSizesLeft[1],(imgOut_aux.shape[1]-diffSizesRight[1]))
            inYrange = (0,imgOut.shape[1])
        else:
            inYrange = (diffSizesLeft[1],(imgOut.shape[1]-diffSizesRight[1]))
            outYrange = (0,imgOut_aux.shape[1])
        # z_scale
        if scaleFactor[2]<1.:
            outZrange = (diffSizesLeft[2],(imgOut_aux.shape[2]-diffSizesRight[2]))
            inZrange = (0,imgOut.shape[2])
        else:
            inZrange = (diffSizesLeft[2],(imgOut.shape[2]-diffSizesRight[2]))
            outZrange = (0,imgOut_aux.shape[2])
        
        imgOut_aux[outXrange[0]:outXrange[1], outYrange[0]:outYrange[1], outZrange[0]:outZrange[1]] = imgOut[inXrange[0]:inXrange[1], inYrange[0]:inYrange[1],inZrange[0]:inZrange[1]]
                        
    return imgOut_aux 



def rotate3Dvolume(imgIn, rotAngles = (0,0,0), FLAG_PADDING = 0, interpOrder = 1):
    # Rotation of a 3D volume through the three axes.
    # Input: 
    #  - imgIn: 3D ndarray
    #  - rotAngles: tuple of the three rotation angles (degrees)
    #  - FLAG_PADDING: Flag indicating if including zero padding when computing the rotations
    #                   0 -> no padding
    #                   1 -> independent padding (independent padding for each axis)
    #                   2 -> composed padding (compute sequential padding, more accurate but more computational expensive)
    #  - interpOrder: Interpolation order while rotating the volume
    
    # NO PADDING
    if (FLAG_PADDING == 0):
        imgRot = np.zeros(imgIn.shape)
        row, col = imgRot.shape[0:2]
        M_0 = cv2.getRotationMatrix2D((row / 2, col / 2), rotAngles[0], 1.0)
        for z in range(imgRot.shape[2]):
            imgRot[:, :, z] = scipy.ndimage.interpolation.affine_transform(imgIn[:, :, z], M_0[:, :2], M_0[:, 2], order = interpOrder)
        
        aux = np.swapaxes(imgRot,1,2)    
        row, col = aux.shape[0:2]
        M_0 = cv2.getRotationMatrix2D((row / 2, col / 2), rotAngles[1], 1.0)
        for z in range(aux.shape[2]):
             aux[:, :, z] = scipy.ndimage.interpolation.affine_transform(aux[:, :, z], M_0[:, :2], M_0[:, 2], order = interpOrder)
        imgRot = np.swapaxes(aux,1,2)
    
        aux = np.swapaxes(imgRot,0,2)    
        row, col = aux.shape[0:2]
        M_0 = cv2.getRotationMatrix2D((row / 2, col / 2), rotAngles[2], 1.0)
        for z in range(aux.shape[2]):
             aux[:, :, z] = scipy.ndimage.interpolation.affine_transform(aux[:, :, z], M_0[:, :2], M_0[:, 2], order = interpOrder)
        imgRot = np.swapaxes(aux,0,2)
   
    # INDEPENDENT PADDING
    elif (FLAG_PADDING > 0):
        
        paddingList = np.zeros((3,),dtype=int)
                   
        # 1st rotation ........................................................
        angle_i = rotAngles[0]
        row, col = imgIn.shape[0:2]    
        M_0 = cv2.getRotationMatrix2D((row / 2, col / 2), angle_i, 1.0)
        R_0 = M_0[:, :2]
        V_orig = np.zeros((2,1))
        V_orig[0],V_orig[1] = col/2, row / 2
        V_rot = np.dot(R_0,V_orig)
        pad_i = int(np.ceil(abs(max(V_rot-V_orig))))
        paddingList[0]  = pad_i
            # padding volume
        imgRot = np.lib.pad(imgIn,((pad_i, pad_i),(pad_i, pad_i),(0,0)),'constant', constant_values=(0, 0))
             # rotate
        row, col = imgRot.shape[0:2]
        M_0 = cv2.getRotationMatrix2D((row / 2, col / 2), angle_i, 1.0)
        for z in range(imgRot.shape[2]):
             imgRot[:, :, z] = scipy.ndimage.interpolation.affine_transform(imgRot[:, :, z], M_0[:, :2], M_0[:, 2], order = interpOrder)

        if (FLAG_PADDING==1): imgRot = imgRot[pad_i:-pad_i,pad_i:-pad_i,:]
        
        # 2nd rotation .........................................................
        imgRot = np.swapaxes(imgRot,1,2) 
        angle_i = rotAngles[1]
        row, col = imgRot.shape[0:2]    
        M_0 = cv2.getRotationMatrix2D((row / 2, col / 2), angle_i, 1.0)
        R_0 = M_0[:, :2]
        V_orig = np.zeros((2,1))
        V_orig[0],V_orig[1] = col/2, row / 2
        V_rot = np.dot(R_0,V_orig)
        pad_i = int(np.ceil(abs(max(V_rot-V_orig))))
        paddingList[1]  = pad_i
            # padding volume
        imgRot = np.lib.pad(imgRot,((pad_i, pad_i),(pad_i, pad_i),(0,0)),'constant', constant_values=(0, 0))
            # rotate
        row, col = imgRot.shape[0:2]
        M_0 = cv2.getRotationMatrix2D((row / 2, col / 2), angle_i, 1.0)
        for z in range(imgRot.shape[2]):
            imgRot[:, :, z] = scipy.ndimage.interpolation.affine_transform(imgRot[:, :, z], M_0[:, :2], M_0[:, 2], order = interpOrder)
       
        if (FLAG_PADDING==1): imgRot = imgRot[pad_i:-pad_i,pad_i:-pad_i,:]
        imgRot = np.swapaxes(imgRot,1,2) 

      
       # 3rd rotation ..........................................................
        imgRot = np.swapaxes(imgRot,0,2) 
        angle_i = rotAngles[2]
        row, col = imgRot.shape[0:2]    
        M_0 = cv2.getRotationMatrix2D((row / 2, col / 2), angle_i, 1.0)
        R_0 = M_0[:, :2]
        V_orig = np.zeros((2,1))
        V_orig[0],V_orig[1] = col/2, row / 2
        V_rot = np.dot(R_0,V_orig)
        pad_i = int(np.ceil(abs(max(V_rot-V_orig))))
        paddingList[2]  = pad_i
            # padding volume
        imgRot = np.lib.pad(imgRot,((pad_i, pad_i),(pad_i, pad_i),(0,0)),'constant', constant_values=(0, 0))
            # rotate
        row, col = imgRot.shape[0:2]
        M_0 = cv2.getRotationMatrix2D((row / 2, col / 2), angle_i, 1.0)
        for z in range(imgRot.shape[2]):
            imgRot[:, :, z] = scipy.ndimage.interpolation.affine_transform(imgRot[:, :, z], M_0[:, :2], M_0[:, 2], order = interpOrder)
       
        if (FLAG_PADDING==1): 
            imgRot = imgRot[pad_i:-pad_i,pad_i:-pad_i,:]
            imgRot = np.swapaxes(imgRot,0,2) 


        if (FLAG_PADDING==2):
           imgRot = imgRot[paddingList[2]:-paddingList[2],paddingList[2]:-paddingList[2],:]     
           imgRot = np.swapaxes(imgRot,0,2) 
           imgRot = np.swapaxes(imgRot,1,2)  
           imgRot = imgRot[paddingList[1]:-paddingList[1],paddingList[1]:-paddingList[1],:] 
           imgRot = np.swapaxes(imgRot,1,2) 
           imgRot = imgRot[paddingList[0]:-paddingList[0],paddingList[0]:-paddingList[0],:] 

    return imgRot


def applyTransformToVolume(imgIn, rotAngles = (0,0,0), transOffset = (0,0,0), scaleFactor = 1.0, FLAGpaddingRot = 0, rotInterpOrder = 1, transPaddingValue = 0, FLAGscalePreserveSize =True, scalePaddingValue = 0):
    
    # 1st) Rotation
    imOut = rotate3Dvolume(imgIn, rotAngles = rotAngles, FLAG_PADDING = FLAGpaddingRot, interpOrder = rotInterpOrder)
    # 2nd) Translation
    imOut = translate3Dvolume(imOut, transOffset = transOffset, paddingValue = transPaddingValue)
    # 3rd) Scaling
    imOut = scale3DVolume(imOut, scaleFactor = scaleFactor, FLAG_sameSize = FLAGscalePreserveSize, paddingValue = scalePaddingValue)
    
    return imOut


def applyTransformToVolumeAnisoScale(imgIn, rotAngles = (0,0,0), transOffset = (0,0,0), scaleFactor = (1.0, 1.0, 1.0), FLAGpaddingRot = 0, rotInterpOrder = 1, transPaddingValue = 0, FLAGscalePreserveSize =True, scalePaddingValue = 0):
    
    # 2nd) Scaling
    if isinstance(scaleFactor, tuple):
        # Anisotropic scale
        imOut = anisoScale3DVolume(imgIn, scaleFactor = scaleFactor, FLAG_sameSize = FLAGscalePreserveSize, paddingValue = scalePaddingValue)

    elif isinstance(scaleFactor, float):
        # Isotropic scale
        imOut = scale3DVolume(imgIn, scaleFactor = scaleFactor, FLAG_sameSize = FLAGscalePreserveSize, paddingValue = scalePaddingValue)
        
    # 1st) Rotation
    imOut = rotate3Dvolume(imOut, rotAngles = rotAngles, FLAG_PADDING = FLAGpaddingRot, interpOrder = rotInterpOrder)
    
    # 3rd) Translation
    imOut = translate3Dvolume(imOut, transOffset = transOffset, paddingValue = transPaddingValue)

    return imOut
      
  
    

def applyTransformTo2Dplane(imgIn, scaleFactor = (1.0,1.0), rotAngle = 0., transOffset = (0,0)):
    
    # stdplane_in: 2D nparray
    # view_in:  'sagital', 'axial', 'coronal'
    # scaling_list: tuple of scaling factors applied to the 3D volume (coronal_axis_scaling, sagital_axis_scaling, axial_axis_scaling)
    #   - sagital view: combination of coronal-axis-scale (scaling_list[0]) and axial-axis-scale (scaling_list[2])
    #   - coronal view: combination of sagital-axis-scale (scaling_list[1]) and axial-axis-scale (scaling_list[2])
    #   - axial view: combination of coronal-axis-scale (scaling_list[0]) and sagital-axis-scale (scaling_list[1])
    # translation_in: tuple of translation in both axis
    # rotation_in: angle of rotation in degrees
    
    #    pos_rnd = get_random_viewID(IDs_axial, 66)
    #    stdplane_in = all_2D_axial[:,:,pos_rnd]
    #    view_in = 'axial'
    #    scaling_list = (1.5, 1.1, 0.5)
    #    rotation_in = 10
    #    translation_in = (-5,10)
    
    
    orig_shape = np.array(imgIn.shape)
    orig_min = imgIn.min()
    orig_max = imgIn.max()
    
    # 1st scaling
    zoom_i = np.array([scaleFactor[0], scaleFactor[1]])
    img_out = scipy.ndimage.zoom(imgIn, zoom_i)
    
    # 2nd rotate 
    img_out = scipy.ndimage.rotate(img_out,rotAngle)
    
    # 3rd translate
    if transOffset[0]<0:
        trans_rows = (abs(transOffset[0]),0)
    else:
        trans_rows = (0,abs(transOffset[0]))
    if transOffset[1]<0:
        trans_cols = (abs(transOffset[1]),0)  
    else:
        trans_cols = (0,abs(transOffset[1]))
    img_out = np.pad(img_out,(trans_rows, trans_cols),'constant',constant_values = img_out.min())

    # 4th recover the orginal shape    
    
    # i) get center
    center_i = np.round(np.array(img_out.shape)/2.)
    # ii) rows                
    if center_i[0] < orig_shape[0]/2:
        padd_rows = (int(orig_shape[0]/2 - center_i[0] +2), int(orig_shape[0]/2 - center_i[0] +2))
    else:
        padd_rows = (0,0)   
    # ii) cols                
    if center_i[1] < orig_shape[1]/2:
        padd_cols = (int(orig_shape[1]/2 - center_i[1] +2), int(orig_shape[1]/2 - center_i[1] +2))
    else:
        padd_cols = (0,0)                  
     
    img_out = np.pad(img_out, (padd_rows, padd_cols),'constant',constant_values = img_out.min())                 
    center_i = np.round(np.array(img_out.shape)/2.)
    img_out = img_out[int(center_i[0]- orig_shape[0]/2):int(center_i[0]+orig_shape[0]/2), int(center_i[1]- orig_shape[1]/2):int(center_i[1]+orig_shape[1]/2)]
    
    # normalize intensity
    img_out = (img_out - img_out.min())/(img_out.max()-img_out.min()) # image  in [0 1]
    img_out = (img_out * (orig_max - orig_min)) + orig_min

    return img_out
    #    plt.figure()
    #    plt.subplot(131)
    #    plt.imshow(stdplane_in,cmap='gray')
    #    plt.subplot(132)
    #    plt.imshow(img_out,cmap='gray')
    #    plt.subplot(133)
    #    plt.imshow(img_out_ROI,cmap='gray')



#==============================================================================
# # Examples
#
# sys.path.append("F:\myPyLibrary")
# from volumeTransformations import *
# from GenerateNiftiFilesFromData import *
#
#  --- translate3Dvolume ---
#
# PATH_IMG = 'F:\\Data\\iFind\\Brain_3DUS_corr_and_preproc_mat\\images'
# File_name = 'ifind52-98.nii.gz'
# imgOrig = nib.load(os.path.join(PATH_IMG, File_name))
# imgTransl = translate3Dvolume(imgOrig.get_data(), transOffset = (30,-20,10), paddingValue = imgOrig.get_data().min() )
# generateNiiFileFromData(imgTransl, ImgNiiRef=imgOrig, fileName = 'F:\\myPyLibrary\\imgTrans.nii.gz')
#
#
#  --- scale3DVolume ---
#
# PATH_IMG = 'F:\\Data\\iFind\\Brain_3DUS_corr_and_preproc_mat\\images'
# File_name = 'ifind52-98.nii.gz'
# imgOrig = nib.load(os.path.join(PATH_IMG, File_name))
# imgIn = imgOrig.get_data()
# imgScale = scale3DVolume(imgIn, scaleFactor = 1.3, FLAG_sameSize = True, paddingValue = imgIn.min())
# 
# imgScale = anisoScale3DVolume(imgIn, scaleFactor = (1.3, 1.0, 0.5), FLAG_sameSize = True, paddingValue = imgIn.min())
# generateNiiFileFromData(imgScale, ImgNiiRef=imgOrig, fileName = 'F:\\myPyLibrary\\imgScale.nii.gz')
#
# imgTransf = applyTransformToVolumeAnisoScale(imgIn, rotAngles = (15,0,0), transOffset = (10,0,20), scaleFactor = (1.3, 1.0, 0.5), FLAGpaddingRot = 0, rotInterpOrder = 1, transPaddingValue = 0, FLAGscalePreserveSize =True, scalePaddingValue = 0)
# generateNiiFileFromData(imgTransf, ImgNiiRef=imgOrig, fileName = 'F:\\myPyLibrary\\imgTransf.nii.gz')
#
#  --- rotate3Dvolume ---
#
# PATH_IMG = 'F:\\Data\\iFind\\Brain_3DUS_corr_and_preproc_mat\\images'
# File_name = 'ifind52-98.nii.gz'
# imgOrig = nib.load(os.path.join(PATH_IMG, File_name))
# imgRot = rotate3Dvolume(imgOrig.get_data(), rotAngles = (-10,10,10), FLAG_PADDING = 0, interpOrder = 1)
# generateNiiFileFromData(imgRot, ImgNiiRef=imgOrig, fileName = 'F:\\myPyLibrary\\imgRot.nii.gz')
# 
# PATH_ROI = 'F:\\Data\\iFind\\Brain_3DUS_corr_and_preproc_mat\\mask'
# File_name = 'ifind52-98.nii.gz'
# imgOrig = nib.load(os.path.join(PATH_ROI, File_name))
# imgRot = rotate3Dvolume(imgOrig.get_data(), rotAngles = (-10,10,10), FLAG_PADDING = 0, interpOrder = 1)
# generateNiiFileFromData(imgRot, ImgNiiRef=imgOrig, fileName = 'F:\\myPyLibrary\\roiRot.nii.gz')
# 
# PATH_GT = 'F:\\Data\\iFind\\Brain_3DUS_corr_and_preproc_mat\\partial_seg2'
# File_name = 'ifind52-98.nii.gz'
# imgOrig = nib.load(os.path.join(PATH_GT, File_name))
# imgRot = rotate3Dvolume(imgOrig.get_data(), rotAngles = (-10,10,10), FLAG_PADDING = 0, interpOrder = 1)
# generateNiiFileFromData(imgRot, ImgNiiRef=imgOrig, fileName = 'F:\\myPyLibrary\\gtRot.nii.gz')
#
# NOTE:     Additional post-processing of the image would be necessary if a binary mask or a groundtruth is rotated in order 
#           to compensate for the interpolation of the image.
#
#
#  --- restoreIntValues ---
#
# import volumeTransformations
# reload(volumeTransformations)
# from volumeTransformations import *
# PATH_GT = 'F:\\Data\\iFind\\Brain_3DUS_corr_and_preproc_mat\\partial_seg2'
# File_name = 'ifind52-98.nii.gz'
# imgOrig = nib.load(os.path.join(PATH_GT, File_name))
# imgRot = rotate3Dvolume(imgOrig.get_data(), rotAngles = (-10,10,10), FLAG_PADDING = 0, interpOrder = 1)
# generateNiiFileFromData(imgRot, ImgNiiRef=imgOrig, fileName = 'F:\\myPyLibrary\\gtRot.nii.gz')
# imgRot = restoreIntValues (imgRot, np.unique(imgOrig.get_data()))
# generateNiiFileFromData(imgRot, ImgNiiRef=imgOrig, fileName = 'F:\\myPyLibrary\\gtRotRestored.nii.gz')
#
#
#  --- applyTransformToVolume  ---
#
# PATH_IMG = 'F:\\Data\\iFind\\Brain_3DUS_corr_and_preproc_mat\\images'
# File_name = 'ifind52-98.nii.gz'
# imgOrig = nib.load(os.path.join(PATH_IMG, File_name))

# imgOut = applyTransformToVolume(imgOrig.get_data(), rotAngles = (10,-7.5,12.3), transOffset = (30, -50, 20), scaleFactor = 1.2, FLAGpaddingRot = 1, rotInterpOrder = 1, transPaddingValue = imgOrig.get_data().min(), FLAGscalePreserveSize =True, scalePaddingValue = imgOrig.get_data().min())
# generateNiiFileFromData(imgOut, ImgNiiRef=imgOrig, fileName = 'F:\\myPyLibrary\\imgTrans.nii.gz')
#==============================================================================



