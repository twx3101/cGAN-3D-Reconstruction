
import os
import numpy as np
import nibabel as nib


#==============================================================================
# Functions to generate a Nifti file from an input matrix data.
# Useful to visualize the intermediate results when working on or processing a
# volume with an external image viewer.
#==============================================================================


def createNiiImageFromeData(imgData, ImgNiiRef=None, ImgNiiAffine=None, ImgNiiHeader=None):
    if ImgNiiRef!=None:
         INiiOut = nib.Nifti1Image(imgData, ImgNiiRef.affine, ImgNiiRef.header)
    else:
        if ImgNiiAffine==None:
            affineAux = np.identity(4)
        else:
            affineAux = ImgNiiAffine
    
        if ImgNiiHeader==None:
            INiiOut = nib.Nifti1Image(imgData, affineAux)
        else:
            INiiOut = nib.Nifti1Image(imgData, affineAux, ImgNiiHeader)          
    return INiiOut


def generateNiiFileFromData(imgData, ImgNiiRef=None, ImgNiiAffine=None, ImgNiiHeader=None, fileName = 'niifromdata.nii.gz'):
    INiiOut = createNiiImageFromeData(imgData, ImgNiiRef, ImgNiiAffine, ImgNiiHeader)
    nib.save(INiiOut,fileName)
    
    

#==============================================================================
# # Example:
#     
# PATH_TMP = 'F:\\Data\\iFind\\Data_to_mark_manual_landmarks\\US'
# File_i = 'iFIND30_IM_0001.nii.gz'
# ImgNiiRef = nib.load(os.path.join(PATH_TMP, File_i))
# imgData = np.random.randn(ImgNiiRef.shape[0],ImgNiiRef.shape[1],ImgNiiRef.shape[2])
# generateNiiFileFromData(imgData, ImgNiiRef = ImgNiiRef, fileName = 'F:\\myPyLibrary\\test_1.nii.gz')
# 
# generateNiiFileFromData(imgData, fileName = 'F:\\myPyLibrary\\test_2.nii.gz')
#==============================================================================
