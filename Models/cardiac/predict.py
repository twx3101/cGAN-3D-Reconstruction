from keras.models import load_model
import numpy as np
import nibabel as nib
import segment2d
import os

import utils

generator = load_model('./models/3iter/cGAN_generator.h5')

_, x1, x2, x3 = segment2d.load_evaluate()

def test():
    for j in range(len(x1)):
        for i in range(10):
            if i == 0:
                noise = np.zeros((1,80,80))
            else:
                np.random.seed(i)
                noise = np.random.normal(0, 1, (1, 80,80))

        #noise = np.zeros((1,96,96))
            noise = np.reshape(noise,(1,80,80,1))
        #x_condition = np.concatenate((x1_test_real[i], x2[i]), axis=-1)
        #x_condition = np.concatenate((x_condition,x3[i]), axis=-1)
        #x_condition = np.reshape(x_condition, (1,96,96,3))
        #sampled_labels = x_condition
            x1_c =  np.reshape(x1[j], (1,80,80,1))
            x2_c =  np.reshape(x2[j], (1,80,80,1))
            x3_c =  np.reshape(x3[j], (1,80,80,1))
            gen_imgs = generator.predict([noise, x1_c, x2_c, x3_c])
            gen_imgs = np.reshape(gen_imgs, (80,80,80))

            a = utils.binarisation(gen_imgs)
            filename = '/vol/gpudata/wt814/cardiacImage/noise/%d' %j
            if not os.path.isdir(filename):
                try:
                    os.makedirs(filename)
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            save_img = nib.Nifti1Image(a, np.eye(4))
            nib.save(save_img, '/vol/gpudata/wt814/cardiacImage/noise/%d/%d.nii.gz' %(j,i))

def DICE(path):
    a = np.zeros((100,10))
    for i in range(100):
        path2 = os.path.join(path, str(i))

        filenames = os.listdir(path2)
        testfile = os.path.join(path2, filenames[0])
        test = nib.load(testfile)
        testdata = test.get_fdata()
        for j in range(len(filenames[1:])):
            imagefile = os.path.join(path2, filenames[j+1])
            img = nib.load(imagefile)
            data = img.get_fdata()

            value = utils.DICE(data, testdata)
            a[i][j] = value
    save_path = path + "DICE.csv"
    a = a.T
    np.savetxt(save_path, a, delimiter=",")


def convert_to_nifti():
    for j in range(len(x1)):
        noise = np.random.normal(0, 1, (1, 80,80))
        noise = np.reshape(noise,(1,80,80,1))
    #x_condition = np.concatenate((x1_test_real[i], x2[i]), axis=-1)
    #x_condition = np.concatenate((x_condition,x3[i]), axis=-1)
    #x_condition = np.reshape(x_condition, (1,96,96,3))
    #sampled_labels = x_condition
        x1_c =  np.reshape(x1[j], (1,80,80,1))
        x2_c =  np.reshape(x2[j], (1,80,80,1))
        x3_c =  np.reshape(x3[j], (1,80,80,1))
        gen_imgs = generator.predict([noise, x1_c, x2_c, x3_c])
        gen_imgs = np.reshape(gen_imgs, (80,80,80))

        a = utils.binarisation(gen_imgs)
        filename = '/vol/gpudata/wt814/cardiacImage/3iter/'
        if not os.path.isdir(filename):
            try:
                os.makedirs(filename)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        save_img = nib.Nifti1Image(a, np.eye(4))
        nib.save(save_img, '/vol/gpudata/wt814/cardiacImage/noise/%d.nii.gz' %(j))
if __name__ == '__main__':
    convert_to_nifti()
