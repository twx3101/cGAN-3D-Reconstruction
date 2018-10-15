import os
import numpy as np
import nibabel as nib
from nilearn import plotting
np.set_printoptions(precision=15)
np.set_printoptions(threshold=np.inf)

#for filename in os.listdir(path):
#     prefix, left = filename.split('_', maxsplit=1)
#     num = left.split('_', maxsplit=1)
#     if len(num) == 1 :
#         num1 = num[0]
#         num1 = num1.zfill(3)
#         new_filename = prefix + "_" + num1
#     else:
#         num1 = num[0]
#         num1 = num1.zfill(3)
#         rest = num[1]
#         new_filename = prefix + "_" + num1 + "_" + rest
#     os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

config_path = os.path.abspath("./BatchGeneratorClass/config")

def load_data():
    filenames = []
    path = os.path.join(".", "data_newOK")
    for filename in os.listdir(path):
        if filename.count('_') == 2:
            filenames.append(filename)

    filenames.sort()
    x = (len(filenames))

    data_x = np.zeros(shape = (x, 96, 96, 96))

    i = 0
    for filename in filenames:
        image = os.path.join(path, filename)
        img = nib.load(image)
        data = img.get_data()
        data_x[i] = data
        i += 1

    print(data_x.shape)
    print("Done loading data")

    train, test = data_x[:138, :, :, :], data_x[138:, :, :, :]
    return train, test


def load_US_3d():
    training_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_img_training_K1_v2.cfg')
    validation_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_img_validation_K1_v2.cfg')

    f = open(training_path, 'r')

    x_train = np.zeros(shape = (100, 96, 96, 96))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x_train[i] = data
        i += 1

    g = open(validation_path, 'r')
    x_val = np.zeros(shape=(35,96,96,96))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x_val[j] = data
        j += 1
        print(line)

    f.close()
    g.close()

    return x_train, x_val

def load_US_3d_fold(fold=0):
    training_path = os.path.join(config_path, 'Train_%d/fold_%d_train.cfg' %(fold+1, fold+1))
    validation_path = os.path.join(config_path, 'k-fold_config/Skull_Reconstruction_from_2DStdPlanes_img_test_K1_v2.cfg')

    f = open(training_path, 'r')

    x_train = np.zeros(shape = (90, 96, 96, 96))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x_train[i] = data
        i += 1

    g = open(validation_path, 'r')
    x_val = np.zeros(shape=(15,96,96,96))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x_val[j] = data
        j += 1
        #print(line)

    f.close()
    g.close()

    return x_train, x_val





def load_segments():


    cor_t_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_cor_seg_training_K1_v2.cfg')
    cor_v_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_cor_seg_validation_K1_v2.cfg')

    sag_t_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_sag_seg_training_K1_v2.cfg')
    sag_v_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_sag_seg_validation_K1_v2.cfg')

    trvent_t_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_trvent_seg_training_K1_v2.cfg')
    trvent_v_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_trvent_seg_validation_K1_v2.cfg')

    f = open(cor_t_path, 'r')

    x1_train = np.zeros(shape = (100, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x1_train[i] = data
        i += 1

    g = open(cor_v_path, 'r')
    x1_val = np.zeros(shape=(35,96,96, 1))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x1_val[j] = data
        j += 1

    f.close()
    g.close()

    f = open(sag_t_path, 'r')

    x2_train = np.zeros(shape = (100, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x2_train[i] = data
        i += 1

    g = open(sag_v_path, 'r')
    x2_val = np.zeros(shape=(35,96,96, 1))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x2_val[j] = data
        j += 1

    f.close()
    g.close()

    f = open(trvent_t_path, 'r')

    x3_train = np.zeros(shape = (100, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x3_train[i] = data
        i += 1

    g = open(trvent_v_path, 'r')
    x3_val = np.zeros(shape=(35,96,96,1 ))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x3_val[j] = data
        j += 1

    f.close()
    g.close()

    return x1_train, x1_val, x2_train, x2_val, x3_train, x3_val

def load_2d():
    cor_t_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_cor_training_K1_v2.cfg')
    cor_v_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_cor_validation_K1_v2.cfg')

    sag_t_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_sag_training_K1_v2.cfg')
    sag_v_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_sag_validation_K1_v2.cfg')

    trvent_t_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_trvent_training_K1_v2.cfg')
    trvent_v_path = os.path.join(config_path, 'Skull_Reconstruction_from_2DStdPlanes_trvent_validation_K1_v2.cfg')
    f = open(cor_t_path, 'r')

    x1_train = np.zeros(shape = (90, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x1_train[i] = data
        i += 1

    g = open(cor_v_path, 'r')
    x1_val = np.zeros(shape=(15,96,96, 1))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x1_val[j] = data
        j += 1

    f.close()
    g.close()

    f = open(sag_t_path, 'r')

    x2_train = np.zeros(shape = (90, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x2_train[i] = data
        i += 1

    g = open(sag_v_path, 'r')
    x2_val = np.zeros(shape=(15,96,96, 1))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x2_val[j] = data
        j += 1

    f.close()
    g.close()

    f = open(trvent_t_path, 'r')

    x3_train = np.zeros(shape = (90, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x3_train[i] = data
        i += 1

    g = open(trvent_v_path, 'r')
    x3_val = np.zeros(shape=(15,96,96,1 ))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x3_val[j] = data
        j += 1

    f.close()
    g.close()

    return x1_train, x1_val, x2_train, x2_val, x3_train, x3_val

def load_2d_fold(fold=0):


    training_path = os.path.join(config_path, 'Train_%d/fold_%d_train.cfg' %(fold+1, fold+1))
    validation_path = os.path.join(config_path, 'k-fold_config/Skull_Reconstruction_from_2DStdPlanes_img_test_K1_v2.cfg')

    cor_t_path = os.path.join(config_path, 'Train_%d/fold_%d_train_cor.cfg' %(fold+1, fold+1))
    cor_v_path = os.path.join(config_path, 'k-fold_config/Skull_Reconstruction_from_2DStdPlanes_cor_test_K1_v2.cfg')

    sag_t_path = os.path.join(config_path, 'Train_%d/fold_%d_train_sag.cfg' %(fold+1, fold+1))
    sag_v_path = os.path.join(config_path, 'k-fold_config/Skull_Reconstruction_from_2DStdPlanes_sag_test_K1_v2.cfg')

    trvent_t_path = os.path.join(config_path, 'Train_%d/fold_%d_train_trvent.cfg' %(fold+1, fold+1))
    trvent_v_path = os.path.join(config_path, 'k-fold_config/Skull_Reconstruction_from_2DStdPlanes_trvent_test_K1_v2.cfg')


    f = open(cor_t_path, 'r')

    x1_train = np.zeros(shape = (100, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x1_train[i] = data
        i += 1

    g = open(cor_v_path, 'r')
    x1_val = np.zeros(shape=(15,96,96, 1))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x1_val[j] = data
        j += 1

    f.close()
    g.close()

    f = open(sag_t_path, 'r')

    x2_train = np.zeros(shape = (100, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x2_train[i] = data
        i += 1

    g = open(sag_v_path, 'r')
    x2_val = np.zeros(shape=(15,96,96, 1))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x2_val[j] = data
        j += 1

    f.close()
    g.close()

    f = open(trvent_t_path, 'r')

    x3_train = np.zeros(shape = (100, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x3_train[i] = data
        i += 1

    g = open(trvent_v_path, 'r')
    x3_val = np.zeros(shape=(15,96,96,1 ))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x3_val[j] = data
        j += 1

    f.close()
    g.close()

    return x1_train, x1_val, x2_train, x2_val, x3_train, x3_val

def load_2d_seg_fold(fold=0):


    training_path = os.path.join(config_path, 'k-fold_config/Skull_Reconstruction_from_2DStdPlanes_img_training_K1_v2.cfg')
    validation_path = os.path.join(config_path, 'k-fold_config/Skull_Reconstruction_from_2DStdPlanes_img_test_K1_v2.cfg')

    cor_t_path = os.path.join(config_path, 'k-fold_config/Segmentation/Skull_Reconstruction_from_2DStdPlanes_cor_seg_training_K1_v2.cfg')
    cor_v_path = os.path.join(config_path, 'k-fold_config/Segmentation/Skull_Reconstruction_from_2DStdPlanes_cor_seg_test_K1_v2.cfg')

    sag_t_path = os.path.join(config_path, 'k-fold_config/Segmentation/Skull_Reconstruction_from_2DStdPlanes_sag_seg_training_K1_v2.cfg')
    sag_v_path = os.path.join(config_path, 'k-fold_config/Segmentation/Skull_Reconstruction_from_2DStdPlanes_sag_seg_test_K1_v2.cfg')

    trvent_t_path = os.path.join(config_path, 'k-fold_config/Segmentation/Skull_Reconstruction_from_2DStdPlanes_trvent_seg_training_K1_v2.cfg')
    trvent_v_path = os.path.join(config_path, 'k-fold_config/Segmentation/Skull_Reconstruction_from_2DStdPlanes_trvent_seg_test_K1_v2.cfg')


    f = open(cor_t_path, 'r')

    x1_train = np.zeros(shape = (120, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x1_train[i] = data
        i += 1

    g = open(cor_v_path, 'r')
    x1_val = np.zeros(shape=(15,96,96, 1))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x1_val[j] = data
        j += 1

    f.close()
    g.close()

    f = open(sag_t_path, 'r')

    x2_train = np.zeros(shape = (120, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x2_train[i] = data
        i += 1

    g = open(sag_v_path, 'r')
    x2_val = np.zeros(shape=(15,96,96, 1))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x2_val[j] = data
        j += 1

    f.close()
    g.close()

    f = open(trvent_t_path, 'r')

    x3_train = np.zeros(shape = (120, 96, 96, 1))
    i = 0
    for line in f:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x3_train[i] = data
        i += 1

    g = open(trvent_v_path, 'r')
    x3_val = np.zeros(shape=(15,96,96,1 ))
    j = 0


    for line in g:
        line = line.rstrip('\n')
        img = nib.load(line)
        data = img.get_data()
        x3_val[j] = data
        j += 1

    f.close()
    g.close()

    return x1_train, x1_val, x2_train, x2_val, x3_train, x3_val

def get_affine_3d():
    path = os.path.join(".", "data_newOK")

    filenames = os.listdir(path)
    filenames.sort()
    #print(filenames[7])
    imagefile = os.path.join(path, filenames[0])
    img = nib.load(imagefile)

    return img.affine

if __name__ == '__main__':
    load_US_3d_fold(1)
