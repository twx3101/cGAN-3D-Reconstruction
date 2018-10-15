import os
import numpy as np
path = os.path.abspath("/homes/wt814/IndividualProject/code/data_newOK")
print(path)

# train_1 = open('train_1.cfg', 'w')
# train_1_id = open('train_1_id.cfg', 'w')
#
# val_1 = open('val_1.cfg', 'w')
# val_1_id = open('val_1_id.cfg', 'w')
#
# train_cor_1 = open('train_cor_1.cfg', 'w')
# train_cor_1_id = open('train_cor_1_id.cfg', 'w')
#
# val_cor_1 = open('val_cor_1.cfg', 'w')
# val_cor_1_id = open('val_cor_1_id.cfg', 'w')
#
# train_sag_1 = open('train_sag_1.cfg', 'w')
# train_sag_1_id = open('train_sag_1_id.cfg', 'w')
#
# val_sag_1 = open('val_sag_1.cfg', 'w')
# val_sag_1_id = open('val_sag_1_id.cfg', 'w')
#
# train_trvent_1 = open('train_trvent_1.cfg', 'w')
# train_trvent_1_id = open('train_trvent_1_id.cfg', 'w')
#
# val_trvent_1 = open('val_trvent_1.cfg', 'w')
# val_trvent_1_id = open('val_trvent_1_id.cfg', 'w')
#
# ############################################### Fold 2 ##############################
# train_2 = open('train_2.cfg', 'w')
# train_2_id = open('train_2_id.cfg', 'w')
#
# val_2 = open('val_2.cfg', 'w')
# val_2_id = open('val_2_id.cfg', 'w')
#
# train_cor_2 = open('train_cor_2.cfg', 'w')
# train_cor_2_id = open('train_cor_2_id.cfg', 'w')
#
# val_cor_2 = open('val_cor_2.cfg', 'w')
# val_cor_2_id = open('val_cor_2_id.cfg', 'w')
#
# train_sag_2 = open('train_sag_2.cfg', 'w')
# train_sag_2_id = open('train_sag_2_id.cfg', 'w')
#
# val_sag_2 = open('val_sag_2.cfg', 'w')
# val_sag_2_id = open('val_sag_2_id.cfg', 'w')
#
# train_trvent_2 = open('train_trvent_2.cfg', 'w')
# train_trvent_2_id = open('train_trvent_2_id.cfg', 'w')
#
# val_trvent_2 = open('val_trvent_2.cfg', 'w')
# val_trvent_2_id = open('val_trvent_2_id.cfg', 'w')
#
#
# ###################################### Fold 3 ###################################
# train_4 = open('train_4.cfg', 'w')
# train_4_id = open('train_4_id.cfg', 'w')
#
# val_4 = open('val_4.cfg', 'w')
# val_4_id = open('val_4_id.cfg', 'w')
#
# train_cor_4 = open('train_cor_4.cfg', 'w')
# train_cor_4_id = open('train_cor_4_id.cfg', 'w')
#
# val_cor_4 = open('val_cor_4.cfg', 'w')
# val_cor_4_id = open('val_cor_4_id.cfg', 'w')
#
# train_sag_4 = open('train_sag_4.cfg', 'w')
# train_sag_4_id = open('train_sag_4_id.cfg', 'w')
#
# val_sag_4 = open('val_sag_4.cfg', 'w')
# val_sag_4_id = open('val_sag_4_id.cfg', 'w')
#
# train_trvent_4 = open('train_trvent_4.cfg', 'w')
# train_trvent_4_id = open('train_trvent_4_id.cfg', 'w')
#
# val_trvent_3 = open('val_trvent_3.cfg', 'w')
# val_trvent_3_id = open('val_trvent_3_id.cfg', 'w')
#
# ############################### Fold_4 #########################################
#
# train_4 = open('train_4.cfg', 'w')
# train_4_id = open('train_4_id.cfg', 'w')
#
# val_4 = open('val_4.cfg', 'w')
# val_4_id = open('val_4_id.cfg', 'w')
#
# train_cor_4 = open('train_cor_4.cfg', 'w')
# train_cor_4_id = open('train_cor_4_id.cfg', 'w')
#
# val_cor_4 = open('val_cor_4.cfg', 'w')
# val_cor_4_id = open('val_cor_4_id.cfg', 'w')
#
# train_sag_4 = open('train_sag_4.cfg', 'w')
# train_sag_4_id = open('train_sag_4_id.cfg', 'w')
#
# val_sag_4 = open('val_sag_4.cfg', 'w')
# val_sag_4_id = open('val_sag_4_id.cfg', 'w')
#
# train_trvent_4 = open('train_trvent_4.cfg', 'w')
# train_trvent_4_id = open('train_trvent_4_id.cfg', 'w')
#
# val_trvent_4 = open('val_trvent_4.cfg', 'w')
# val_trvent_4_id = open('val_trvent_4_id.cfg', 'w')


############################## SPLIT #########################################

def split_files():

    with open('Skull_Reconstruction_from_2DStdPlanes_img_training_K1_v2.cfg') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % 30 == 0:
                small_filename = 'train_{}.cfg'.format(lineno/30 + 1)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

    with open('Skull_Reconstruction_from_2DStdPlanes_img_training_K1_ID_v2.cfg') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % 30 == 0:
                small_filename = 'train_ID_{}.cfg'.format(lineno/30 + 1)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

    with open('Skull_Reconstruction_from_2DStdPlanes_cor_training_K1_v2.cfg') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % 30 == 0:
                small_filename = 'train_cor_{}.cfg'.format(lineno/30 + 1)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

    with open('Skull_Reconstruction_from_2DStdPlanes_cor_training_K1_ID_v2.cfg') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % 30 == 0:
                small_filename = 'train_cor_ID{}.cfg'.format(lineno/30 + 1)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

    with open('Skull_Reconstruction_from_2DStdPlanes_sag_training_K1_v2.cfg') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % 30 == 0:
                small_filename = 'train_sag_{}.cfg'.format(lineno/30 + 1)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

    with open('Skull_Reconstruction_from_2DStdPlanes_sag_training_K1_ID_v2.cfg') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % 30 == 0:
                small_filename = 'train_sag_ID_{}.cfg'.format(lineno/30 + 1)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

    with open('Skull_Reconstruction_from_2DStdPlanes_trvent_training_K1_v2.cfg') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % 30 == 0:
                small_filename = 'train_trvent_{}.cfg'.format(lineno/30 + 1)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

    with open('Skull_Reconstruction_from_2DStdPlanes_trvent_training_K1_ID_v2.cfg') as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % 30 == 0:
                small_filename = 'train_trvent_ID_{}.cfg'.format(lineno/30 + 1)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()


def create_k_fold():
    ####################### FOLD 1 ##############################

    filenames = ['./3/train_3.0.cfg', './2/train_2.0.cfg', './4/train_4.0.cfg']
    with open('./Train_3/fold_3_train.cfg', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

    with open('./1/train_1.0.cfg') as f:
        read_data = f.read()
        with open('./Train_3/fold_3_val.cfg', 'w') as outfile:
            outfile.write(read_data)

    filenames = ['./3/train_ID_3.0.cfg', './2/train_ID_2.0.cfg', './4/train_ID_4.0.cfg']
    with open('./Train_3/fold_3_train_ID.cfg', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

    with open('./1/train_ID_1.0.cfg') as f:
        read_data = f.read()
        with open('./Train_3/fold_3_val_ID.cfg', 'w') as outfile:
            outfile.write(read_data)

    filenames = ['./3/train_cor_3.0.cfg', './2/train_cor_2.0.cfg', './4/train_cor_4.0.cfg']
    with open('./Train_3/fold_3_train_cor.cfg', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

    with open('./1/train_cor_1.0.cfg') as f:
        read_data = f.read()
        with open('./Train_3/fold_3_val_cor.cfg', 'w') as outfile:
            outfile.write(read_data)

    filenames = ['./3/train_cor_ID3.0.cfg', './2/train_cor_ID2.0.cfg', './4/train_cor_ID4.0.cfg']
    with open('./Train_3/fold_3_train_cor_ID.cfg', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

    with open('./1/train_cor_ID1.0.cfg') as f:
        read_data = f.read()
        with open('./Train_3/fold_3_val_cor_ID.cfg', 'w') as outfile:
            outfile.write(read_data)

    filenames = ['./3/train_sag_3.0.cfg', './2/train_sag_2.0.cfg', './4/train_sag_4.0.cfg']
    with open('./Train_3/fold_3_train_sag.cfg', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

    with open('./1/train_sag_1.0.cfg') as f:
        read_data = f.read()
        with open('./Train_3/fold_3_val_sag.cfg', 'w') as outfile:
            outfile.write(read_data)

    filenames = ['./3/train_sag_ID_3.0.cfg', './2/train_sag_ID_2.0.cfg', './4/train_sag_ID_4.0.cfg']
    with open('./Train_3/fold_3_train_sag_ID.cfg', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

    with open('./1/train_trvent_ID_1.0.cfg') as f:
        read_data = f.read()
        with open('./Train_3/fold_3_val_trvent_ID.cfg', 'w') as outfile:
            outfile.write(read_data)

    filenames = ['./3/train_trvent_3.0.cfg', './2/train_trvent_2.0.cfg', './4/train_trvent_4.0.cfg']
    with open('./Train_3/fold_3_train_trvent.cfg', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

    with open('./3/train_trvent_3.0.cfg') as f:
        read_data = f.read()
        with open('./Train_1/fold_1_val_trvent.cfg', 'w') as outfile:
            outfile.write(read_data)

    filenames = ['./3/train_trvent_ID_3.0.cfg', './2/train_trvent_ID_2.0.cfg', './4/train_trvent_ID_4.0.cfg']
    with open('./Train_3/fold_3_train_trvent_ID.cfg', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

    with open('./1/train_trvent_ID_1.0.cfg') as f:
        read_data = f.read()
        with open('./Train_3/fold_3_val_trvent_ID.cfg', 'w') as outfile:
            outfile.write(read_data)

if __name__ == '__main__':
    create_k_fold()
