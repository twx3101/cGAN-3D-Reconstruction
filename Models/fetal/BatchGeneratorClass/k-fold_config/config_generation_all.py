import os
import numpy as np
path = "/homes/wt814/IndividualProject/code/data_newOK"
print(path)
import itertools

# for filename in os.listdir(path):
#     prefix, left = filename.split('_', maxsplit=1)
#     num = left.split('_', maxsplit=1)
#     if filename.count('_') == 2 :
#         num1 = num[0]
#         num1 = num1.zfill(3)
#         new_filename = prefix + "_" + num1 + "_" + num[1]
#
#     else:
#         num1 = num[0]
#         num1 = num1.zfill(3)
#         rest = num[1]
#         new_filename = prefix + "_" + num1 + "_" + rest
#
#     os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

def generate_config():
    axial_seg = []
    coronal_seg = []
    sagital_seg = []
    full_config =[]
    axial = []
    coronal = []
    sagital = []

    for filename in os.listdir(path):
        if filename.count('_') == 4:
            if "Cor" in filename:
                coronal_seg.append(filename)
            elif "Sagital" in filename:
                sagital_seg.append(filename)
            elif "TrVent" in filename:
                axial_seg.append(filename)
        if filename.count('_') == 3:
            if "Cor" in filename:
                coronal.append(filename)
            elif "Sagital" in filename:
                sagital.append(filename)
            elif "TrVent" in filename:
                axial.append(filename)
        elif filename.count('_') == 2:
            full_config.append(filename)


    axial_seg.sort()
    coronal_seg.sort()
    sagital_seg.sort()
    axial.sort()
    coronal.sort()
    sagital.sort()
    full_config.sort()
    #
    a_config =[]
    c_config = []
    s_config = []
    a_seg_config = []
    c_seg_config =[]
    s_seg_config = []
    f_config = []
    j = 413
    #
    # #remove duplicates
    for no in range(j):
        num1 = str(no)
        num1 = num1.zfill(3)

        for filename in axial:
            if num1 in filename:
                new_filename = path + "/" + filename
                a_config.append(new_filename)
                break

        for filename in coronal:
            if num1 in filename:
                new_filename = path + "/" + filename
                c_config.append(new_filename)
                break

        for filename in sagital:
            if num1 in filename:
                new_filename = path + "/" + filename
                s_config.append(new_filename)
                break

        for filename in full_config:
            if num1 in filename:
                new_filename = path + "/" + filename
                f_config.append(new_filename)
                break

        for filename in axial_seg:
            if num1 in filename:
                new_filename = path + "/" + filename
                a_seg_config.append(new_filename)
                print(new_filename)
                break

        for filename in coronal_seg:
            if num1 in filename:
                new_filename = path + "/" + filename
                c_seg_config.append(new_filename)
                print(new_filename)
                break

        for filename in sagital_seg:
            if num1 in filename:
                new_filename = path + "/" + filename
                s_seg_config.append(new_filename)
                print(new_filename)
                break


    #
    all = np.vstack([f_config, c_config, a_config, s_config, c_seg_config, a_seg_config, s_seg_config])
    np.random.seed(100)
    np.random.shuffle(all.T)



    img_train = open('Skull_Reconstruction_from_2DStdPlanes_img_training_K1_v2.cfg', 'w')
    img_train_id = open('Skull_Reconstruction_from_2DStdPlanes_img_training_K1_ID_v2.cfg', 'w')

    img_val = open('Skull_Reconstruction_from_2DStdPlanes_img_test_K1_v2.cfg', 'w')
    img_val_id = open('Skull_Reconstruction_from_2DStdPlanes_img_test_K1_ID_v2.cfg', 'w')

    gt_train = open('Skull_Reconstruction_from_2DStdPlanes_gt_training_K1_v2.cfg', 'w')
    gt_train_id = open('Skull_Reconstruction_from_2DStdPlanes_gt_training_K1_ID_v2.cfg', 'w')

    gt_val = open('Skull_Reconstruction_from_2DStdPlanes_gt_test_K1_v2.cfg', 'w')
    gt_val_id = open('Skull_Reconstruction_from_2DStdPlanes_gt_test_K1_ID_v2.cfg', 'w')

    coronal_train = open('Skull_Reconstruction_from_2DStdPlanes_cor_training_K1_v2.cfg', 'w')
    coronal_train_id = open('Skull_Reconstruction_from_2DStdPlanes_cor_training_K1_ID_v2.cfg', 'w')

    sag_train = open('Skull_Reconstruction_from_2DStdPlanes_sag_training_K1_v2.cfg', 'w')
    sag_train_id = open('Skull_Reconstruction_from_2DStdPlanes_sag_training_K1_ID_v2.cfg', 'w')

    axial_train = open('Skull_Reconstruction_from_2DStdPlanes_trvent_training_K1_v2.cfg', 'w')
    axial_train_id = open('Skull_Reconstruction_from_2DStdPlanes_trvent_training_K1_ID_v2.cfg', 'w')

    coronal_val = open('Skull_Reconstruction_from_2DStdPlanes_cor_test_K1_v2.cfg', 'w')
    coronal_val_id = open('Skull_Reconstruction_from_2DStdPlanes_cor_test_K1_ID_v2.cfg', 'w')

    sag_val = open('Skull_Reconstruction_from_2DStdPlanes_sag_test_K1_v2.cfg', 'w')
    sag_val_id = open('Skull_Reconstruction_from_2DStdPlanes_sag_test_K1_ID_v2.cfg', 'w')

    axial_val = open('Skull_Reconstruction_from_2DStdPlanes_trvent_test_K1_v2.cfg', 'w')
    axial_val_id = open('Skull_Reconstruction_from_2DStdPlanes_trvent_test_K1_ID_v2.cfg', 'w')

    coronal_seg_train = open('Skull_Reconstruction_from_2DStdPlanes_cor_seg_training_K1_v2.cfg', 'w')
    coronal_seg_train_id = open('Skull_Reconstruction_from_2DStdPlanes_cor_seg_training_K1_ID_v2.cfg', 'w')

    sag_seg_train = open('Skull_Reconstruction_from_2DStdPlanes_sag_seg_training_K1_v2.cfg', 'w')
    sag_seg_train_id = open('Skull_Reconstruction_from_2DStdPlanes_sag_seg_training_K1_ID_v2.cfg', 'w')

    axial_seg_train = open('Skull_Reconstruction_from_2DStdPlanes_trvent_seg_training_K1_v2.cfg', 'w')
    axial_seg_train_id = open('Skull_Reconstruction_from_2DStdPlanes_trvent_seg_training_K1_ID_v2.cfg', 'w')

    coronal_seg_val = open('Skull_Reconstruction_from_2DStdPlanes_cor_seg_test_K1_v2.cfg', 'w')
    coronal_seg_val_id = open('Skull_Reconstruction_from_2DStdPlanes_cor_seg_test_K1_ID_v2.cfg', 'w')

    sag_seg_val = open('Skull_Reconstruction_from_2DStdPlanes_sag_seg_test_K1_v2.cfg', 'w')
    sag_seg_val_id = open('Skull_Reconstruction_from_2DStdPlanes_sag_seg_test_K1_ID_v2.cfg', 'w')

    axial_seg_val = open('Skull_Reconstruction_from_2DStdPlanes_trvent_seg_test_K1_v2.cfg', 'w')
    axial_seg_val_id = open('Skull_Reconstruction_from_2DStdPlanes_trvent_seg_test_K1_ID_v2.cfg', 'w')

    for item in all[1][:120]:
        coronal_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        coronal_train_id.write("%s\n" % num)


    for item in all[1][120:]:
        coronal_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        coronal_val_id.write("%s\n" % num)


    for item in all[2][:120]:
        axial_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        axial_train_id.write("%s\n" % num)

    for item in all[2][120:]:
        axial_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        axial_val_id.write("%s\n" % num)

    for item in all[3][:120]:
        sag_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        sag_train_id.write("%s\n" % num)

    for item in all[3][120:]:
        sag_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        sag_val_id.write("%s\n" % num)

    ########################## Ground Truth ######################################

    for item in all[0][:120]:
        img_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        img_train_id.write("%s\n" % num)

    for item in all[0][120:]:
        img_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        img_val_id.write("%s\n" % num)

    for item in all[0][:120]:
        gt_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        gt_train_id.write("%s\n" % num)

    for item in all[0][120:]:
        gt_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        gt_val_id.write("%s\n" % num)

    ############################ Segementation #################################

    for item in all[4][:120]:
        coronal_seg_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        coronal_seg_train_id.write("%s\n" % num)


    for item in all[4][120:]:
        coronal_seg_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        coronal_seg_val_id.write("%s\n" % num)


    for item in all[5][:120]:
        axial_seg_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        axial_seg_train_id.write("%s\n" % num)

    for item in all[5][120:]:
        axial_seg_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        axial_seg_val_id.write("%s\n" % num)

    for item in all[6][:120]:
        sag_seg_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        sag_seg_train_id.write("%s\n" % num)

    for item in all[6][120:]:
        sag_seg_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        sag_seg_val_id.write("%s\n" % num)


def generate_all():
    axial_seg = []
    coronal_seg = []
    sagital_seg = []
    full_config =[]
    axial = []
    coronal = []
    sagital = []

    # create a boolean array to verify whether a number is present in an array#
    id_val = np.zeros((413))

    # create an int array to count the number of alternate views #
    cor_view = np.zeros((413), dtype=int)
    sag_view = np.zeros((413), dtype=int)
    axial_view = np.zeros((413), dtype=int)
    full_view = np.zeros((413), dtype=int)

    for filename in os.listdir(path):
        if filename.count('_') == 4:
            if "Cor" in filename:
                left = filename.split('_')
                num = left[1]
                num = int(num)
                id_val[num] = True
                cor_view[num] += 1
                coronal_seg.append(filename)
            elif "Sagital" in filename:
                left = filename.split('_')
                num = left[1]
                num = int(num)

                sag_view[num] += 1
                sagital_seg.append(filename)
            elif "TrVent" in filename:
                left = filename.split('_')
                num = left[1]
                num = int(num)

                axial_view[num] += 1
                axial_seg.append(filename)
        if filename.count('_') == 3:
            if "Cor" in filename:
                coronal.append(filename)
            elif "Sagital" in filename:
                sagital.append(filename)
            elif "TrVent" in filename:
                axial.append(filename)
        elif filename.count('_') == 2:
            left = filename.split('_')
            num = left[1]
            num = int(num)

            full_view[num] += 1
            full_config.append(filename)


    axial_seg.sort()
    coronal_seg.sort()
    sagital_seg.sort()
    axial.sort()
    coronal.sort()
    sagital.sort()
    full_config.sort()

    ### create permutations ######
    for no in range(j):
        if id_val[no] == False:
            continue

        num1 = str(no)
        num1 = num1.zfill(3)

        ##### create permutations ######
        permutations = list(itertools.product(range(1,cor_view[no] +1), range(1,sag_view[no] +1),
                        range(1,axial_view[no] +1), range(1, full_view[no]+1)))

        for a in permuations:

            #### coronal ######
            cor_filename = path + "/ifind_" + num1 + "_Cor_" + str(a[0]) + ".nii.gz"
            cor_seg_name = path + "/ifind_" + num1 + "_Cor_" + str(a[0]) + "_seg.nii.gz"


            ##### sagital #######
            sag_filename = path + "/ifind_" + num1 + "_Sagital_" + str(a[1]) + ".nii.gz"
            sag_seg_name = path + "/ifind_" + num1 + "_Sagital_" + str(a[1]) + "_seg.nii.gz"

            ##### axial ########
            ax_filename = path + "/ifind_" + num1 + "_TrVent_" + str(a[2]) + ".nii.gz"
            ax_seg_name = path + "/ifind_" + num1 + "_TrVent_" + str(a[2]) + "_seg.nii.gz"

            ###### full #######
            full_file = path + "/ifind_" + num1 + "_" + str(a[3]) + ".nii.gz"

            c_config.append(cor_filename)
            a_config.append(ax_filename)
            s_config.append(sag_filename)

            c_seg_config.append(cor_seg_name)
            a_seg_config.append(ax_seg_name)
            s_config.append(sag_seg_name)

            f_config.append(full_file)


############## shuffle ##############################
    all = np.vstack([f_config, c_config, a_config, s_config, c_seg_config, a_seg_config, s_seg_config])
    np.random.seed(100)
    np.random.shuffle(all.T)


########### create files ################################################
    img_train = open('Skull_Reconstruction_from_2DStdPlanes_img_training_K1_v2.cfg', 'w')
    img_train_id = open('Skull_Reconstruction_from_2DStdPlanes_img_training_K1_ID_v2.cfg', 'w')

    img_val = open('Skull_Reconstruction_from_2DStdPlanes_img_test_K1_v2.cfg', 'w')
    img_val_id = open('Skull_Reconstruction_from_2DStdPlanes_img_test_K1_ID_v2.cfg', 'w')

    gt_train = open('Skull_Reconstruction_from_2DStdPlanes_gt_training_K1_v2.cfg', 'w')
    gt_train_id = open('Skull_Reconstruction_from_2DStdPlanes_gt_training_K1_ID_v2.cfg', 'w')

    gt_val = open('Skull_Reconstruction_from_2DStdPlanes_gt_test_K1_v2.cfg', 'w')
    gt_val_id = open('Skull_Reconstruction_from_2DStdPlanes_gt_test_K1_ID_v2.cfg', 'w')

    coronal_train = open('Skull_Reconstruction_from_2DStdPlanes_cor_training_K1_v2.cfg', 'w')
    coronal_train_id = open('Skull_Reconstruction_from_2DStdPlanes_cor_training_K1_ID_v2.cfg', 'w')

    sag_train = open('Skull_Reconstruction_from_2DStdPlanes_sag_training_K1_v2.cfg', 'w')
    sag_train_id = open('Skull_Reconstruction_from_2DStdPlanes_sag_training_K1_ID_v2.cfg', 'w')

    axial_train = open('Skull_Reconstruction_from_2DStdPlanes_trvent_training_K1_v2.cfg', 'w')
    axial_train_id = open('Skull_Reconstruction_from_2DStdPlanes_trvent_training_K1_ID_v2.cfg', 'w')

    coronal_val = open('Skull_Reconstruction_from_2DStdPlanes_cor_test_K1_v2.cfg', 'w')
    coronal_val_id = open('Skull_Reconstruction_from_2DStdPlanes_cor_test_K1_ID_v2.cfg', 'w')

    sag_val = open('Skull_Reconstruction_from_2DStdPlanes_sag_test_K1_v2.cfg', 'w')
    sag_val_id = open('Skull_Reconstruction_from_2DStdPlanes_sag_test_K1_ID_v2.cfg', 'w')

    axial_val = open('Skull_Reconstruction_from_2DStdPlanes_trvent_test_K1_v2.cfg', 'w')
    axial_val_id = open('Skull_Reconstruction_from_2DStdPlanes_trvent_test_K1_ID_v2.cfg', 'w')

    coronal_seg_train = open('Skull_Reconstruction_from_2DStdPlanes_cor_seg_training_K1_v2.cfg', 'w')
    coronal_seg_train_id = open('Skull_Reconstruction_from_2DStdPlanes_cor_seg_training_K1_ID_v2.cfg', 'w')

    sag_seg_train = open('Skull_Reconstruction_from_2DStdPlanes_sag_seg_training_K1_v2.cfg', 'w')
    sag_seg_train_id = open('Skull_Reconstruction_from_2DStdPlanes_sag_seg_training_K1_ID_v2.cfg', 'w')

    axial_seg_train = open('Skull_Reconstruction_from_2DStdPlanes_trvent_seg_training_K1_v2.cfg', 'w')
    axial_seg_train_id = open('Skull_Reconstruction_from_2DStdPlanes_trvent_seg_training_K1_ID_v2.cfg', 'w')

    coronal_seg_val = open('Skull_Reconstruction_from_2DStdPlanes_cor_seg_test_K1_v2.cfg', 'w')
    coronal_seg_val_id = open('Skull_Reconstruction_from_2DStdPlanes_cor_seg_test_K1_ID_v2.cfg', 'w')

    sag_seg_val = open('Skull_Reconstruction_from_2DStdPlanes_sag_seg_test_K1_v2.cfg', 'w')
    sag_seg_val_id = open('Skull_Reconstruction_from_2DStdPlanes_sag_seg_test_K1_ID_v2.cfg', 'w')

    axial_seg_val = open('Skull_Reconstruction_from_2DStdPlanes_trvent_seg_test_K1_v2.cfg', 'w')
    axial_seg_val_id = open('Skull_Reconstruction_from_2DStdPlanes_trvent_seg_test_K1_ID_v2.cfg', 'w')

    for item in all[1][:120]:
        coronal_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        coronal_train_id.write("%s\n" % num)


    for item in all[1][120:]:
        coronal_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        coronal_val_id.write("%s\n" % num)


    for item in all[2][:120]:
        axial_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        axial_train_id.write("%s\n" % num)

    for item in all[2][120:]:
        axial_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        axial_val_id.write("%s\n" % num)

    for item in all[3][:120]:
        sag_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        sag_train_id.write("%s\n" % num)

    for item in all[3][120:]:
        sag_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        sag_val_id.write("%s\n" % num)

    ########################## Ground Truth ######################################

    for item in all[0][:120]:
        img_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        img_train_id.write("%s\n" % num)

    for item in all[0][120:]:
        img_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        img_val_id.write("%s\n" % num)

    for item in all[0][:120]:
        gt_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        gt_train_id.write("%s\n" % num)

    for item in all[0][120:]:
        gt_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        gt_val_id.write("%s\n" % num)

    ############################ Segementation #################################

    for item in all[4][:120]:
        coronal_seg_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        coronal_seg_train_id.write("%s\n" % num)


    for item in all[4][120:]:
        coronal_seg_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        coronal_seg_val_id.write("%s\n" % num)


    for item in all[5][:120]:
        axial_seg_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        axial_seg_train_id.write("%s\n" % num)

    for item in all[5][120:]:
        axial_seg_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        axial_seg_val_id.write("%s\n" % num)

    for item in all[6][:120]:
        sag_seg_train.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        sag_seg_train_id.write("%s\n" % num)

    for item in all[6][120:]:
        sag_seg_val.write("%s\n" % item)
        left = item.split('_')
        num = left[2]
        num = num.zfill(3)
        sag_seg_val_id.write("%s\n" % num)

if __name__ == '__main__':
    #generate_all()
    a = list(itertools.product(range(1,6), range(1,3), range(1,2)))

    for i in a:
        print(i[0])
