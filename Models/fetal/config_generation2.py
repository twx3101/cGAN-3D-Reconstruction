import os
import numpy as np
path = os.path.abspath("./data_newOK")
print(path)

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
axial_seg = []
coronal_seg = []
sagital_seg = []
full_config =[]

for filename in os.listdir(path):
    if filename.count('_') == 3:
        if "Cor" in filename:
            coronal_seg.append(filename)
        elif "Sagital" in filename:
            sagital_seg.append(filename)
        elif "TrVent" in filename:
            axial_seg.append(filename)
    elif filename.count('_') == 2:
        full_config.append(filename)


axial_seg.sort()
coronal_seg.sort()
sagital_seg.sort()
full_config.sort()
#
a_config =[]
c_config = []
s_config = []
f_config = []
j = 413
#
# #remove duplicates
for no in range(j):
    num1 = str(no)
    num1 = num1.zfill(3)

    for filename in axial_seg:
        if num1 in filename:
            new_filename = path + "/" + filename
            a_config.append(new_filename)
            break

    for filename in coronal_seg:
        if num1 in filename:
            new_filename = path + "/" + filename
            c_config.append(new_filename)
            break

    for filename in sagital_seg:
        if num1 in filename:
            new_filename = path + "/" + filename
            s_config.append(new_filename)
            break

    for filename in full_config:
        if num1 in filename:
            new_filename = path + "/" + filename
            f_config.append(new_filename)
            break



#
all = np.vstack([f_config, c_config, a_config, s_config])

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
