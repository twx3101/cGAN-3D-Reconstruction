import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from skimage import filters
from nipype.algorithms import metrics
import load_3d
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, OPTICS



def histogram(data):
    data = data.flatten()
    plt.hist(data, 256, range=(0,1), histtype='bar')
    plt.show()

def binarisation(data, threshold=0.9):
    #val = filters.threshold_otsu(data)
    binary_data = (data > threshold).astype(np.float64)
    #print(val)
    return binary_data



def Hausdorff(file1, file2):
    value = metrics.Distance()
    value.inputs.volume1 = file1
    value.inputs.volume2 = file2
    value.inputs.method = 'eucl_max'
    result = value.run()
    return result.outputs.distance

# DICE Coefficient

def DICE(prediction, ground_truth):
    if prediction.shape != ground_truth.shape:
        raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
    else:
        intersection = np.logical_and(prediction, ground_truth)
        value = (2. * intersection.sum())  / (prediction.sum() + ground_truth.sum())
        return value

def SampleTrain():

    path = os.path.join(".", "data_newOK")
    path2 = os.path.join(".", "niftiImage/split/Test")

    filenames = os.listdir(path)
    filenames.sort()

    filenames2 = os.listdir(path2)
    filenames2.sort()
    #print(filenames2)

    imagefile = os.path.join(path, filenames[0])
    b = []
    for name in filenames2:
        imagefile2 = os.path.join(path2, name)

        img = nib.load(imagefile)
        img2 = nib.load(imagefile2)
        data = img.get_fdata()
        data2 = img2.get_fdata()

    #histogram(data2)
        a = binarisation(data2, 0.5)
    #print(data)
    #print(np.array_equal(a, data.astype(bool)))
    #print(data.dtype)
        affine = load_3d.get_affine_3d()
        save_img = nib.Nifti1Image(a, affine)

    #nib.save(save_img, 'check/1000.nii.gz')
        diff = DICE(a, data)
        b.append(diff)
        print(diff)#
        c = np.asarray(b)
    #np.savetxt('test', c)

def SampleTest(path2, fold_no):

    path = os.path.join(".", "data_newOK")
    #path2 = os.path.join(".", "niftiImage/genIterReal/Sample")
    id = os.path.join(".", "BatchGeneratorClass/config/k-fold_config")
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



    b = []
    idpath = os.path.join(id, 'Skull_Reconstruction_from_2DStdPlanes_gt_test_K1_v2.cfg')
    f = open(idpath, 'r')
    id_val = f.read().splitlines()
    f.close()
    i = 0
    for file in os.listdir(path2):
        print(file)
        if file.count('.') == 2:
            imagefile2 = os.path.join(path2, file)
            x = id_val[i]

            img = nib.load(x)
            img2 = nib.load(imagefile2)
            data = img.get_fdata()
            data2 = img2.get_fdata()
    #histogram(data2)
            a = data2
    #print(data)
    #print(np.array_equal(a, data.astype(bool)))
    #print(data.dtype)
    #affine = load_3d.get_affine_3d()
    #save_img = nib.Nifti1Image(a, affine)

    #nib.save(save_img, 'check/1000.nii.gz')
            diff = DICE(a, data)
            b.append(diff)

            i += 1
        #print(diff)#
    c = np.asarray(b)
    #np.savetxt("Similarity/1axis/DICE_similarity_%d.txt" %(fold_no+1), c)





def reset_Data():

    path = os.path.join(".", "data_newOK")
    path2 = os.path.join(".", "niftiImage/genIter/Sample")
    id = os.path.join(".", "BatchGeneratorClass/config/k-fold_config")
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
    affine = load_3d.get_affine_3d()
    i = 0
    for file in os.listdir(path2):
        imagefile2 = os.path.join(path2, file)
        img2 = nib.load(imagefile2)
        data2 = img2.get_data()
        save_img = nib.Nifti1Image(data2, affine)

        nib.save(save_img, 'niftiImage/genIter/SampleReset/%d.nii.gz' %i)
        i += 1

def binarise_data(path2, fold_no, threshold):
    #path = os.path.join(".", "data_newOK")
    #path2 = os.path.join(".", "niftiImage/genIterReal/Sample")
    #id = os.path.join(".", "BatchGeneratorClass/config/k-fold_config")

    affine = load_3d.get_affine_3d()
    i = 0
    for file in os.listdir(path2):
        imagefile2 = os.path.join(path2, file)
        img2 = nib.load(imagefile2)
        data2 = img2.get_data()

        a = binarisation(data2, threshold)
        save_img = nib.Nifti1Image(a, affine)
        nib.save(save_img, 'niftiImage/splitGenIter3/SampleBinaryMask/%d/%d.nii.gz' %(fold_no+1, i))
        i += 1

def clean_binarise(binary_path, fold_no):
    affine = load_3d.get_affine_3d()

    i = 0
    for file in os.listdir(binary_path):
        imagefile2 = os.path.join(binary_path, file)
        img2 = nib.load(imagefile2)
        data2 = img2.get_data()

        points = np.transpose(np.where(data2))
        points = remove_outliers(points)

        #using DBSCAN
        data = np.zeros((96,96,96))
        data[points[:,0], points[:,1], points[:,2]] = 1.
        save_img = nib.Nifti1Image(data, affine)
        nib.save(save_img, 'niftiImage/splitGenIter3/CleanBinaryMask/%d/%d.nii.gz' %(fold_no+1, i))
        i += 1

def DBSCAN_binarise(path2, fold_no, fold=False):
    affine = load_3d.get_affine_3d()

    i = 0
    for file in os.listdir(path2):
        imagefile2 = os.path.join(path2, file)
        img2 = nib.load(imagefile2)
        data2 = img2.get_data()
        a = binarisation(data2, 0.5)
        points = np.transpose(np.where(a))
        points = remove_outliers(points)

        #using DBSCAN
        points, noise = DBSCAN_outliers(points)
        data = np.zeros((96,96,96))
        data[points[:,0], points[:,1], points[:,2]] = 1.
        data[noise[:,0], noise[:,1], noise[:,2]] = 0.

        save_img = nib.Nifti1Image(data, affine)
        #nib.save(save_img, 'niftiImage/1axis/DBSCANBinaryMask/%d/%d.nii.gz' %(fold_no+1, i))
        i += 1

def full_binarise(path2, fold_no, fold=False):
    affine = load_3d.get_affine_3d()

    i = 0
    for file in os.listdir(path2):
        imagefile2 = os.path.join(path2, file)
        img2 = nib.load(imagefile2)
        data2 = img2.get_data()
        a = binarisation(data2, 0.5)
        points = np.transpose(np.where(a))
        points = remove_outliers(points)

        a = np.zeros((96,96,96))
        a[points[:,0], points[:,1], points[:,2]] = 1.

        save_img = nib.Nifti1Image(a, affine)
        nib.save(save_img, 'niftiImage/splitGenIterSave/CleanBinaryMask/%d/%d.nii.gz' %(fold_no+1, i))
        i += 1


def H_dist(path2, fold_no):
        path = os.path.join(".", "data_newOK")
        #path2 = os.path.join(".", "niftiImage/genIterReal/SampleBinaryMask")
        id = os.path.join(".", "BatchGeneratorClass/config/k-fold_config")
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


        b = []
        idpath = os.path.join(id, 'Skull_Reconstruction_from_2DStdPlanes_gt_test_K1_v2.cfg')
        f = open(idpath, 'r')
        id_val = f.read().splitlines()
        f.close()
        i = 0

        for file in os.listdir(path2):
            print(file)
            if file.count('.') == 2:
                imagefile2 = os.path.join(path2, file)
                x = id_val[i]

                h = Hausdorff(x, imagefile2)
                b.append(h)

                i += 1
            #print(diff)#
        c = np.asarray(b)
        print(c)
        #np.savetxt("Similarity/1axis/Hausdorff_%d.txt" %(fold_no+1), c)
def check():
    path = os.path.join(".", "data_newOK")
    path2 = os.path.join(".", "niftiImage/genIterReal/SampleBinaryMask")

    filenames = os.listdir(path)
    filenames.sort()
    #print(filenames[7])
    imagefile = os.path.join(path, filenames[2])
    img = nib.load(imagefile)
    data = img.get_data()
    print(img.header)

def flood_fill_hull(image):

    path = os.path.join(".", "data_newOK")

    points = np.transpose(np.where(image))
    #points = remove_outliers(points)
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1

    return out_img, hull, points

def plot_hull(points, hull):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.plot(points[:,0], points[:,1], points[:,2], 'o')


    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(points[s, 0], points[s, 1], points[s, 2], "r-")

    plt.show()

def DICE_closed_hull(image, ground_truth):
    sum_image = len(image)
    sum_gt = len(ground_truth)
    #intersect =
    return(2*intersect/(sum_image+sum_gt))

def closed_DICE(path2, fold_no):
    path = os.path.join(".", "data_newOK")
    id = os.path.join(".", "BatchGeneratorClass/config/k-fold_config")

    b = []
    idpath = os.path.join(id, 'Skull_Reconstruction_from_2DStdPlanes_gt_test_K1_v2.cfg')
    f = open(idpath, 'r')
    id_val = f.read().splitlines()
    f.close()
    i = 0
    for file in os.listdir(path2):
        #print(file)
        if file.count('.') == 2:
            imagefile2 = os.path.join(path2, file)
            x = id_val[i]
            #print(x)

            img = nib.load(x)
            img2 = nib.load(imagefile2)
            data = img.get_fdata()
            data2 = img2.get_fdata()
            out, hull, points = flood_fill_hull(data)
            out2, hull2, points2 = flood_fill_hull(data2)

            diff = DICE(out, out2)
            #if i == 3:
                #remove_outliers(points2)
                #print(x)
                #plot_hull(points2, hull2)
                #print(np.count_nonzero(out))
                #print(np.count_nonzero(out2))
            b.append(diff)

            i += 1
        #print(diff)#
    c = np.asarray(b)
    np.savetxt("Similarity/1axis/closedDICE_%d.txt" %(fold_no+1), c)

def remove_outliers(points, delta=1.):
    # mean= np.mean(points, axis=0)
    # std = np.std(points, axis=0)
    #modified z-score
    x = points.T[0]
    y = points.T[1]
    z = points.T[2]
    # med_x = np.abs(x - np.median(x))
    # med_y = np.abs(y - np.median(y))
    # med_z = np.abs(z - np.median(z))
    x0 = np.median(x)
    y0 = np.median(y)
    z0 = np.median(z)
    print(x0)
    print(y0)
    print(z0)
    # s_x =  med_x/median_x if median_x else 0
    # s_y =  med_y/median_y if median_y else 0
    # s_z =  med_z/median_z if median_z else 0

    # mask_x = s_x < delta
    # mask_y = s_y < delta
    # mask_z = s_z < delta
    #
    # mask = np.logical_and(mask_x, mask_y, mask_z)

    distances = np.sqrt((x-x0)**2+(y-y0)**2+(z-z0)**2)
    #plt.hist(distances)
    #plt.show
    dist_med = np.median(distances)
    mask = np.abs(distances-dist_med if dist_med else 0)

    point_mask = mask < delta * dist_med

    print(points.shape)
    print(points[point_mask].shape)

    return(points[point_mask])

def DBSCAN_outliers(points, distance=0.3, size=10):
    db = DBSCAN(eps=5, min_samples=100).fit(points)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(len(labels))
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")
    #plt.plot(points[:,0], points[:,1], points[:,2], 'o')

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = points[class_member_mask & core_samples_mask]
        # ax.plot(xy[:, 0], xy[:, 1], xy[:,2], 'o', markerfacecolor='b',
        #          markeredgecolor='k')

        xy = points[class_member_mask & ~core_samples_mask]
        #ax.plot(xy[:, 0], xy[:, 1], xy[:,2], '*', markerfacecolor=tuple(col))

    #plt.title('Estimated number of clusters: %d' % n_clusters_)
    #print(len(xy))
    #plt.show()
# Number of clusters in labels, ignoring noise if present.
    print(points.shape)
    print(xy.shape)
    #input("Press Enter to continue...")
    return points, xy

if __name__ == '__main__':
    #clean data then find binary masks
    #for fold_no in range(3):
    #
    #Find binary mask of data
    #    DBSCAN_binarise("niftiImage/1axis/Sample/%d" %(fold_no+1), fold_no)

    for fold_no in range(1):
    #get DICE of normal
        #full_binarise("niftiImage/splitGenIterSave/Sample/%d" %(fold_no+1), fold_no)
        #full_binarise("niftiImage/splitGenIterSave/Sample", fold_no)
        SampleTest("niftiImage/1axis/DBSCANBinaryMask/%d" %(fold_no+1), fold_no)

    #get Hausdorff Distance
    #    H_dist("niftiImage/1axis/DBSCANBinaryMask/%d" %(fold_no+1), fold_no)

    #get closed_DICE
    #    closed_DICE("niftiImage/1axis/DBSCANBinaryMask/%d" %(fold_no+1), fold_no)
