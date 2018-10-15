import h5py
import numpy as np
import nibabel as nb

hfTrain = h5py.File('/vol/gpudata/wt814/hvols_3D/segms3d_train_reg_80.h5', 'r')
hfEvaluate= h5py.File('/vol/gpudata/wt814/hvols_3D/segms3d_eval_reg_80.h5', 'r')
hfTest = h5py.File('/vol/gpudata/wt814/hvols_3D/segms3d_test_reg_80.h5', 'r')




def get_slices():
    a1 = hfTrain.get("segms3d_train")
    a2 = hfEvaluate.get("segms3d_eval")
    a3 = hfTest.get("segms3d_test")
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)

    print(a1.shape)
    slice_1 = a1[:,40,:,:,:]
    slice_2 = a1[:,:,40,:,:]
    slice_3 = a1[:,:,:,40,:]

    print(slice_1.shape)
    print(slice_2.shape)
    print(slice_3.shape)

    np.save('hvols_2D/train/plane_1.npy', slice_1)
    np.save('hvols_2D/train/plane_2.npy', slice_2)
    np.save('hvols_2D/train/plane_3.npy', slice_3)

    slice_1 = a2[:,40,:,:,:]
    slice_2 = a2[:,:,40,:,:]
    slice_3 = a2[:,:,:,40,:]

    np.save('hvols_2D/evaluate/plane_1.npy', slice_1)
    np.save('hvols_2D/evaluate/plane_2.npy', slice_2)
    np.save('hvols_2D/evaluate/plane_3.npy', slice_3)

    slice_1 = a3[:,40,:,:,:]
    slice_2 = a3[:,:,40,:,:]
    slice_3 = a3[:,:,:,40,:]

    np.save('hvols_2D/test/plane_1.npy', slice_1)
    np.save('hvols_2D/test/plane_2.npy', slice_2)
    np.save('hvols_2D/test/plane_3.npy', slice_3)

def load_train():
    a1 = hfTrain.get("segms3d_train")
    full = np.array(a1)

    x_1 = np.load('hvols_2D/train/plane_1.npy')
    x_2 = np.load('hvols_2D/train/plane_2.npy')
    x_3 = np.load('hvols_2D/train/plane_3.npy')

    return full, x_1, x_2, x_3

def load_test():
    a1 = hfTest.get("segms3d_test")
    full = np.array(a1)

    x_1 = np.load('hvols_2D/test/plane_1.npy')
    x_2 = np.load('hvols_2D/test/plane_2.npy')
    x_3 = np.load('hvols_2D/test/plane_3.npy')

    return full, x_1, x_2, x_3

def load_evaluate():
    a1 = hfEvaluate.get("segms3d_eval")
    full = np.array(a1)

    x_1 = np.load('hvols_2D/evaluate/plane_1.npy')
    x_2 = np.load('hvols_2D/evaluate/plane_2.npy')
    x_3 = np.load('hvols_2D/evaluate/plane_3.npy')
    return full, x_1, x_2, x_3

if __name__ == '__main__':
    load_test()
