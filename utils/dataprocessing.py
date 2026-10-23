import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import random
import shutil


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
    
def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

def data_spilt():
    root = "../datasets/TVNF/imgs"
    save_root = "../datasets/TVNF"

    modalities = ["VIS", "TIR", "NIR"]
    files = sorted(os.listdir(os.path.join(root, "VIS")))

    random.seed(0)
    random.shuffle(files)

    n_total = len(files)
    n_train = int(n_total * 0.8)

    train_files = files[:n_train]
    test_files = files[n_train:]

    print("train:", len(train_files))
    print("test:", len(test_files))

    for split in ["train", "test"]:
        for m in modalities:
            os.makedirs(os.path.join(save_root, split, m), exist_ok=True)

    def copy_files(file_list, split):
        for f in file_list:
            for m in modalities:
                src = os.path.join(root, m, f)
                dst = os.path.join(save_root, split, m, f)
                shutil.copy(src, dst)

    copy_files(train_files, "train")
    copy_files(test_files, "test")

    print("done")
def patches():
    img_size=128   #patch size
    stride=200     #patch stride

    TIR_files = sorted(get_img_file(r"../datasets/TVNF/train/TIR"))
    NIR_files = sorted(get_img_file(r"../datasets/TVNF/train/NIR"))
    VIS_files   = sorted(get_img_file(r"../datasets/TVNF/train/VIS"))

    assert len(TIR_files) == len(NIR_files) == len(VIS_files)
    h5f = h5py.File(os.path.join('../datasets/TVNF/train', 'imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),
                        'w')
    h5_tir = h5f.create_group('Tir_patchs')
    h5_nir = h5f.create_group('Nir_patchs')
    h5_vis = h5f.create_group('vis_patchs')
    train_num=0
    for i in tqdm(range(len(VIS_files))):
            I_VIS = imread(VIS_files[i]).astype(np.float32).transpose(2,0,1)/255. # [3, H, W] Uint8->float32
            I_VIS = rgb2y(I_VIS) # [1, H, W] Float32

            # I_TIR = imread(TIR_files[i]).astype(np.float32)[None, :, :]/255.  # [1, H, W] Float32
            I_TIR = cv2.imread(TIR_files[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            I_TIR = I_TIR[None, :, :] / 255.
            # I_NIR = imread(NIR_files[i]).astype(np.float32)[None, :, :]/255.  # [1, H, W] Float32
            I_NIR = cv2.imread(NIR_files[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            I_NIR = I_NIR[None, :, :] / 255.


            # crop
            I_NIR_Patch_Group = Im2Patch(I_NIR,img_size,stride)
            I_TIR_Patch_Group = Im2Patch(I_TIR,img_size,stride)
            I_VIS_Patch_Group = Im2Patch(I_VIS, img_size, stride)  # (3, 256, 256, 12)

            for ii in range(I_TIR_Patch_Group.shape[-1]):
                bad_TIR = is_low_contrast(I_TIR_Patch_Group[0,:,:,ii])
                bad_NIR = is_low_contrast(I_NIR_Patch_Group[0,:,:,ii])
                bad_VIS = is_low_contrast(I_VIS_Patch_Group[0,:,:,ii])
                # Determine if the contrast is low
                if not (bad_NIR or bad_TIR or bad_VIS):
                    avl_NIR= I_NIR_Patch_Group[0,:,:,ii]  #  available IR
                    avl_TIR= I_TIR_Patch_Group[0,:,:,ii]  #  available IR
                    avl_VIS= I_VIS_Patch_Group[0,:,:,ii]
                    avl_TIR=avl_TIR[None,...]
                    avl_NIR=avl_NIR[None,...]
                    avl_VIS=avl_VIS[None,...]

                    h5_tir.create_dataset(str(train_num),     data=avl_TIR,
                                    dtype=avl_TIR.dtype,   shape=avl_TIR.shape)
                    h5_nir.create_dataset(str(train_num), data=avl_NIR,
                                          dtype=avl_NIR.dtype, shape=avl_NIR.shape)
                    h5_vis.create_dataset(str(train_num),    data=avl_VIS,
                                    dtype=avl_VIS.dtype,  shape=avl_VIS.shape)
                    train_num += 1

    h5f.close()

    with h5py.File(os.path.join('../datasets/TVNF/train', 'imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),"r") as f:
        for key in f.keys():
            print(f[key], key, f[key].name)


    
if __name__ == '__main__':
    data_spilt()
    patches()

    
