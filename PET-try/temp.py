import os
import scipy.io as sio

img_list = os.listdir('data/MyData/test_data/images')
for img_name in img_list:
    img_path = f"./data/MyData/test_data/images/{img_name}"
    gt_path = f"./data/MyData/test_data/ground_truth/GT_{img_name}"
    gt_path = gt_path.replace('jpg','mat')
    if not os.path.exists(gt_path):
        mat = sio.loadmat('GT_IMG_0.mat')
        sio.savemat(gt_path, mat)