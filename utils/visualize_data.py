import cv2 as cv
import glob
import nibabel as nib
import numpy as np
import os

def nii2png(save_dir, image_path, is_label=False):
    num = 0
    os.mkdir(os.path.join(save_dir, os.path.basename(image_path)[:-4]))
    image = nib.load(image_path)
    fdata = image.get_fdata()
    x, y, z = fdata.shape
    for i in range(z):
        image_arr = fdata[:, :, i].astype(np.uint8)
        if is_label:
            image_arr = np.where(image_arr == 1, 255, image_arr)
            image_arr = np.where(image_arr == 2, 128, image_arr)
        cv.imwrite(os.path.join(save_dir, os.path.basename(image_path)[:-4], os.path.basename(image_path)[:-4] + '-' + str(num) + '.png'), image_arr)
        num += 1

def main():
    train_image_dir = 'data/train/image'
    train_label_dir = 'data/train/label'
    test_image_dir = 'data/test/image'
    train_image_path = glob.glob(os.path.join(train_image_dir, '*.nii'))
    train_label_path = glob.glob(os.path.join(train_label_dir, '*.nii'))
    test_image_path = glob.glob(os.path.join(test_image_dir, '*.nii'))
    train_image_path.sort()
    train_label_path.sort()
    test_image_path.sort()

    train_image_save = 'visualized/train/image'
    train_label_save = 'visualized/train/label'
    test_image_save = 'visualized/test/image'

    for train_label in train_label_path:
        nii2png(train_label_save, train_label, True)

    for train_image in train_image_path:
        nii2png(train_image_save, train_image)

    for test_image in test_image_path:
        nii2png(test_image_save, test_image)

if __name__ == '__main__':
    main()