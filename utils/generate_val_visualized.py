import cv2 as cv
import glob
import os
import numpy as np 
from PIL import Image

def main():
    image_dir = 'D:/pytorch/Segmentation/Liver_Tumor/visualized/val/image'
    mask_dir = 'D:/pytorch/Segmentation/Liver_Tumor/visualized/val/label'

    images_path = os.listdir(image_dir)
    masks_path = os.listdir(mask_dir)

    images_path.sort()
    masks_path.sort()

    for i in range(len(images_path)):
        images_list = glob.glob(os.path.join(image_dir, images_path[i], '*.png'))
        masks_list = glob.glob(os.path.join(mask_dir, masks_path[i], '*.png'))
        images_list.sort()
        masks_list.sort()
        for j in range(len(images_list)):
            image = Image.open(images_list[j])
            image = image.resize((256, 256))
            mask = Image.open(masks_list[j])
            mask = mask.resize((256, 256), 0)
            image = np.array(image)
            mask = np.array(mask)
            result = np.concatenate((image, mask), axis=1)
            cv.imwrite(os.path.join('D:/pytorch/Segmentation/Liver_Tumor/visualized/val/', 'visualized', os.path.basename(images_list[j])), result)

if __name__ == '__main__':
    main()