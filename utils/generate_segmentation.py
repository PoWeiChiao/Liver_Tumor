import glob
import os
import numpy as np
from PIL import Image
from logger import Logger

def main():
    train_data_dir = 'D:/pytorch/Segmentation/Liver_Tumor/visualized/train'
    val_data_dir = 'D:/pytorch/Segmentation/Liver_Tumor/visualized/val'

    train_seg = Logger('D:/pytorch/Segmentation/Liver_Tumor/visualized/train/train_seg.txt')
    val_seg = Logger('D:/pytorch/Segmentation/Liver_Tumor/visualized/val/val_seg.txt')

    train_label = os.listdir(os.path.join(train_data_dir, 'label'))
    val_label = os.listdir(os.path.join(val_data_dir, 'label'))

    for label in train_label:
        images = glob.glob(os.path.join(train_data_dir, 'label', label, '*.png'))
        for image in images:
            im = Image.open(image)
            im = np.array(im)
            if np.count_nonzero(im) > 0:
                train_seg.write_line(os.path.join(label, os.path.basename(image)))

    for label in val_label:
        images = glob.glob(os.path.join(val_data_dir, 'label', label, '*.png'))
        for image in images:
            im = Image.open(image)
            im = np.array(im)
            if np.count_nonzero(im) > 0:
                val_seg.write_line(os.path.join(label, os.path.basename(image)))

if __name__ == '__main__':
    main()
