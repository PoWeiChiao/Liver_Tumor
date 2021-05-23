import glob
import os
from utils.generate_val import generate_val
from utils.visualize_data import nii2png

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

    train_dir = 'visualized/train'
    val_dir = 'visualized/val'

    generate_val(train_dir, val_dir)

if __name__ == '__main__':
    main()

    