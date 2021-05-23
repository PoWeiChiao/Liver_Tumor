import glob
import os
import random
import shutil

def generate_val(train_dir, val_dir):
    train_list = []
    for i in range(131):
        train_list.append(i)
    random.shuffle(train_list)

    for i in range(13):
        shutil.move(os.path.join(train_dir, 'image', 'volume-' + str(train_list[i])), os.path.join(val_dir, 'image', 'volume-' + str(train_list[i])))
        shutil.move(os.path.join(train_dir, 'label', 'segmentation-' + str(train_list[i])), os.path.join(val_dir, 'label', 'segmentation-' + str(train_list[i])))

def main():
    train_dir = 'visualized/train'
    val_dir = 'visualized/val'

    generate_val(train_dir, val_dir)

if __name__ == '__main__':
    main()