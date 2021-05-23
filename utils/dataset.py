import glob
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LiverClassificationDataset(Dataset):
    def __init__(self, data_dir, image_transforms):
        self.data_dir = data_dir
        self.image_transforms = image_transforms

        images_dir = os.listdir(os.path.join(data_dir, 'image'))
        labels_dir = os.listdir(os.path.join(data_dir, 'label'))

        self.images_list = []
        self.labels_list = []
        for dir in images_dir:
            self.images_list.extend(glob.glob(os.path.join(data_dir, 'image', dir, '*.png')))
        for dir in labels_dir:
            self.labels_list.extend(glob.glob(os.path.join(data_dir, 'label', dir, '*.png')))

        self.images_list.sort()
        self.labels_list.sort()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])
        image = self.image_transforms(image)

        label = Image.open(self.labels_list[idx])
        label = np.array(label)
        count = np.count_nonzero(label)

        out_class = 0 if count == 0 else 1

        return image, out_class

class LiverSegmentationDataset(Dataset):
    def __init__(self, data_dir, segmentation_txt, image_transforms):
        self.data_dir = data_dir
        self.image_transforms = image_transforms

        self.images_list = []
        self.labels_list = []

        seg_list = open(segmentation_txt).readlines()
        for seg in seg_list:
            self.labels_list.append(os.path.join(data_dir, 'label', seg[:-1]))
            self.images_list.append(os.path.join(data_dir, 'image', seg[:-1].replace('segmentation', 'volume')))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])
        image = self.image_transforms(image)

        label = Image.open(self.labels_list[idx])
        label = label.resize(size=(256, 256), resample=0)
        label = np.array(label)
        label = np.where(label == 255, 1, label)
        label = np.where(label == 128, 2, label)
        mask = torch.from_numpy(label)

        return image, mask

def main():
    train_dir = 'D:/pytorch/Segmentation/Liver_Tumor/visualized/train'
    image_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset_c = LiverClassificationDataset(train_dir, image_transforms)
    dataset_s = LiverSegmentationDataset(train_dir, image_transforms=image_transforms)
    print(dataset_c.__len__())
    print(dataset_s.__len__())

if __name__ == '__main__':
    main()