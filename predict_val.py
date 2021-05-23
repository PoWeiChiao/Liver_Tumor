import cv2 as cv
import glob
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model.ResNet import BasicBlock, BottleNeck, ResNet
from model.UNet import UNet
from utils.dataset import LiverClassificationDataset, LiverSegmentationDataset

def predict(net_segmentation, device, data_dir, save_dir, image_transforms):
    images_path = os.listdir(os.path.join(data_dir, 'image'))
    masks_path = os.listdir(os.path.join(data_dir, 'label'))

    images_path.sort()
    masks_path.sort()

    net_segmentation.eval()
    with torch.no_grad():
        for i in range(len(images_path)):
            images_list = glob.glob(os.path.join(data_dir, 'image', images_path[i], '*.png'))
            masks_list = glob.glob(os.path.join(data_dir, 'label', masks_path[i], '*.png'))

            images_list.sort()
            masks_list.sort()
            for j in range(len(images_list)):
                mask = Image.open(masks_list[j])
                mask = mask.resize((256, 256), 0)
                mask = np.array(mask)
                if np.count_nonzero(mask) > 0:
                    image = Image.open(images_list[j])
                    image_out = image.copy()
                    image_out = image_out.resize((256, 256))
                    image_out = np.array(image_out)
                    image = image_transforms(image)
                    image = image.unsqueeze(0)
                    image = image.to(device=device, dtype=torch.float32)
                    pred_mask = net_segmentation(image)
                    pred_mask = F.softmax(pred_mask, dim=1)
                    pred_mask = pred_mask.argmax(1)
                    pred_mask = np.array(pred_mask.data.cpu()[0])
                    pred_mask = np.where(pred_mask==1, 255, pred_mask)
                    pred_mask = np.where(pred_mask==2, 128, pred_mask)
                    pred_mask = np.array(pred_mask, dtype=np.uint8)
                    
                    result = np.concatenate((image_out, mask), axis=1)
                    result = np.concatenate((result, pred_mask), axis=1)
                    cv.imwrite(os.path.join(save_dir, os.path.basename(images_list[j])), result)
                    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ' + str(device)) 

    test_dir = 'visualized/val'
    save_dir = 'visualized/val/predict'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])

    net_segmentation = UNet(n_channels=1, n_classes=3)
    net_segmentation.load_state_dict(torch.load('saved/20210430_ResNet_UNet/model_segmentation.pth', map_location=device))
    net_segmentation.to(device=device)

    predict(net_segmentation, device, test_dir, save_dir, image_transforms)

if __name__ == '__main__':
    main()



            

