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

def predict(net_classification, net_segmentation, device, data_dir, save_dir, image_transforms):
    images_path = os.listdir(os.path.join(data_dir, 'image'))

    net_classification.eval()
    net_segmentation.eval()
    with torch.no_grad():
        for path in images_path:
            images_list = glob.glob(os.path.join(data_dir, 'image', path, '*.png'))
            images_list.sort()
            for image_path in images_list:
                image = Image.open(image_path)
                image_out = image.copy()
                image_out = image_out.resize((256, 256))
                image_out = np.array(image_out)
                image = image_transforms(image)
                image = image.unsqueeze(0)
                image = image.to(device=device, dtype=torch.float32)
                pred = net_classification(image)
                if pred.argmax(1) == 1:
                    pred_mask = net_segmentation(image)
                    pred_mask = F.softmax(pred_mask, dim=1)
                    pred_mask = pred_mask.argmax(1)
                    pred_mask = np.array(pred_mask.data.cpu()[0])
                    pred_mask = np.where(pred_mask==1, 255, pred_mask)
                    pred_mask = np.where(pred_mask==2, 128, pred_mask)
                    pred_mask = np.array(pred_mask, dtype=np.uint8)
                    
                    result = np.concatenate((image_out, pred_mask), axis=1)
                    cv.imwrite(os.path.join(save_dir, os.path.basename(image_path)), result)
                    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ' + str(device)) 

    test_dir = 'visualized/test'
    save_dir = 'predict'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])

    net_classification = ResNet(in_channel=1, n_classes=2, block=BottleNeck, num_block=[3, 4, 6, 3])
    net_classification.load_state_dict(torch.load('saved/20210430_ResNet_UNet/model_classification.pth', map_location=device))
    net_classification.to(device=device)

    net_segmentation = UNet(n_channels=1, n_classes=3)
    net_segmentation.load_state_dict(torch.load('saved/20210430_ResNet_UNet/model_segmentation.pth', map_location=device))
    net_segmentation.to(device=device)

    predict(net_classification, net_segmentation, device, test_dir, save_dir, image_transforms)

if __name__ == '__main__':
    main()



            

