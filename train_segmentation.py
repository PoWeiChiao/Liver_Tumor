import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.UNet import UNet
from utils.dataset import LiverSegmentationDataset
from utils.logger import Logger

def train(net, device, dataset_train, dataset_val, batch_size=4, epochs=50, lr=0.00001):
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(params=net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = np.inf

    log_train = Logger('log_train_segmentation.txt')
    log_val = Logger('log_val_segmentation.txt')

    for epoch in range(epochs):
        loss_train = 0.0
        loss_val = 0.0
        print('running epoch {}'.format(epoch))

        net.train()
        for image, label in tqdm(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = loss_fn(pred, label)
            loss_train += loss.item() * image.size(0)

            loss.backward()
            optimizer.step()

        net.eval()
        for image, label in tqdm(val_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = loss_fn(pred, label)
            loss_val += loss.item() * image.size(0)

        loss_train = loss_train / len(train_loader.dataset)
        loss_val = loss_val / len(val_loader.dataset)
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(loss_train, loss_val))
        log_train.write_line(str(epoch) + ',' + str(round(loss_train, 6)))
        log_val.write_line(str(epoch) + ',' + str(round(loss_val, 6)))

        if loss_val <= best_loss:
            torch.save(net.state_dict(), 'model_segmentation.pth')
            best_loss = loss_val
            print('model saved')
        if epoch >= 40:
            torch.save(net.state_dict(), 'model_segmentation_' + str(epoch) + '.pth')
            print('model saved')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ' + str(device))

    net = UNet(n_channels=1, n_classes=3)
    net.to(device=device)

    train_dir = 'visualized/train'
    val_dir = 'visualized/val'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])
    dataset_train = LiverSegmentationDataset(data_dir=train_dir, segmentation_txt=os.path.join(train_dir, 'train_seg.txt'), image_transforms=image_transforms)
    dataset_val = LiverSegmentationDataset(data_dir=val_dir, segmentation_txt=os.path.join(val_dir, 'val_seg.txt'), image_transforms=image_transforms)

    train(net, device, dataset_train, dataset_val)

if __name__ == '__main__':
    main()