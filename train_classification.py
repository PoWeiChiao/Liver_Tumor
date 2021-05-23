import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.ResNet import BasicBlock, BottleNeck, ResNet
from utils.dataset import LiverClassificationDataset
from utils.logger import Logger

def train(net, device, dataset_train, dataset_val, batch_size=16, epochs=50, lr=0.00001):
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay=1e-8)
    loss_fn = nn.CrossEntropyLoss()
    
    log_train = Logger('log_train_classification.txt')
    log_val = Logger('log_val_classification.txt')

    best_loss = np.inf
    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        print('running epoch {}'.format(epoch))
        
        net.train()
        for image, label in tqdm(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = loss_fn(pred, label)
            train_loss += loss.item() * image.size(0)

            loss.backward()
            optimizer.step()

        net.eval()
        for image, label in tqdm(val_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            pred = net(image)
            loss = loss_fn(pred, label)
            val_loss += loss.item() * image.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, val_loss))
        log_train.write_line(str(epoch) + ',' + str(round(train_loss, 6)))
        log_val.write_line(str(epoch) + ',' + str(round(val_loss, 6)))

        if val_loss <= best_loss:
            torch.save(net.state_dict(), 'model_classification.pth')
            best_loss = val_loss
            print('model saved')
        if epoch >= 15:
            torch.save(net.state_dict(), 'model_clasification_' + str(epoch) + '.pth')
            print('model saved')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ' + str(device))

    train_dir = 'visualized/train'
    val_dir = 'visualized/val'

    image_transforms = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset_train = LiverClassificationDataset(data_dir=train_dir, image_transforms=image_transforms)
    dataset_val = LiverClassificationDataset(data_dir=val_dir, image_transforms=image_transforms)

    net = ResNet(in_channel=1, n_classes=2, block=BottleNeck, num_block=[3, 4, 6, 3])
    net.to(device=device)

    train(net=net, device=device, dataset_train=dataset_train, dataset_val=dataset_val)

if __name__ == '__main__':
    main()