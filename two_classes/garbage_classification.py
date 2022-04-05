import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class Garbage(Dataset):
    train_list = [
        'data_batch_train'
    ]
    test_list = [
        'data_batch_test'
    ]

    def __init__(self, root, train, transform=None, target_transform=None):
        super(Garbage, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in file_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')
        torch.nn.init.constant_(self.bn.weight, 0.5)
        torch.nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.relu(out)
        return out + x

class Net(nn.Module):
    def __init__(self, n_chans=32, n_blocks=10):
        super(Net, self).__init__()
        self.n_chans = n_chans
        self.conv1 = nn.Conv2d(3, n_chans, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans)])
        )
        self.fc1 = nn.Linear(8 * 8 * n_chans, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def mean_std_data(pth):
    train = Garbage(pth, train=True, transform=transforms.ToTensor())

    imgs = torch.stack([img_t for img_t,_ in train], dim=3)
    print(imgs.shape)

    mean = imgs.view(3, -1).mean(dim=1)
    std = imgs.view(3, -1).std(dim=1)

    return mean, std

def train_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.cpu().item()

        if epoch == 1 or epoch % 10 == 0:
            print('Epoch {}, Training loss {}'.format(
                epoch,
                loss_train / len(train_loader)
            ))

if __name__ == '__main__':
    pth = r'D:\Program_self\basicTorch\inputs\garbage\data_32'
    # O - Organic, label is 0
    # R - Recycle, label is 1
    class_names = ["O", "R"]

    # get mean and std
    mean, std = mean_std_data(pth)
    # create dataset
    train = Garbage(pth, train=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std),
                    ]))

    val = Garbage(pth, train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std),
                    ]))

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Training on device {device}.")

    # model
    model = Net().to(device=device)
    # dataloader
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_loop(
        n_epochs=100,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device
    )
