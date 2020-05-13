import time

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models


class Reshaper(nn.Module):
    def __init__(self, target_shape):
        super(Reshaper, self).__init__()
        self.target_shape = target_shape
    
    def forward(self, input):
        return torch.reshape(input, (-1, *self.target_shape))


class EyesNet(nn.Module):
    def __init__(self):
        super(EyesNet, self).__init__()
        
        self.features_left = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            Reshaper([64])
        )
        self.features_right = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            Reshaper([64])
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x_left, x_right):
        x_left = self.features_left(x_left)
        x_right = self.features_right(x_right)
        x = torch.cat((x_left, x_right), 1)
        x = self.fc(x)
        
        return x


class VGG16Based(nn.Module):
    def __init__(self):
        super(VGG16Based, self).__init__()
        
        self.vgg = models.vgg16(pretrained=False)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x_left, x_right):
        x_mid = (x_left + x_right) / 2
        x = torch.cat((x_left, x_mid, x_right), dim=1)

        x_pad = torch.zeros((x.shape[0], 3, 32, 32))
        x_pad[:, :, :16, :] = x

        x = self.vgg(x_pad)
        
        return x


def make_2eyes_datasets(size=256*5, train_size=0.8):
    n, height, width = size, 16, 32
    
    images_left = np.zeros((n, 1, height, width))
    images_right = np.zeros((n, 1, height, width))

    pupils = np.zeros((n, 2))
    
    images_left_train, images_left_val, images_right_train, images_right_val, pupils_train, pupils_val = train_test_split(
        images_left, images_right, pupils, train_size=train_size
    )
    
    def make_dataset(im_left, im_right, pups):
        return TensorDataset(
            torch.from_numpy(im_left.astype(np.float32)), torch.from_numpy(im_right.astype(np.float32)), torch.from_numpy(pups.astype(np.float32))
        )
    
    train_dataset = make_dataset(images_left_train, images_right_train, pupils_train)
    val_dataset = make_dataset(images_left_val, images_right_val, pupils_val)
    
    return train_dataset, val_dataset


def make_dataloaders(train_dataset, val_dataset, batch_size=256):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader


def measure_time(model, data_loader, n_batches=5):
    begin_time = time.time()

    batch_num = 0
    n_samples = 0

    predicted = []
    for *xs, y in data_loader:
        xs = [x.cpu() for x in xs]

        y_pred = model(*xs)
        predicted.append(y_pred.detach().cpu().numpy().reshape(-1))

        batch_num += 1
        n_samples += len(y)

        if batch_num >= n_batches:
          break
    
    end_time = time.time()

    time_per_sample = (end_time - begin_time) / n_samples
    return time_per_sample


eyesnet_cpu = EyesNet().cpu()
eyesnet_cpu.load_state_dict(torch.load("epoch_299.pth", map_location="cpu"))

eyes_datasets = make_2eyes_datasets(256*5)
_, eyes_val_loader = make_dataloaders(*eyes_datasets, batch_size=1)

tps = measure_time(eyesnet_cpu, eyes_val_loader)
print(f"{tps} seconds per sample, EyesNet")

vgg16 = VGG16Based()

vgg16_tps = measure_time(vgg16, eyes_val_loader)
print(f"{vgg16_tps} seconds per sample, VGG16-based")


