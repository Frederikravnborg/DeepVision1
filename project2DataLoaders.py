import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time

import matplotlib.pyplot as plt
from IPython.display import clear_output
import random


PH2_path = '/dtu/datasets1/02516/PH2_Dataset_images/'
DRIVE_path = '/dtu/datasets1/02516/DRIVE/training'

class DRIVE(torch.utils.data.Dataset):
    def __init__(self ,transform, data_path=DRIVE_path):
        'Initialization'
        self.transform = transform
        self.image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
        self.label_paths = sorted(glob.glob(data_path + '/1st_manual/*.gif'))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y

class Ph2(torch.utils.data.Dataset):
    def __init__(self, transform, data_path=PH2_path):
        'Initialization'
        self.transform = transform
        self.image_paths = sorted(glob.glob(data_path + '*/*_Dermoscopic_Image/*.bmp'))
        self.label_paths = sorted(glob.glob(data_path + '*/*_lesion/*.bmp'))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y
    

ph2_size = 128
ph2_train_transform = transforms.Compose([transforms.Resize((ph2_size, ph2_size)), 
                                    transforms.ToTensor()])
ph2_test_transform = transforms.Compose([transforms.Resize((ph2_size, ph2_size)), 
                                    transforms.ToTensor()])

ph2_batch_size = 10
ph2set = Ph2(transform=ph2_train_transform)
train_size = int(0.7 * len(ph2set))
val_size = int(0.1 * len(ph2set))
test_size = len(ph2set) - train_size - val_size
train_ph2, val_ph2, test_ph2 = random_split(ph2set, [train_size, val_size,test_size])
ph2_train_loader = DataLoader(train_ph2, batch_size=ph2_batch_size, shuffle=True, num_workers=3)
ph2_val_loader = DataLoader(val_ph2, batch_size=ph2_batch_size, shuffle=False, num_workers=3)
ph2_test_loader = DataLoader(test_ph2, batch_size=ph2_batch_size, shuffle=False, num_workers=3)


drive_size = 512
drive_train_transform = transforms.Compose([transforms.Resize((drive_size, drive_size)), 
                                    transforms.ToTensor()])
drive_test_transform = transforms.Compose([transforms.Resize((drive_size, drive_size)), 
                                    transforms.ToTensor()])

drive_batch_size = 2
DRIVEset = DRIVE(transform=drive_train_transform)
train_size = int(0.8 * len(DRIVEset))
val_size = int( 0.1 * len(DRIVEset))
test_size = len(DRIVEset) - val_size - train_size

train_drive, val_drive, test_drive = random_split(DRIVEset, [train_size, val_size, test_size])
drive_train_loader = DataLoader(train_drive, batch_size=drive_batch_size, shuffle=True, num_workers=3)
drive_val_loader = DataLoader(val_drive, batch_size=drive_batch_size, shuffle=True, num_workers=3)
drive_test_loader = DataLoader(test_drive, batch_size=drive_batch_size, shuffle=False, num_workers=3)