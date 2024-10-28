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
import cv2
from sklearn.cluster import KMeans

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
    
click_count = 3
def generate_clicks(mask, num_pos_clicks=10, num_neg_clicks=10):
    # Get all coordinates for positive clicks (lesion area)

    pos_coords = np.argwhere(mask == 1).T
    # Get all coordinates for negative clicks (background area)
    neg_coords = np.argwhere(mask == 0).T

    # Randomly select points for positive and negative clicks
    pos_clicks = pos_coords[random.sample(range(len(pos_coords)), num_pos_clicks)]
    neg_clicks = neg_coords[random.sample(range(len(neg_coords)), num_neg_clicks)]

    return pos_clicks, neg_clicks

def plot_clicks(image, pos_clicks, neg_clicks):
    plt.imshow(image.squeeze())
    plt.scatter(np.array(pos_clicks)[:,2], np.array(pos_clicks)[:,1], color='green', label='Positive Clicks')
    plt.scatter(np.array(neg_clicks)[:,2], np.array(neg_clicks)[:,1],  color='red', label='Negative Clicks')
    plt.legend()
    plt.show()

def generate_cluster_based_clicks(mask, num_pos_clicks=10, num_neg_clicks=10, num_clusters=5):
    # Convert to NumPy if mask is a PyTorch tensor
    if num_clusters == 0:
        num_clusters = 1
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Get positive and negative coordinates
    pos_coords = np.argwhere(mask == 1)
    neg_coords = np.argwhere(mask == 0)
    
    def select_clustered_points(coords, num_clicks):
        # Perform clustering if there are enough points for clusters
        if len(coords) >= num_clusters:
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(coords)
            clusters = [coords[kmeans.labels_ == i] for i in range(num_clusters)]
            
            # Sample one point from each cluster
            clicks = []
            for cluster in clusters:
                if len(cluster) > 0:
                    point = cluster[random.randint(0, len(cluster) - 1)]
                    clicks.append(point)
                    
            # Add random points if we need more clicks
            while len(clicks) < num_clicks:
                extra_click = coords[random.randint(0, len(coords) - 1)]
                clicks.append(extra_click)
            return np.array(clicks[:num_clicks])
        
        # If not enough points for clustering, sample randomly
        return coords[random.sample(range(len(coords)), num_clicks)]
    
    # Select positive and negative clicks
    pos_clicks = select_clustered_points(pos_coords, num_pos_clicks)
    neg_clicks = select_clustered_points(neg_coords, num_neg_clicks)
    
    # Convert back to tensors if needed
    pos_clicks = torch.tensor(pos_clicks)
    neg_clicks = torch.tensor(neg_clicks)
    
    return pos_clicks, neg_clicks



def generate_edge_based_clicks(mask, num_pos_clicks=10, num_neg_clicks=10):
    # Convert PyTorch tensor to NumPy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Detect edges in the mask
    edges = cv2.Canny((mask * 255).astype(np.uint8), threshold1=50, threshold2=150)

    # Get all coordinates on edges for positive and negative clicks
    pos_coords = np.argwhere((mask == 1) & (edges > 0))
    neg_coords = np.argwhere((mask == 0) & (edges > 0))

    def sample_unique_points(coords, num_clicks):
        # Shuffle and select unique points to avoid concentrated alignment
        random.shuffle(coords)
        return np.array(coords[:num_clicks])

    # If there are not enough edge points, fall back to all mask points
    if len(pos_coords) < num_pos_clicks:
        pos_coords = np.argwhere(mask == 1)
    if len(neg_coords) < num_neg_clicks:
        neg_coords = np.argwhere(mask == 0)

    # Randomly sample points on the edges
    pos_clicks = sample_unique_points(pos_coords.tolist(), num_pos_clicks)
    neg_clicks = sample_unique_points(neg_coords.tolist(), num_neg_clicks)

    # Convert back to tensors
    pos_clicks = torch.tensor(pos_clicks)
    neg_clicks = torch.tensor(neg_clicks)
    
    return pos_clicks, neg_clicks

"""
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

"""