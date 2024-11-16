import os
import json
import pickle
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.ops import roi_pool
from PIL import Image
import numpy as np
import cv2

# ===============================
# Configuration and Parameters
# ===============================

# Paths
DATASET_DIR = 'Potholes/'
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
TRAINING_DATA_FILE = os.path.join(DATASET_DIR, 'training_data_with_gt.pkl')

# Training Parameters
NUM_CLASSES = 2  # 1 object class + 1 background
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ===============================
# Task 1: Custom Dataset
# ===============================

class FastRCNNDataset(Dataset):
    def __init__(self, proposals, ground_truths, image_dir, transform=None):
        self.proposals = proposals
        self.ground_truths = ground_truths
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.proposals)

    def __getitem__(self, idx):
        proposal = self.proposals[idx]
        image_filename = proposal['image_filename']
        bbox = proposal['bbox']
        label = proposal['label']

        # Load the image
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert bounding box to tensor
        bbox_tensor = torch.tensor([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']], dtype=torch.float32)

        return image, bbox_tensor, label, image_filename


# ===============================
# Task 2: Fast R-CNN Model
# ===============================

class FastRCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FastRCNN, self).__init__()
        # Load a pre-trained ResNet18 as the backbone
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Use all layers except the last two

        # RoI Pooling layer
        self.roi_pool = roi_pool

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        # Bounding box regression head
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4)  # Predicts [dx, dy, dw, dh] adjustments
        )

    def forward(self, images, rois):
        feature_maps = self.backbone(images)
        pooled_features = self.roi_pool(feature_maps, rois, output_size=(7, 7))

        # Flatten pooled features
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # Forward through classifier and regressor
        class_logits = self.classifier(pooled_features)
        bbox_deltas = self.regressor(pooled_features)

        return class_logits, bbox_deltas

# ===============================
# Task 3: Loss Function
# ===============================

def compute_loss(class_logits, bbox_deltas, labels, gt_bboxes):
    classification_loss = nn.CrossEntropyLoss()(class_logits, labels)
    bbox_regression_loss = nn.SmoothL1Loss()(bbox_deltas, gt_bboxes)
    return classification_loss + bbox_regression_loss


# ===============================
# Task 4: Train Function
# ===============================

def train_model(model, dataloader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, bboxes, labels, filenames in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            # Convert to RoI format and ensure all tensors have the correct shape
            rois = torch.stack([
                torch.cat([torch.tensor([i], dtype=torch.float32, device=device), bbox]) 
                for i, bbox in enumerate(bboxes)
            ])


            optimizer.zero_grad()

            class_logits, bbox_deltas = model(images, rois)
            loss = compute_loss(class_logits, bbox_deltas, labels, bboxes)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# ===============================
# Main Execution
# ===============================

def main():
    # Load training data
    with open(TRAINING_DATA_FILE, 'rb') as f:
        combined_data = pickle.load(f)
    proposals = combined_data['proposals']
    ground_truths = combined_data['ground_truths']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = FastRCNNDataset(proposals, ground_truths, ANNOTATED_IMAGES_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = FastRCNN(NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, dataloader, optimizer, NUM_EPOCHS)

if __name__ == '__main__':
    main()
