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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models import ResNet18_Weights
from torchvision import models

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
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 43

USE_FULL_DATA = True  # Set to True to use the full dataset, False to use 5%

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
        # Load ResNet-18 as the backbone, excluding the last two layers
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Excluding the last two layers

        # RoI Pooling layer
        self.roi_pool = roi_pool

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),  # Size of features from ResNet-18
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)  # Output classes for object detection (including background)
        )
        
        # Bounding box regression head
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4)  # Predicting the bounding box deltas
        )

    def forward(self, images, rois):
        feature_maps = self.backbone(images)
        
        # RoI Pooling
        feature_map_height, feature_map_width = feature_maps.shape[2:]
        rois[:, 1:] = rois[:, 1:] * torch.tensor(
            [feature_map_width, feature_map_height, feature_map_width, feature_map_height],
            device=rois.device
        )

        pooled_features = self.roi_pool(feature_maps, rois, output_size=(7, 7))
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        class_logits = self.classifier(pooled_features)
        bbox_deltas = self.regressor(pooled_features)

        return class_logits, bbox_deltas



# ===============================
# Task 3: Loss Function
# ===============================

def compute_loss(class_logits, bbox_deltas, labels, gt_bboxes, class_weights=None):
    # Classification loss with class weights (if provided)
    classification_loss = nn.CrossEntropyLoss(weight=class_weights)(class_logits, labels)
    
    # Bounding box regression loss
    bbox_regression_loss = nn.SmoothL1Loss()(bbox_deltas, gt_bboxes)
    
    return classification_loss + bbox_regression_loss

# Example: Calculate class weights based on class frequencies
def compute_class_weights(labels):
    # Get the frequency of each class (0 for background, 1 for object)
    class_counts = torch.bincount(labels)
    
    # Total number of samples
    total_samples = labels.size(0)
    
    # Inverse frequency as weights: higher weight for less frequent classes
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    
    return class_weights


# ===============================
# Task 4: Train Function
# ===============================

def train_model(model, dataloader, optimizer, num_epochs=10):
    """
    Trains the Fast R-CNN model.
    
    Args:
        model: The model to be trained.
        dataloader: DataLoader for the training dataset.
        optimizer: The optimizer for training.
        num_epochs: Number of epochs to train.

    Returns:
        model: The trained model.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, bboxes, labels, filenames in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            # Compute class weights for the current batch
            class_weights = compute_class_weights(labels)

            # Convert bounding boxes to RoI format
            rois = torch.stack([
                torch.cat([torch.tensor([i], dtype=torch.float32, device=device), bbox]) 
                for i, bbox in enumerate(bboxes)
            ])

            optimizer.zero_grad()

            # Forward pass
            class_logits, bbox_deltas = model(images, rois)

            # Compute the loss
            loss = compute_loss(class_logits, bbox_deltas, labels, bboxes, class_weights)


            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the current epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Return the trained model
    return model




def evaluate_model(model, dataloader):
    model.eval()
    correct_classifications = 0
    total_classifications = 0

    with torch.no_grad():
        for images, bboxes, labels, filenames in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            rois = torch.stack([
                torch.cat([torch.tensor([i], dtype=torch.float32, device=device), bbox]) 
                for i, bbox in enumerate(bboxes)
            ])
            
            class_logits, _ = model(images, rois)
            _, predicted_classes = torch.max(class_logits, 1)

            total_classifications += labels.size(0)
            correct_classifications += (predicted_classes == labels).sum().item()

    classification_accuracy = correct_classifications / total_classifications if total_classifications > 0 else 0
    print(f'Validation Classification Accuracy: {classification_accuracy * 100:.2f}%')
    return classification_accuracy


# ===============================
# Main Execution
# ===============================

def main():
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    with open(TRAINING_DATA_FILE, 'rb') as f:
        combined_data = pickle.load(f)
    proposals = combined_data['proposals']
    ground_truths = combined_data['ground_truths']
    print(f'Total Proposals Loaded: {len(proposals)}')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet-18 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])    # ImageNet stds
    ])

    # Create the full dataset using FastRCNNDataset
    full_dataset = FastRCNNDataset(proposals, ground_truths, ANNOTATED_IMAGES_DIR, transform=transform)

    # Use full dataset or reduced dataset based on a flag
    if USE_FULL_DATA:
        used_dataset = full_dataset
        print("Using the full dataset.")
    else:
        # Reduce the dataset size to 5% for quick testing
        total_samples = len(full_dataset)
        reduced_sample_size = max(1, int(total_samples * 0.05))  # Ensure at least one sample
        sampled_indices = random.sample(range(total_samples), reduced_sample_size)
        used_dataset = torch.utils.data.Subset(full_dataset, sampled_indices)
        print(f'Reduced Dataset: Using 5% of data ({reduced_sample_size} samples).')

    # Split into training and validation sets
    dataset_size = len(used_dataset)
    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(used_dataset, [train_size, val_size])

    print(f'Training Samples: {len(train_dataset)}, Validation Samples: {len(val_dataset)}')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Build the Fast R-CNN model
    model = FastRCNN(NUM_CLASSES).to(device)

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("\nStarting Training...\n")
    model = train_model(model, train_loader, optimizer, num_epochs=NUM_EPOCHS)

    # Evaluate the model
    print("\nEvaluating Model on Validation Set...\n")
    evaluate_model(model, val_loader)

    # Save the trained model
    MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'fast_rcnn_model.pth')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'\nModel saved to {MODEL_SAVE_PATH}')


if __name__ == '__main__':
    main()
