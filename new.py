import os
import json
import pickle
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import FasterRCNN
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# ===============================
# Configuration and Parameters
# ===============================

# Paths
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
TRAINING_DATA_FILE = os.path.join(DATASET_DIR, 'training_data.pkl')  # Combined proposals and ground truths

# Training Parameters
NUM_CLASSES = 2  # 1 object class + 1 background
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

USE_FULL_DATA = True  # Set to True to use the full dataset, False to use 5%

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ===============================
# Task 1: Build the Fast-RCNN Model
# ===============================

def build_fast_rcnn_model(num_classes):
    # Use a pretrained ResNet backbone, without RPN
    backbone = resnet_fpn_backbone('resnet18', pretrained=True)
    
    # We will need a RoIAlign layer for Fast-RCNN, which requires the proposals
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=None,   # No RPN (for Fast-RCNN)
        box_roi_pool=None            # RoI Align pooling layer
    )
    
    return model

# ===============================
# Task 2: Create the DataLoader
# ===============================

class ProposalDataset(Dataset):
    """
    Custom Dataset for Object Proposals.
    """
    def __init__(self, annotations, image_dir, transform=None):
        self.annotations = annotations
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_filename = annotation['image_filename']
        bbox = annotation['bbox']  # Proposals
        label = annotation['label']  # Class label

        if isinstance(bbox, dict):
            bbox = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': torch.tensor([bbox], dtype=torch.float32),
            'labels': torch.tensor([label], dtype=torch.int64)
        }

        return image, target




def load_data(training_data_file):
    """
    Loads the training data from a pickle file.
    """
    with open(training_data_file, 'rb') as f:
        combined_data = pickle.load(f)
    
    # Ensure we're accessing the correct keys
    proposals = combined_data.get('proposals', [])
    ground_truths = combined_data.get('ground_truths', [])
    
    return proposals, ground_truths


def inference_fast_rcnn(model, image, proposals):
    model.eval()
    with torch.no_grad():
        # Pass image and proposals to the model
        target = {'boxes': proposals, 'labels': torch.ones(len(proposals), dtype=torch.int64)}  # Dummy labels
        prediction = model([image.to(device)], [target])
    
    return prediction


# ===============================
# Task 3: Train the Model
# ===============================

def train_model(model, criterion, optimizer, dataloader, dataset_size, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}

            # Debugging: Check targets structure
            print(f"Targets type: {type(targets)}, keys: {list(targets.keys())}")

            optimizer.zero_grad()

            loss_dict = model(inputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        epoch_loss = running_loss / dataset_size
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model


# ===============================
# Task 4: Evaluate the Model
# ===============================

def evaluate_model(model, dataloader):
    """
    Evaluates the model's detection accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}

            # Forward pass
            output = model(inputs, targets)

            # Here you can apply IoU or other metrics to evaluate the detection performance
            # For simplicity, we assume we just evaluate accuracy (in a real setting, you will need mAP)
            predicted_labels = output[0]['labels']
            total += len(predicted_labels)
            correct += (predicted_labels == targets['labels']).sum().item()

    accuracy = correct / total if total > 0 else 0
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')
    return accuracy


# ===============================
# Main Execution
# ===============================

def main():
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    proposals, ground_truths = load_data(TRAINING_DATA_FILE)
    print(f'Total Annotations Loaded: {len(proposals)}')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((800, 800)),  # Resize to match Fast-RCNN input size
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = ProposalDataset(proposals, ANNOTATED_IMAGES_DIR, transform=transform)

    # Split into training and validation sets
    dataset_size = len(dataset)
    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f'Training Samples: {len(train_dataset)}, Validation Samples: {len(val_dataset)}')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Build the model
    model = build_fast_rcnn_model(NUM_CLASSES)
    model = model.to(device)

    # Define Loss and Optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Train the model
    print("\nStarting Training...\n")
    model = train_model(model, None, optimizer, train_loader, train_size, num_epochs=NUM_EPOCHS)

    # Evaluate the model
    print("\nEvaluating Model on Validation Set...\n")
    accuracy = evaluate_model(model, val_loader)

    # Save the trained model
    MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'fast_rcnn_resnet18.pth')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'\nModel saved to {MODEL_SAVE_PATH}')

if __name__ == "__main__":
    main()
