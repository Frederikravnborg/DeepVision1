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
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import cv2
import numpy as np
from torchvision.ops import MultiScaleRoIAlign

# ===============================
# Configuration and Parameters
# ===============================

# Paths
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
TRAINING_DATA_FILE = os.path.join(DATASET_DIR, 'splits.json')  # Splits for train/test
PROPOSALS_FILE = os.path.join(DATASET_DIR, 'selective_search_proposals_fast.json')  # Selective Search proposals

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
# Task 1: Build Fast R-CNN Model
# ===============================

class FastRCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNNModel, self).__init__()
        # Load a pre-trained ResNet model as the backbone
        self.backbone = models.resnet50(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layer
        
        # Specify the feature maps to use
        self.featmap_names = ['0']  # Use feature map from the last layer before AvgPool
        
        # Create an ROI Pooling layer
        self.roi_pooling = MultiScaleRoIAlign(
            featmap_names=self.featmap_names,
            output_size=(7, 7),
            sampling_ratio=2
        )
        
        # Define the classification and regression heads
        self.classifier = nn.Linear(2048 * 7 * 7, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)


    def forward(self, images, proposals):
        # images: batch of images
        # proposals: list of region proposals (bounding boxes)

        # Pass the image through the backbone to extract feature maps
        feature_map = self.backbone(images)  # Shape: [batch_size, 512, H, W]

        # RoI pooling: proposals must be in format [batch_idx, xmin, ymin, xmax, ymax]
        pooled_features = self.roi_pooling(feature_map, proposals)  # Shape: [batch_size, num_proposals, 512, 7, 7]

        # Flatten the pooled features
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # Classifier and regressor
        cls_scores = self.cls_fc(pooled_features)
        bbox_preds = self.bbox_fc(pooled_features)

        return cls_scores, bbox_preds


# ===============================
# Task 2: Create the DataLoader
# ===============================

class FastRCNNDataset(Dataset):
    """
    Custom Dataset for Fast R-CNN with selective search proposals.
    """
    def __init__(self, image_dir, proposals_file, splits_file, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            proposals_file (str): File containing selective search proposals.
            splits_file (str): File containing train/test splits.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load proposals and splits
        with open(proposals_file, 'r') as f:
            self.proposals = json.load(f)
        with open(splits_file, 'r') as f:
            self.splits = json.load(f)

        # Select the dataset split
        self.train_files = self.splits['train']
        self.test_files = self.splits['test']

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, idx):
        xml_filename = self.train_files[idx]
        image_filename = xml_filename.replace('.xml', '.jpg')
        image_path = os.path.join(self.image_dir, image_filename)

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Load region proposals from selective search
        proposals = self.proposals.get(image_filename, [])

        # For ground truth bounding boxes (annotations)
        # This can be fetched from the XML files (not shown here)
        ground_truths = self.parse_annotations(xml_filename)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert region proposals to the proper format for RoI pooling
        rois = []
        for proposal in proposals:
            rois.append([0, proposal['xmin'], proposal['ymin'], proposal['xmax'], proposal['ymax']])  # [batch_idx, xmin, ymin, xmax, ymax]
        
        rois = torch.tensor(rois, dtype=torch.float32)

        return image, rois, ground_truths

    import xml.etree.ElementTree as ET

    def parse_annotations(self, xml_filename):
        """
        Parse the XML annotation file and extract the bounding boxes and class labels.
        """
        tree = ET.parse(xml_filename)
        root = tree.getroot()

        objects = []
        
        # Loop through each object in the XML file
        for obj in root.findall('object'):
            class_name = obj.find('name').text  # Class name (e.g., 'pothole')
            
            # Extract bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Map class names to integer labels (assuming 'pothole' is class 1, others are background)
            class_label = 1 if class_name == 'pothole' else 0  # Background is 0

            # Append the class label and bounding box as a tuple
            objects.append((class_label, xmin, ymin, xmax, ymax))

        return objects

# ===============================
# Task 3: Training and Evaluation
# ===============================

def train_model(model, criterion_cls, criterion_bbox, optimizer, dataloader, dataset_size, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss_cls = 0.0
        running_loss_bbox = 0.0

        for inputs, proposals, ground_truths in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            proposals = proposals.to(device)
            ground_truths = torch.tensor(ground_truths).to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            cls_scores, bbox_preds = model(inputs, proposals)

            # Calculate classification loss
            cls_loss = criterion_cls(cls_scores, ground_truths[:, 0])

            # Calculate bounding box regression loss
            bbox_loss = criterion_bbox(bbox_preds, ground_truths[:, 1:])

            # Total loss
            loss = cls_loss + bbox_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss_cls += cls_loss.item() * inputs.size(0)
            running_loss_bbox += bbox_loss.item() * inputs.size(0)

        epoch_loss_cls = running_loss_cls / dataset_size
        epoch_loss_bbox = running_loss_bbox / dataset_size
        print(f'Epoch {epoch+1}/{num_epochs}, Loss Classifier: {epoch_loss_cls:.4f}, Loss BBox: {epoch_loss_bbox:.4f}')

    return model

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, proposals, ground_truths in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            proposals = proposals.to(device)

            cls_scores, bbox_preds = model(inputs, proposals)
            _, predicted = torch.max(cls_scores, 1)

            total += ground_truths.size(0)
            correct += (predicted == ground_truths[:, 0]).sum().item()

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

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet-18 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])    # ImageNet stds
    ])

    # Create dataset
    dataset = FastRCNNDataset(ANNOTATED_IMAGES_DIR, PROPOSALS_FILE, TRAINING_DATA_FILE, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Build the model
    model = FastRCNNModel(NUM_CLASSES)
    model = model.to(device)

    # Define Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Train the model
    print("\nStarting Training...\n")
    model = train_model(model, criterion_cls, criterion_bbox, optimizer, dataloader, len(dataset), num_epochs=NUM_EPOCHS)

    # Evaluate the model
    print("\nEvaluating Model...\n")
    accuracy = evaluate_model(model, dataloader)

if __name__ == "__main__":
    main()
