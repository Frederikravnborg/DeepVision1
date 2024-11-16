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
        
        # Convert RoI coordinates to match feature map dimensions
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
        for images, bboxes, labels, filenames in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

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

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")



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


def visualize_samples(model, dataset, ground_truths, num_samples=5):
    """
    Visualizes a few samples from the dataset with ground truth and model predictions.

    Args:
        model: Trained PyTorch Fast R-CNN model.
        dataset: The dataset to sample images from.
        ground_truths: Dictionary of ground truth boxes.
        num_samples: Number of samples to visualize.
    """
    model.eval()
    samples = random.sample(range(len(dataset)), num_samples)

    for idx in samples:
        image, bbox, label, image_filename = dataset[idx]

        # Load original image
        image_path = os.path.join(dataset.image_dir, image_filename)
        original_image = Image.open(image_path).convert('RGB')

        # Prepare image for model input
        image = image.unsqueeze(0).to(device)  # Add batch dimension

        # Forward pass through the model to get predictions
        with torch.no_grad():
            feature_maps = model.backbone(image)
            
            # Convert bbox tensor to RoI format
            roi = torch.tensor([[0, bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float32).to(device)
            pooled_features = model.roi_pool(feature_maps, roi, output_size=(7, 7))
            pooled_features = pooled_features.view(pooled_features.size(0), -1)
            
            class_logits, bbox_deltas = model.classifier(pooled_features), model.regressor(pooled_features)

            # Get predicted class
            _, predicted_class = torch.max(class_logits, 1)
            predicted_class = predicted_class.item()

            # Apply bbox regression to refine the predicted box
            bbox_deltas = bbox_deltas.squeeze().cpu().numpy()
            refined_bbox = refine_bbox(bbox.cpu().numpy(), bbox_deltas)

        # Convert image for plotting
        original_image_np = np.array(original_image)

        # Plot the image
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(original_image_np)
        ax.axis('off')
        
        # Draw predicted bounding box
        pred_rect = patches.Rectangle((refined_bbox[0], refined_bbox[1]),
                                      refined_bbox[2] - refined_bbox[0],
                                      refined_bbox[3] - refined_bbox[1],
                                      linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(pred_rect)
        ax.text(refined_bbox[0], refined_bbox[1] - 5, f'Pred: {"Object" if predicted_class == 1 else "Background"}',
                color='red', fontsize=12, weight='bold')
        
        # Draw ground truth boxes (if available)
        gt_boxes = ground_truths.get(image_filename, [])
        for gt in gt_boxes:
            gt_rect = patches.Rectangle((gt['xmin'], gt['ymin']),
                                        gt['xmax'] - gt['xmin'],
                                        gt['ymax'] - gt['ymin'],
                                        linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(gt_rect)
            ax.text(gt['xmin'], gt['ymin'] - 5, 'Ground Truth', color='green', fontsize=12, weight='bold')

        plt.show()

def refine_bbox(original_bbox, deltas):
    """
    Applies bounding box deltas to refine the original bounding box.
    """
    x_min, y_min, x_max, y_max = original_bbox
    dx, dy, dw, dh = deltas

    # Compute the center, width, and height of the original bbox
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + 0.5 * width
    center_y = y_min + 0.5 * height

    # Apply deltas
    refined_center_x = center_x + dx * width
    refined_center_y = center_y + dy * height
    refined_width = np.exp(dw) * width
    refined_height = np.exp(dh) * height

    # Convert back to corner coordinates
    refined_x_min = refined_center_x - 0.5 * refined_width
    refined_y_min = refined_center_y - 0.5 * refined_height
    refined_x_max = refined_center_x + 0.5 * refined_width
    refined_y_max = refined_center_y + 0.5 * refined_height

    return [refined_x_min, refined_y_min, refined_x_max, refined_y_max]



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
    train_model(model, train_loader, optimizer, num_epochs=NUM_EPOCHS)

    # Evaluate the model
    print("\nEvaluating Model on Validation Set...\n")
    evaluate_model(model, val_loader)

    # Visualize some samples
    print("\nVisualizing Sample Predictions...\n")
    visualize_samples(model, val_dataset, used_dataset, full_dataset, ground_truths, num_samples=5)

    # Save the trained model
    MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'fast_rcnn_model.pth')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'\nModel saved to {MODEL_SAVE_PATH}')


if __name__ == '__main__':
    main()
