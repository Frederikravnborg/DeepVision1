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
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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

def build_model(num_classes):
    """
    Builds a pre-trained Faster-RCNN model with custom number of classes.
    """
    model = FasterRCNN(backbone=models.resnet18(weights='IMAGENET1K_V1').features, 
                       num_classes=num_classes)
    return model

# ===============================
# Task 2: Create the DataLoader
# ===============================

class ProposalDataset(Dataset):
    """
    Custom Dataset for Object Proposals.
    """
    def __init__(self, annotations, image_dir, transform=None):
        """
        Args:
            annotations (list): List of annotation dictionaries.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = annotations
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_filename = annotation['image_filename']
        bbox = annotation['bbox']
        label = annotation['label']

        # Load image
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if available
        if self.transform:
            image = self.transform(image)

        # Create the target dictionary in the format that FasterRCNN expects
        target = {}
        target['boxes'] = torch.tensor([bbox], dtype=torch.float32)
        target['labels'] = torch.tensor([label], dtype=torch.int64)

        return image, target

def load_data(training_data_file):
    """
    Loads the training data from a pickle file.
    """
    with open(training_data_file, 'rb') as f:
        data = pickle.load(f)
    annotations = data['annotations']  # Assuming 'annotations' contains the bounding box info
    return annotations

# ===============================
# Task 3: Train the Model
# ===============================

def train_model(model, criterion, optimizer, dataloader, dataset_size, num_epochs=10):
    """
    Trains the model.
    """
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(inputs, targets)

            # Get total loss from all parts
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimize
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
# Visualization Function
# ===============================

def visualize_samples(model, dataset, num_samples=5):
    """
    Visualizes a few samples from the dataset with ground truth and predictions.

    Args:
        model: Trained PyTorch model.
        dataset: Dataset for visualization.
        num_samples: Number of samples to visualize.
    """
    model.eval()
    samples = random.sample(range(len(dataset)), num_samples)

    for idx in samples:
        image, target = dataset[idx]
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)

        # Forward pass
        with torch.no_grad():
            prediction = model([image.to(device)])

        # Get the predicted boxes and labels
        predicted_boxes = prediction[0]['boxes'].cpu().numpy()
        predicted_labels = prediction[0]['labels'].cpu().numpy()

        # Plot the image
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(image_np)
        ax.axis('off')

        # Draw predicted boxes
        for box, label in zip(predicted_boxes, predicted_labels):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], f'Class: {label}', color='red')

        plt.show()

# ===============================
# Main Execution
# ===============================

def main():
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load data
    annotations = load_data(TRAINING_DATA_FILE)
    print(f'Total Annotations Loaded: {len(annotations)}')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((800, 800)),  # Resize to match Fast-RCNN input size
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = ProposalDataset(annotations, ANNOTATED_IMAGES_DIR, transform=transform)

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
    model = build_model(NUM_CLASSES)
    model = model.to(device)

    # Define Loss and Optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Train the model
    print("\nStarting Training...\n")
    model = train_model(model, None, optimizer, train_loader, train_size, num_epochs=NUM_EPOCHS)

    # Evaluate the model
    print("\nEvaluating Model on Validation Set...\n")
    accuracy = evaluate_model(model, val_loader)

    # Visualize some samples
    #print("\nVisualizing Sample Predictions...\n")
    #visualize_samples(model, val_dataset, num_samples=5)

    # Save the trained model
    MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'faster_rcnn_resnet18.pth')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'\nModel saved to {MODEL_SAVE_PATH}')

if __name__ == "__main__":
    main()
