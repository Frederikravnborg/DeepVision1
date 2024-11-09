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
from torchvision.models import ResNet18_Weights

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
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

USE_FULL_DATA = True  # Set to True to use the full dataset, False to use 5%

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ===============================
# Task 1: Build the CNN
# ===============================

def build_model(num_classes):
    """
    Builds a pre-trained ResNet-18 model and modifies the final layer for classification.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ===============================
# Task 2: Create the DataLoader
# ===============================

class ProposalDataset(Dataset):
    """
    Custom Dataset for Object Proposals.
    """
    def __init__(self, proposals, image_dir, transform=None):
        """
        Args:
            proposals (list): List of proposal dictionaries.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.proposals = proposals
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.proposals)

    def __getitem__(self, idx):
        proposal = self.proposals[idx]
        image_filename = proposal['image_filename']
        bbox = proposal['bbox']
        label = proposal['label']

        # Load image
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        # Crop the proposal region
        cropped_image = image.crop((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))

        # Apply transforms
        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, label

def load_data(training_data_file):
    """
    Loads the training data from a pickle file.
    """
    with open(training_data_file, 'rb') as f:
        combined_data = pickle.load(f)
    proposals = combined_data['proposals']
    ground_truths = combined_data['ground_truths']
    return proposals, ground_truths

# ===============================
# Task 3: Fine-tune the Network
# ===============================

def train_model(model, criterion, optimizer, dataloader, dataset_size, num_epochs=10):
    """
    Trains the model.
    """
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_size
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model

# ===============================
# Task 4: Evaluate the Model
# ===============================

def evaluate_model(model, dataloader):
    """
    Evaluates the model's classification accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')
    return accuracy

# ===============================
# Visualization Function
# ===============================

def visualize_samples(model, subset, used_dataset, full_dataset, ground_truths, num_samples=5):
    """
    Visualizes a few samples from the dataset with ground truth and predictions.

    Args:
        model: Trained PyTorch model.
        subset: Subset of the dataset (e.g., validation set).
        used_dataset: The dataset used for training/validation (either full or reduced).
        full_dataset: The original ProposalDataset.
        ground_truths: Dictionary of ground truth boxes.
        num_samples: Number of samples to visualize.
    """
    model.eval()
    samples = random.sample(range(len(subset)), num_samples)

    for idx in samples:
        # Map subset index to original dataset index
        if isinstance(used_dataset, Subset):
            reduced_idx = used_dataset.indices[idx]
        else:
            reduced_idx = idx  # When using full dataset
        proposal = full_dataset.proposals[reduced_idx]

        image_filename = proposal['image_filename']
        bbox = proposal['bbox']
        label = proposal['label']

        # Load image
        image_path = os.path.join(full_dataset.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        cropped_image = image.crop((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))

        # Apply transforms
        transform = full_dataset.transform
        transformed_image = transform(cropped_image).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(transformed_image)
            _, predicted = torch.max(output, 1)
            predicted = predicted.item()

        # Convert image for plotting
        image_np = transformed_image.squeeze(0).cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)

        # Plot the image
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(image_np)
        ax.axis('off')
        title = f"True Label: {'Object' if label == 1 else 'Background'} | Predicted: {'Object' if predicted == 1 else 'Background'}"
        ax.set_title(title)

        # Draw ground truth boxes (if available)
        gt_boxes = ground_truths.get(image_filename, [])
        for gt in gt_boxes:
            rect = patches.Rectangle((gt['xmin'], gt['ymin']),
                                     gt['xmax'] - gt['xmin'],
                                     gt['ymax'] - gt['ymin'],
                                     linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

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
    proposals, ground_truths = load_data(TRAINING_DATA_FILE)
    print(f'Total Proposals Loaded: {len(proposals)}')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet-18 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])    # ImageNet stds
    ])

    # Create the full dataset
    full_dataset = ProposalDataset(proposals, ANNOTATED_IMAGES_DIR, transform=transform)

    if USE_FULL_DATA:
        used_dataset = full_dataset
        print("Using the full dataset.")
    else:
        # Reduce the dataset size to 5% for quick testing
        total_samples = len(full_dataset)
        reduced_sample_size = max(1, int(total_samples * 0.05))  # Ensure at least one sample
        sampled_indices = random.sample(range(total_samples), reduced_sample_size)
        used_dataset = Subset(full_dataset, sampled_indices)
        print(f'Reduced Dataset: Using 5% of data ({reduced_sample_size} samples).')

    # Split into training and validation sets
    if isinstance(used_dataset, Subset):
        dataset_size = len(used_dataset)
    else:
        dataset_size = len(used_dataset)

    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(used_dataset, [train_size, val_size])

    print(f'Training Samples: {len(train_dataset)}, Validation Samples: {len(val_dataset)}')

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Build the model
    model = build_model(NUM_CLASSES)
    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Train the model
    print("\nStarting Training...\n")
    model = train_model(model, criterion, optimizer, train_loader, train_size, num_epochs=NUM_EPOCHS)

    # Evaluate the model
    print("\nEvaluating Model on Validation Set...\n")
    accuracy = evaluate_model(model, val_loader)

    # Visualize some samples
    print("\nVisualizing Sample Predictions...\n")
    visualize_samples(model, val_dataset, used_dataset, full_dataset, ground_truths, num_samples=5)

    # Save the trained model
    MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'proposal_classifier_resnet18.pth')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'\nModel saved to {MODEL_SAVE_PATH}')

if __name__ == "__main__":
    main()