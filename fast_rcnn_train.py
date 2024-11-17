import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torchvision.ops import RoIPool
from PIL import Image
import numpy as np
import random
from tqdm import tqdm

# ===============================
# Configuration and Parameters
# ===============================
# Paths
DATASET_DIR = 'Potholes/'
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
TRAINING_DATA_FILE = os.path.join(DATASET_DIR, 'training_data.pkl')

# Training Parameters
NUM_CLASSES = 2  # 1 object class + 1 background
BATCH_SIZE = 4
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 43

USE_FULL_DATA = True  # Set to True to use the full dataset, False to use 5%

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ===============================
# Custom Dataset for Fast R-CNN
# ===============================

class FastRCNNDataset(Dataset):
    """
    Custom Dataset for Object Proposals.
    """
    def __init__(self, proposals, image_dir, transform=None):
        """
        Args:
            proposals (list): List of dictionaries, each containing 'bbox' (proposal bounding box) and 'label' for each proposal.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.proposals = proposals
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.proposals)

    def __getitem__(self, idx):
        # Get the proposal for this sample
        proposal = self.proposals[idx]
        
        # Get the bounding box and label
        bbox = proposal['bbox']  # Proposal bounding box (should be a dictionary)
        label = proposal['label']  # Proposal label
        
        # Convert bbox dict to a tensor
        bbox_tensor = torch.tensor([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']], dtype=torch.float32)

        image_filename = proposal['image_filename']
        
        # Load the image
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        # Apply transformations (resize, normalization, etc.) to the image
        if self.transform:
            image = self.transform(image)

        # Return the image, proposal bbox tensor, and proposal label
        return image, bbox_tensor, label


# ===============================
# Fast R-CNN Model
# ===============================

class FastRCNN(nn.Module):
    def __init__(self, num_classes=2, roi_output_size=(7, 7)):
        super(FastRCNN, self).__init__()
        # Load ResNet-18 as the backbone
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Excluding the last two layers
        
        # RoI Pooling
        self.roi_pool = RoIPool(output_size=roi_output_size, spatial_scale=1/32)  # Adjust scale based on feature map reduction
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 * roi_output_size[0] * roi_output_size[1], 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, images, rois):
        """
        Forward pass of the Fast R-CNN model.
        
        Args:
            images: The input images (entire images, not cropped).
            rois: The bounding boxes (Regions of Interest) for each image.
        
        Returns:
            class_logits: The classification logits for each region.
            bbox_deltas: The predicted bounding box refinements.
        """
        # Step 1: Feature extraction with backbone (e.g., ResNet)
        feature_maps = self.backbone(images)

        # Step 2: RoI Pooling to extract features corresponding to the bounding boxes
        pooled_features = self.roi_pool(feature_maps, rois)

        # Step 3: Flatten pooled features and classify
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        class_logits, bbox_deltas = self.classifier(pooled_features)

        return class_logits, bbox_deltas



# ===============================
# Loss Function
# ===============================

def compute_loss(class_logits, bbox_deltas, class_labels, proposal_bboxes):
    """
    Compute the classification and bounding box regression loss.
    
    Args:
        class_logits: Logits from the model for classification.
        bbox_deltas: Predicted bounding box deltas (refinements).
        class_labels: Ground truth class labels (for classification).
        proposal_bboxes: Proposal bounding boxes (for regression).
        
    Returns:
        total_loss: Combined classification and bounding box regression loss.
    """
    # Classification loss
    classification_loss = nn.CrossEntropyLoss()(class_logits, class_labels)

    # Bounding box regression loss (Smooth L1 Loss)
    bbox_regression_loss = nn.SmoothL1Loss()(bbox_deltas, proposal_bboxes)

    # Total loss (sum of classification and regression loss)
    total_loss = classification_loss + bbox_regression_loss
    return total_loss


# ===============================
# Train Function
# ===============================

def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, bboxes, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):

            images = images.to(device)
            bboxes = bboxes.to(device)  # Proposal bounding boxes (now tensor)
            labels = labels.to(device)  # Proposal labels

            optimizer.zero_grad()

            # Forward pass: images and bounding boxes (RoIs)
            class_logits, bbox_deltas = model(images, bboxes)

            # Compute loss (classification loss + bounding box regression loss)
            loss = compute_loss(class_logits, bbox_deltas, labels, bboxes)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate model on validation set
        val_accuracy = evaluate_model(model, val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}")






# ===============================
# Evaluate Function
# ===============================

def evaluate_model(model, dataloader):
    model.eval()
    correct_classifications = 0
    total_classifications = 0

    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            class_logits = model(images, labels)  # Pass labels as proposals for eval
            _, predicted_classes = torch.max(class_logits, 1)

            total_classifications += labels.size(0)
            correct_classifications += (predicted_classes == labels).sum().item()

    classification_accuracy = correct_classifications / total_classifications if total_classifications > 0 else 0
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
    print(f'Total Proposals Loaded: {len(proposals)}')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet-18 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])    # ImageNet stds
    ])

    # Create the full dataset using FastRCNNDataset
    full_dataset = FastRCNNDataset(proposals, ANNOTATED_IMAGES_DIR, transform=transform)

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

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = FastRCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    model = train_model(model, train_loader, val_loader, optimizer, num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    main()
