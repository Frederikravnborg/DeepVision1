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
        label = proposal['label']

        # Load the image
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Debugging: Print the proposal and check the bbox structure
        print(f"Proposal {idx}: {proposal}")
        print(f"Bounding Box: {proposal['bbox']}")  # This should print the bbox correctly

        # Return image, label, and proposal
        return image, label, proposal




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
    
    def forward(self, images, proposals):
        """
        Forward pass of the Fast R-CNN model.
        
        Args:
            images: The input images (batch).
            proposals: The proposals for object detection, where each proposal contains a bounding box.
            
        Returns:
            class_logits: The classification logits for each proposal.
        """
        # Get feature maps from the backbone
        feature_maps = self.backbone(images)
        print(f"Feature Map Shape: {feature_maps.shape}")
        
        # Now, proposals already contain bounding boxes in the correct format
        # proposals is a list of bounding boxes for each image in the batch
        # We need to convert them to a tensor of the shape (batch_size, num_proposals, 4)
        
        # Assuming proposals is a list of bounding boxes for each image in the batch
        roi_boxes = []
        for i, proposal_list in enumerate(proposals):
            for proposal in proposal_list:
                # Ensure that 'proposal' is a tuple or list of the form (xmin, ymin, xmax, ymax)
                if isinstance(proposal, (tuple, list)) and len(proposal) == 4:
                    xmin, ymin, xmax, ymax = proposal
                    roi_boxes.append([i, xmin, ymin, xmax, ymax])  # Add batch index and the proposal bounding box
                else:
                    print(f"Skipping invalid proposal: {proposal}")
        
        # Convert to a tensor of shape [num_proposals, 5]
        roi_boxes = torch.tensor(roi_boxes, dtype=torch.float32).to(device)
        
        # Ensure the shape is correct for ROI pooling: [K, 5] where K is the number of proposals
        print(f"RoI Boxes Shape: {roi_boxes.shape}")
        assert roi_boxes.ndimension() == 2 and roi_boxes.size(1) == 5, f"Expected roi_boxes of shape [K, 5], got {roi_boxes.shape}"
        
        # Apply RoI Pooling
        pooled_features = self.roi_pool(feature_maps, roi_boxes)
        print(f"Pooled Feature Shape: {pooled_features.shape}")
        
        # Flatten pooled features for the classifier
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Classification
        class_logits = self.classifier(pooled_features)
        print(f"Class Logits Shape: {class_logits.shape}")

        return class_logits






# ===============================
# Loss Function
# ===============================

def compute_loss(class_logits, labels):
    print("Logits Shape:", class_logits.shape)
    print("Labels Shape:", labels.shape)
    print("Unique Labels:", labels.unique())
    classification_loss = nn.CrossEntropyLoss()(class_logits, labels)
    return classification_loss


# ===============================
# Train Function
# ===============================

# ===============================
# Train Function
# ===============================

def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs=10):
    """
    Trains the Fast R-CNN model.
    
    Args:
        model: The model to be trained.
        train_dataloader: DataLoader for the training dataset.
        val_dataloader: DataLoader for the validation dataset.
        optimizer: The optimizer for training.
        num_epochs: Number of epochs to train.

    Returns:
        model: The trained model.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels, proposals in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):

            # Extract bounding boxes from the proposals
            # Note: proposals is a dictionary, and we need to access the bbox and label from it
            image_filenames = proposals['image_filename']  # List of image filenames
            bbox_xmin = proposals['bbox']['xmin'].tolist()  # Convert tensor to list
            bbox_ymin = proposals['bbox']['ymin'].tolist()  # Convert tensor to list
            bbox_xmax = proposals['bbox']['xmax'].tolist()  # Convert tensor to list
            bbox_ymax = proposals['bbox']['ymax'].tolist()  # Convert tensor to list

            # Debugging: Check the first proposal and bounding box values
            print(f"First Proposal: {proposals['image_filename'][0]}")
            print(f"Bounding Box xmin: {bbox_xmin[0]}, ymin: {bbox_ymin[0]}, xmax: {bbox_xmax[0]}, ymax: {bbox_ymax[0]}")

            # Convert bbox values into the format that Fast R-CNN expects
            proposal_bboxes = list(zip(bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))  # List of bounding boxes

            # Prepare the inputs for the model
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            class_logits = model(images, proposal_bboxes)  # Pass proposals (bounding boxes) to the model

            # Compute the loss
            loss = compute_loss(class_logits, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the current epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate model on validation set after each epoch
        val_accuracy = evaluate_model(model, val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}")

    # Return the trained model
    return model






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
