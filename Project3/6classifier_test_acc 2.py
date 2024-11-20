import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torchvision.models import ResNet18_Weights
import random

# ===============================
# Configuration and Parameters
# ===============================

# Paths
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
TEST_DATA_FILE = os.path.join(DATASET_DIR, 'test_data.pkl')  # Prepared test data
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'proposal_classifier_resnet18.pth')  # Trained model weights

# Parameters
NUM_CLASSES = 2  # 1 object class + 1 background
BATCH_SIZE = 32
RANDOM_SEED = 42

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")
print(f'Using device: {device}')

# ===============================
# Dataset Definition
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

# ===============================
# Data Loading and Preparation
# ===============================

def load_test_data(test_data_file):
    """
    Loads the test data from a pickle file.
    """
    with open(test_data_file, 'rb') as f:
        combined_data = pickle.load(f)
    proposals = combined_data['proposals']
    ground_truths = combined_data['ground_truths']
    return proposals, ground_truths

# Load test data
test_proposals, test_ground_truths = load_test_data(TEST_DATA_FILE)
print(f'Total Test Proposals Loaded: {len(test_proposals)}')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-18 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])   # ImageNet stds
])

# Create test dataset and DataLoader
test_dataset = ProposalDataset(test_proposals, ANNOTATED_IMAGES_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ===============================
# Model Loading
# ===============================

# Initialize the model architecture
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Load the saved model weights
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f'Model loaded from {MODEL_SAVE_PATH}')

# ===============================
# Evaluation Function
# ===============================

def evaluate_model(model, dataloader):
    """
    Evaluates the model's classification accuracy on the test set.
    """
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
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy

# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Evaluate the model
    test_accuracy = evaluate_model(model, test_loader)