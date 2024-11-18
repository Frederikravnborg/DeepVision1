import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from torchvision.ops import box_iou

# ===============================
# Configuration and Parameters
# ===============================

# Paths
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')

# Parameters
NUM_CLASSES = 2  # 1 object class + 1 background
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# Device Configuration
device = torch.device(
                      "cuda" if torch.cuda.is_available() else
                      "cpu")
print(f'Using device: {device}')

# ===============================
# Dataset Definition
# ===============================

def parse_annotation(xml_file):
    """
    Parses a Pascal VOC XML file and extracts bounding box coordinates.
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            
            # Extract coordinates
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # Ensure the bounding box has positive width and height
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                print(f"Invalid box found in {xml_file}: [{xmin}, {ymin}, {xmax}, {ymax}]")
        
        # Return a valid list of boxes and a single label for each box
        return {'boxes': boxes, 'labels': [1] * len(boxes)}
    
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return {'boxes': [], 'labels': []}



class PotholeDataset(Dataset):
    def __init__(self, annotations_dir, transform=None):
        self.annotations_dir = annotations_dir
        self.image_files = [f for f in os.listdir(annotations_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.annotations_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Load annotations
        xml_path = img_path.replace('.jpg', '.xml')
        annotation = parse_annotation(xml_path)
        
        # Convert annotations to Tensor
        boxes = torch.as_tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annotation['labels'], dtype=torch.int64)

        # Create target dictionary
        target = {'boxes': boxes, 'labels': labels}

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, target


# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create Dataset and DataLoaders
dataset = PotholeDataset(ANNOTATED_IMAGES_DIR, transform=transform)
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ===============================
# Model Definition
# ===============================

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(device)

# ===============================
# Training and Validation Loop
# ===============================

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return total_loss / len(dataloader)





def evaluate_on_metrics(model, dataloader, device):
    model.eval()
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    iou_threshold = 0.5

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            
            for idx, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()

                gt_boxes = targets[idx]['boxes'].cpu().numpy()
                gt_labels = targets[idx]['labels'].cpu().numpy()

                iou_matrix = box_iou(torch.tensor(gt_boxes, dtype=torch.float), torch.tensor(pred_boxes, dtype=torch.float))

                print('For Each Target')
                print('=========================================')
                if len(gt_labels):
                    gt_true_positive_counts = len(iou_matrix[iou_matrix > iou_threshold].max())
                    gt_total_positive_counts = len(iou_matrix[iou_matrix > iou_threshold])
                else:
                    gt_true_positive_counts = 0
                    gt_total_positive_counts = 0
                
                gt_total_positive_counts  = sum(gt_total_positive_counts)
                gt_true_positive_counts = sum(gt_true_positive_counts)
                print(gt_total_positive_counts,gt_true_positive_counts)
    return gt_total_positive_counts



# ===============================
# Main Training Loop
# ===============================

import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from torchvision.ops import box_iou

# ===============================
# Configuration and Parameters
# ===============================

# Paths
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')

# Parameters
NUM_CLASSES = 2  # 1 object class + 1 background
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# Device Configuration
device = torch.device(
                      "cuda" if torch.cuda.is_available() else
                      "cpu")
print(f'Using device: {device}')

# ===============================
# Dataset Definition
# ===============================

def parse_annotation(xml_file):
    """
    Parses a Pascal VOC XML file and extracts bounding box coordinates.
    """
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            
            # Extract coordinates
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # Ensure the bounding box has positive width and height
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                print(f"Invalid box found in {xml_file}: [{xmin}, {ymin}, {xmax}, {ymax}]")
        
        # Return a valid list of boxes and a single label for each box
        return {'boxes': boxes, 'labels': [1] * len(boxes)}
    
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return {'boxes': [], 'labels': []}



class PotholeDataset(Dataset):
    def __init__(self, annotations_dir, transform=None):
        self.annotations_dir = annotations_dir
        self.image_files = [f for f in os.listdir(annotations_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.annotations_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Load annotations
        xml_path = img_path.replace('.jpg', '.xml')
        annotation = parse_annotation(xml_path)
        
        # Convert annotations to Tensor
        boxes = torch.as_tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annotation['labels'], dtype=torch.int64)

        # Create target dictionary
        target = {'boxes': boxes, 'labels': labels}

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, target


# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create Dataset and DataLoaders
dataset = PotholeDataset(ANNOTATED_IMAGES_DIR, transform=transform)
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ===============================
# Model Definition
# ===============================

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(device)

# ===============================
# Training and Validation Loop
# ===============================

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        print(losses)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return total_loss / len(dataloader)





def evaluate_on_metrics(model, dataloader, device):
    model.eval()
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    iou_threshold = 0.5

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            
            for idx, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()

                gt_boxes = targets[idx]['boxes'].cpu().numpy()
                gt_labels = targets[idx]['labels'].cpu().numpy()

                iou_matrix = box_iou(torch.tensor(gt_boxes, dtype=torch.float), torch.tensor(pred_boxes, dtype=torch.float))

                print('For Each Target')
                print('=========================================')
                if len(gt_labels):
                    gt_true_positive_counts = len(iou_matrix[iou_matrix > iou_threshold].max())
                    gt_total_positive_counts = len(iou_matrix[iou_matrix > iou_threshold])
                else:
                    gt_true_positive_counts = 0
                    gt_total_positive_counts = 0
                
                gt_total_positive_counts  = sum(gt_total_positive_counts)
                gt_true_positive_counts = sum(gt_true_positive_counts)
                print(gt_total_positive_counts,gt_true_positive_counts)
    return gt_total_positive_counts



# ===============================
# Main Training Loop
# ===============================

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = evaluate_on_metrics(model, val_loader, device=device)
    
    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    lr_scheduler.step()

print("Training complete.")

# ===============================
# Save Model
# ===============================
model_path = os.path.join(DATASET_DIR, "fasterrcnn_pothole_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")
