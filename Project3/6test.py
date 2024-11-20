import os
import json
import pickle
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# For NMS and evaluation
from torchvision.ops import nms
from sklearn.metrics import precision_recall_curve, average_precision_score

# For Gradcam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

# ===============================
# Configuration and Parameters
# ===============================

# Paths
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
PROPOSALS_FILE = os.path.join(DATASET_DIR, 'selective_search_proposals_fast.json')
SPLITS_FILE = os.path.join(DATASET_DIR, 'splits.json')
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, 'proposal_classifier_resnet18.pth')

# Parameters
NUM_CLASSES = 2  # 1 object class + 1 background
BATCH_SIZE = 32
IOU_THRESHOLD_NMS = 0.3  # IoU threshold for NMS
IOU_THRESHOLD_EVAL = 0.5  # IoU threshold for evaluation
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold to consider a detection

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ===============================
# Helper Functions
# ===============================

def load_model(model_path, num_classes):
    """
    Loads the trained model from disk.
    """
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

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
            bbox = {
                'xmin': int(float(bndbox.find('xmin').text)),
                'ymin': int(float(bndbox.find('ymin').text)),
                'xmax': int(float(bndbox.find('xmax').text)),
                'ymax': int(float(bndbox.find('ymax').text))
            }
            boxes.append(bbox)
        return boxes
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return []

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    """
    x_left = max(box1['xmin'], box2['xmin'])
    y_top = max(box1['ymin'], box2['ymin'])
    x_right = min(box1['xmax'], box2['xmax'])
    y_bottom = min(box1['ymax'], box2['ymax'])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    box1_area = (box1['xmax'] - box1['xmin'] + 1) * (box1['ymax'] - box1['ymin'] + 1)
    box2_area = (box2['xmax'] - box2['xmin'] + 1) * (box2['ymax'] - box2['ymin'] + 1)
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def apply_nms(detections, iou_threshold):
    """
    Applies Non-Maximum Suppression (NMS) on the detections.
    """
    if len(detections) == 0:
        return []
    
    boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d['score'] for d in detections], dtype=torch.float32)
    
    # Convert boxes to [xmin, ymin, xmax, ymax] format
    boxes = torch.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]], dim=1)
    
    # Apply NMS
    keep_indices = nms(boxes, scores, iou_threshold)
    nms_detections = [detections[i] for i in keep_indices]
    return nms_detections

# ===============================
# Main Execution
# ===============================

def main():
    # Load the trained model
    model = load_model(MODEL_SAVE_PATH, NUM_CLASSES)
    print("Model loaded.")
    
    # Load splits.json to get test and train files
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)
    test_files = splits.get('test', [])
    train_files = splits.get('train', [])
    print(f"Number of test images: {len(test_files)}")
    print(f"Number of train images: {len(train_files)}")
    
    # Load object proposals
    with open(PROPOSALS_FILE, 'r') as f:
        object_proposals = json.load(f)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])    # ImageNet stds
    ])
    
    # Function to process a split (train or test)
    def process_split(split_name, split_files):
        print(f"\nProcessing {split_name} images...\n")
        # Prepare ground truth annotations
        gt_annotations = {}
        for xml_filename in split_files:
            base_filename = os.path.splitext(xml_filename)[0]
            xml_path = os.path.join(ANNOTATED_IMAGES_DIR, xml_filename)
            gt_boxes = parse_annotation(xml_path)
            gt_annotations[base_filename] = gt_boxes
        
        # Store all detections and ground truths for this split
        all_detections = []
        all_gt_boxes = []
        gradcam_count = 0
        
        for xml_filename in tqdm(split_files, desc=f"{split_name.capitalize()} Images"):
            base_filename = os.path.splitext(xml_filename)[0]
            
            # Find the corresponding image file
            image_filename = None
            possible_extensions = ['.jpg', '.png', '.jpeg']
            for ext in possible_extensions:
                candidate_filename = base_filename + ext
                image_path = os.path.join(ANNOTATED_IMAGES_DIR, candidate_filename)
                if os.path.exists(image_path):
                    image_filename = candidate_filename
                    break
            if image_filename is None:
                print(f"No image found for {xml_filename}")
                continue
            
            # Load the image
            image_path = os.path.join(ANNOTATED_IMAGES_DIR, image_filename)
            image = Image.open(image_path).convert('RGB')
            
            # Get proposals for this image
            proposals = object_proposals.get(image_filename, [])
            if not proposals:
                print(f"No proposals found for {image_filename}")
                continue
            
            detections = []
            
            # Process proposals in batches
            batch_size = BATCH_SIZE
            num_batches = len(proposals) // batch_size + int(len(proposals) % batch_size > 0)
            
            for batch_idx in range(num_batches):
                batch_proposals = proposals[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                images = []
                for prop in batch_proposals:
                    bbox = prop
                    cropped_image = image.crop((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
                    cropped_image = transform(cropped_image)
                    images.append(cropped_image)
                if not images:
                    continue
                inputs = torch.stack(images).to(device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = nn.functional.softmax(outputs, dim=1)
                    scores, preds = torch.max(probs, dim=1)
                # Optionally visualize proposals
                
                # Collect detections
                for i in range(len(batch_proposals)):
                    if preds[i].item() == 1:  # Assuming label 1 is the object class
                        score = scores[i].item()
                        if score >= CONFIDENCE_THRESHOLD:
                            detections.append({
                                'bbox': [
                                    batch_proposals[i]['xmin'],
                                    batch_proposals[i]['ymin'],
                                    batch_proposals[i]['xmax'],
                                    batch_proposals[i]['ymax']
                                ],
                                'score': score
                            })
            
            # Apply NMS
            nms_detections = apply_nms(detections, IOU_THRESHOLD_NMS)
            
            # Store detections and ground truths
            for det in nms_detections:
                all_detections.append({
                    'image_id': base_filename,
                    'bbox': det['bbox'],
                    'score': det['score']
                })
            for gt in gt_annotations.get(base_filename, []):
                all_gt_boxes.append({
                    'image_id': base_filename,
                    'bbox': [
                        gt['xmin'],
                        gt['ymin'],
                        gt['xmax'],
                        gt['ymax']
                    ]
                })
            
            # After NMS, visualize detections with GradCAM only for test split
            if split_name == 'test' and nms_detections:
                # Create visualizations directory if it doesn't exist
                if gradcam_count < 10:
                    gradcam_count += 1
                    vis_dir = os.path.join(DATASET_DIR, 'visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # Visualize all detections in a grid
                    output_path = os.path.join(vis_dir, f'{base_filename}_gradcam_all.png')
                    visualize_all_detections(
                        image_path,
                        nms_detections,
                        model,
                        transform,
                        device,
                        output_path
                    )
        
        return all_detections, all_gt_boxes
    
    # Process both train and test splits
    detections_train, gt_boxes_train = process_split('train', train_files)
    detections_test, gt_boxes_test = process_split('test', test_files)
    
    # Evaluate detections using AP metric for both splits
    print("\nEvaluating detections...\n")
    average_precision_train = evaluate_detections(detections_train, gt_boxes_train, IOU_THRESHOLD_EVAL)
    average_precision_test = evaluate_detections(detections_test, gt_boxes_test, IOU_THRESHOLD_EVAL)
    print(f"Average Precision (AP) on Train: {average_precision_train:.4f}")
    print(f"Average Precision (AP) on Test: {average_precision_test:.4f}")

def evaluate_detections(detections, gt_boxes, iou_threshold):
    """
    Evaluates detections using Average Precision (AP) metric.
    """
    # Prepare data for evaluation
    image_ids = list(set([d['image_id'] for d in detections]))
    gt_dict = {}
    for gt in gt_boxes:
        img_id = gt['image_id']
        if img_id not in gt_dict:
            gt_dict[img_id] = []
        gt_dict[img_id].append(gt['bbox'])
    
    # For all detections, compute IoU with ground truths and assign TP or FP
    y_true = []
    y_scores = []
    
    # Create a dictionary to keep track of matched ground truths per image
    matched_gt = {img_id: [] for img_id in gt_dict.keys()}
    
    # Sort detections by score in descending order
    detections_sorted = sorted(detections, key=lambda x: -x['score'])
    
    for det in detections_sorted:
        img_id = det['image_id']
        det_bbox = det['bbox']
        score = det['score']
        y_scores.append(score)
        
        if img_id in gt_dict:
            gt_bboxes = gt_dict[img_id]
            ious = [compute_iou_bbox(det_bbox, gt_bbox) for gt_bbox in gt_bboxes]
            if len(ious) > 0:
                max_iou = max(ious)
                max_iou_idx = ious.index(max_iou)
                if max_iou >= iou_threshold and max_iou_idx not in matched_gt[img_id]:
                    y_true.append(1)  # True Positive
                    matched_gt[img_id].append(max_iou_idx)
                else:
                    y_true.append(0)  # False Positive
            else:
                y_true.append(0)  # False Positive
        else:
            y_true.append(0)  # False Positive
    
    # Compute Average Precision
    if not y_true:
        print("No detections to evaluate.")
        return 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    return average_precision

def compute_iou_bbox(det_bbox, gt_bbox):
    """
    Computes IoU between two bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    x_left = max(det_bbox[0], gt_bbox[0])
    y_top = max(det_bbox[1], gt_bbox[1])
    x_right = min(det_bbox[2], gt_bbox[2])
    y_bottom = min(det_bbox[3], gt_bbox[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    det_area = (det_bbox[2] - det_bbox[0] + 1) * (det_bbox[3] - det_bbox[1] + 1)
    gt_area = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)
    
    iou = intersection_area / float(det_area + gt_area - intersection_area)
    return iou

def visualize_all_detections(image_path, detections, model, transform, device, output_path=None, max_dets=10):
    """
    Visualize multiple detections with GradCAM in a grid.
    """
    # Load and prepare image
    original_image = Image.open(image_path).convert('RGB')
    image_np = np.array(original_image)
    
    # Initialize GradCAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Sort detections by confidence
    sorted_dets = sorted(detections, key=lambda x: -x['score'])[:max_dets]
    num_dets = len(sorted_dets)
    
    if num_dets == 0:
        return
    
    # Create figure
    rows = (num_dets + 2) // 3
    cols = min(4, num_dets + 1)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).ravel()
    # Show original image with all bboxes
    axes[0].imshow(image_np)
    for det in sorted_dets:
        bbox = det['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), 
                           bbox[2] - bbox[0],
                           bbox[3] - bbox[1],
                           fill=False, color='red', linewidth=2)
        axes[0].add_patch(rect)
        axes[0].text(bbox[0], bbox[1]-5, f"{det['score']:.2f}", 
                    color='red', fontsize=10, backgroundcolor='white')
    axes[0].set_title('All Detections')
    axes[0].axis('off')
    
    # Process each detection
    for idx, det in enumerate(sorted_dets, 1):
        bbox = det['bbox']
        cropped_image = original_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # Keep original size for visualization
        original_size = cropped_image.size
        
        # Transform for model
        input_tensor = transform(cropped_image).unsqueeze(0).to(device)
        
        # Create target
        targets = [ClassifierOutputTarget(1)]
        
        # Generate GradCAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]
        
        # Resize grayscale_cam to match original crop size
        grayscale_cam = cv2.resize(grayscale_cam, (original_size[0], original_size[1]))
        
        # Convert crop to RGB numpy array
        rgb_img = np.array(cropped_image) / 255.0
        
        # Overlay GradCAM on image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Plot visualization
        axes[idx].imshow(visualization)
        axes[idx].set_title(f'Detection {idx} (conf: {det["score"]:.3f})')
        axes[idx].axis('off')
    
    # Turn off any unused subplots
    for idx in range(num_dets + 1, len(axes)):
        axes[idx].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    main()