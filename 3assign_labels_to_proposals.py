import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import random
import pickle

# Paths
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
PROPOSALS_FILE = os.path.join(DATASET_DIR, 'selective_search_proposals_fast.json')
SPLITS_FILE = os.path.join(DATASET_DIR, 'splits.json')

# Load splits.json
with open(SPLITS_FILE, 'r') as f:
    splits = json.load(f)

train_files = splits.get('train', [])

# Load object proposals
with open(PROPOSALS_FILE, 'r') as f:
    object_proposals = json.load(f)

# Function to parse ground truth annotations
def parse_annotation(xml_file):
    """
    Parses a Pascal VOC XML file and extracts bounding box coordinates.
    """
    try:
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
    except ET.ParseError as e:
        print(f"XML parsing error in {xml_file}: {e}")
        return []
    except AttributeError as e:
        print(f"Missing tags in {xml_file}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error parsing {xml_file}: {e}")
        return []

# Function to compute IoU between two bounding boxes
def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Each box is a dict with keys: xmin, ymin, xmax, ymax
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

# Thresholds for assigning labels
POSITIVE_IOU_THRESHOLD = 0.5
NEGATIVE_IOU_THRESHOLD = 0.3

# Maximum number of proposals to consider per image
MAX_PROPOSALS_PER_IMAGE = 2000  # Adjust as needed

# Initialize structures to hold training data and ground truths
training_data = []
ground_truth_data = {}

print("\nAssigning labels to proposals for training...\n")

for xml_filename in tqdm(train_files, desc="Processing training images"):
    # Get base filename
    base_filename = os.path.splitext(xml_filename)[0]
    xml_path = os.path.join(ANNOTATED_IMAGES_DIR, xml_filename)
    
    # Load ground truth boxes
    gt_boxes = parse_annotation(xml_path)
    
    # Store ground truth boxes in ground_truth_data
    # Keyed by image filename
    image_filename = None
    possible_extensions = ['.jpg', '.png', '.jpeg']
    for ext in possible_extensions:
        candidate_filename = base_filename + ext
        if candidate_filename in object_proposals:
            image_filename = candidate_filename
            break
    if image_filename is None:
        print(f"No proposals found for {xml_filename}")
        continue
    
    ground_truth_data[image_filename] = gt_boxes  # Save ground truths
    
    proposals = object_proposals[image_filename]
    # Limit to the first MAX_PROPOSALS_PER_IMAGE
    proposals = proposals[:MAX_PROPOSALS_PER_IMAGE]
    
    # Lists to hold positive and negative proposals for this image
    positive_proposals = []
    negative_proposals = []
    
    for prop_box in proposals:
        max_iou = 0.0
        max_iou_gt_box = None
        for gt_box in gt_boxes:
            iou = compute_iou(gt_box, prop_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_gt_box = gt_box
        
        if max_iou >= POSITIVE_IOU_THRESHOLD:
            # Positive sample
            positive_proposals.append({
                'image_filename': image_filename,
                'bbox': prop_box,
                'label': 1,  # Adjust according to your class labels
                'gt_bbox': max_iou_gt_box
            })
        elif max_iou < NEGATIVE_IOU_THRESHOLD:
            # Negative sample
            negative_proposals.append({
                'image_filename': image_filename,
                'bbox': prop_box,
                'label': 0  # Background class
            })
        else:
            # Neutral sample, ignore or handle separately if needed
            pass
    
    # Balance positive and negative samples to prevent too much class imbalance
    num_positives = len(positive_proposals)
    num_negatives = len(negative_proposals)
    
    if num_positives > 0:
        num_negatives_to_sample = min(num_negatives, num_positives * 3)  # Ratio of 1:3
        sampled_negatives = random.sample(negative_proposals, num_negatives_to_sample)
        
        # Add to training data
        training_data.extend(positive_proposals)
        training_data.extend(sampled_negatives)
    else:
        # If no positive samples, you might decide to skip or handle differently
        pass

print(f"\nTotal training samples: {len(training_data)}")

# Combine proposals and ground truths into a single dictionary
combined_data = {
    'proposals': training_data,
    'ground_truths': ground_truth_data
}

# Save the combined data to a file for later use
TRAINING_DATA_FILE = os.path.join(DATASET_DIR, 'training_data_with_gt.pkl')
with open(TRAINING_DATA_FILE, 'wb') as f:
    pickle.dump(combined_data, f)
print(f"Training data with ground truths saved to {TRAINING_DATA_FILE}")