import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Evaluation parameters
# ================================================
iou_threshold = 0.5
max_proposals = 1000
proposal_counts = range(100, max_proposals+1, 100)
# ================================================


# Paths
DATASET_DIR = 'Potholes/'  
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

# Initialize recall list
recall_list = []

# Loop over different numbers of proposals
for num_proposals in proposal_counts:
    total_gt_boxes = 0
    total_covered = 0
    
    for xml_filename in tqdm(train_files, desc=f"Evaluating with {num_proposals} proposals"):
        # Get base filename
        base_filename = os.path.splitext(xml_filename)[0]
        xml_path = os.path.join(ANNOTATED_IMAGES_DIR, xml_filename)
        
        # Load ground truth boxes
        gt_boxes = parse_annotation(xml_path)
        
        # Total ground truth boxes
        total_gt_boxes += len(gt_boxes)
        
        # Get proposals for this image
        # Note: in your code, proposals are stored with the image filename as key
        # Need to find the corresponding image filename (with extension)
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
        
        proposals = object_proposals[image_filename]
        # Limit to the first num_proposals
        proposals = proposals[:num_proposals]
        
        # For each ground truth box, check if it is covered by any proposal
        for gt_box in gt_boxes:
            covered = False
            for prop_box in proposals:
                iou = compute_iou(gt_box, prop_box)
                if iou >= iou_threshold:
                    covered = True
                    break
            if covered:
                total_covered += 1
    
    # Compute recall
    recall = total_covered / total_gt_boxes if total_gt_boxes > 0 else 0
    recall_list.append(recall)
    print(f"Recall with {num_proposals} proposals: {recall:.4f}")

# Plot Recall vs Number of Proposals
plt.figure(figsize=(8, 6))
plt.plot(proposal_counts, recall_list, marker='o')
plt.title('Recall vs Number of Proposals', fontsize=24)
plt.xlabel('Number of Proposals', fontsize=22)
plt.ylabel('Recall', fontsize=22)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig('recall_vs_proposals.png', dpi=300)
# plt.show()
