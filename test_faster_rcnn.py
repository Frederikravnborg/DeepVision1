import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np


# ===============================
# Configuration
# ===============================
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
MODEL_PATH = os.path.join(DATASET_DIR, 'fasterrcnn_pothole_model.pth')
BATCH_SIZE = 4
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD_EVAL = 0.5
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Utility Functions
# ===============================
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
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
        return {'boxes': boxes, 'labels': [1] * len(boxes)}  # 1 represents the class ID for potholes
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
        img_path = os.path.join(self.annotations_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        xml_path = img_path.replace('.jpg', '.xml')
        annotation = parse_annotation(xml_path)
        boxes = torch.as_tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(annotation['labels'], dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': idx}  # include image_id

        if self.transform:
            image = self.transform(image)
        return image, target
    

def load_model(model_path):
    """
    Loads the Faster R-CNN model with pre-trained weights.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def compute_average_iou(model, dataloader, device, confidence_threshold):
    """
    Computes the average IoU for the validation set.
    """
    model.eval()
    total_iou = 0
    total_boxes = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Computing IoU", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for idx, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                gt_boxes = targets[idx]['boxes'].cpu()

                if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                    continue

                # Filter predictions by confidence threshold
                pred_boxes = pred_boxes[pred_scores >= confidence_threshold]

                # Compute IoU
                iou_matrix = box_iou(gt_boxes, pred_boxes)
                max_iou_per_gt = iou_matrix.max(dim=1).values if iou_matrix.numel() > 0 else torch.tensor([])

                # Accumulate IoUs and counts
                total_iou += max_iou_per_gt.sum().item()
                total_boxes += len(gt_boxes)

    average_iou = total_iou / total_boxes
    return average_iou


def compute_iou_bbox(pred_bbox, gt_bbox):
    """
    Compute Intersection over Union (IoU) for a single bounding box prediction and ground truth.
    """
    x1, y1, x2, y2 = pred_bbox
    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox

    # Calculate intersection area
    inter_x1 = max(x1, x1_gt)
    inter_y1 = max(y1, y1_gt)
    inter_x2 = min(x2, x2_gt)
    inter_y2 = min(y2, y2_gt)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate areas
    pred_area = (x2 - x1) * (y2 - y1)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    # Compute IoU
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / union_area
    return iou


def evaluate_detections(detections, gt_boxes, iou_threshold=0.5):
    """
    Evaluates detections using Average Precision (AP) metric.
    """
    # Prepare ground truth data per image
    image_ids = list(set([d['image_id'] for d in detections]))
    gt_dict = {img_id: [] for img_id in image_ids}
    for gt in gt_boxes:
        img_id = gt['image_id']
        if img_id in gt_dict:
            gt_dict[img_id].append(gt['bbox'])

    # Lists to store true labels (y_true) and detection scores (y_scores)
    y_true = []
    y_scores = []

    # Iterate through each image and evaluate detections
    for img_id in image_ids:
        img_detections = [d for d in detections if d['image_id'] == img_id]
        img_gt_boxes = gt_dict.get(img_id, [])
        matched_gt = set()

        # Sort detections by descending confidence score
        for det in sorted(img_detections, key=lambda x: -x['score']):
            iou_max = 0.0
            assigned_gt = None

            # Compare each detection with all un-matched ground truths
            for gt_bbox in img_gt_boxes:
                # Convert numpy arrays to tuples for hashability
                gt_bbox_tuple = tuple(gt_bbox.tolist()) if isinstance(gt_bbox, np.ndarray) else tuple(gt_bbox)

                if gt_bbox_tuple in matched_gt:
                    continue  # Skip already matched ground truth boxes
                iou = compute_iou_bbox(det['bbox'], gt_bbox)
                if iou > iou_max:
                    iou_max = iou
                    assigned_gt = gt_bbox_tuple

            y_scores.append(det['score'])

            # Check if the IoU meets the threshold to consider the detection a true positive
            if iou_max >= iou_threshold:
                y_true.append(1)
                matched_gt.add(assigned_gt)  # Mark ground truth as matched
            else:
                y_true.append(0)

    # If no true labels exist, return 0 AP (nothing to evaluate)
    if not y_true:
        print("No detections to evaluate.")
        return 0.0

    # Compute Precision, Recall, and Average Precision
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    return average_precision




# ===============================
# Main Script
# ===============================
if __name__ == "__main__":
    # Load Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = PotholeDataset(ANNOTATED_IMAGES_DIR, transform=transform)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Load Model
    model = load_model(MODEL_PATH)

    # Initialize lists to store detections and GT boxes
    detections = []
    gt_boxes = []

    # In the evaluation loop, move tensors to the device, but skip non-tensor fields (like 'image_id')
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating", leave=False):
            images = [img.to(device) for img in images]
            
            # Only move tensor fields to device and leave out non-tensor fields like 'image_id'
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} 
                for t in targets
            ]
            
            outputs = model(images)

            # Collect detections and ground truth boxes
            for idx, output in enumerate(outputs):
                for box, score in zip(output['boxes'], output['scores']):
                    if score >= CONFIDENCE_THRESHOLD:
                        detections.append({
                            'image_id': targets[idx]['image_id'],  # Access image_id correctly
                            'bbox': box.cpu().numpy(),
                            'score': score.cpu().item()
                        })
                for gt_box in targets[idx]['boxes']:
                    gt_boxes.append({
                        'image_id': targets[idx]['image_id'],
                        'bbox': gt_box.cpu().numpy()
                    })


    # Compute Average Precision
    average_precision = evaluate_detections(detections, gt_boxes, iou_threshold=IOU_THRESHOLD_EVAL)
    print(f"Average Precision (AP) on validation set: {average_precision:.4f}")
    
    # Compute Average IoU
    average_iou = compute_average_iou(model, val_loader, device, CONFIDENCE_THRESHOLD)
    print(f"Average IoU on validation set: {average_iou:.4f}")
