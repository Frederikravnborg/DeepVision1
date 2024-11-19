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

DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
SPLITS_FILE = os.path.join(DATASET_DIR, 'splits.json')
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')


# Load splits.json to get test files
with open(SPLITS_FILE, 'r') as f:
    splits = json.load(f)
test_files = splits.get('test', [])
print(f"Number of test images: {len(test_files)}")
test_image_files = [os.path.splitext(f)[0] + '.jpg' for f in test_files]
print([f for f in os.listdir(ANNOTATED_IMAGES_DIR) if f.endswith('.jpg') and f not in test_image_files])