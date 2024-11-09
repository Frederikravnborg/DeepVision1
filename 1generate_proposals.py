import os
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import random
import cv2
import numpy as np
from tqdm import tqdm

# Define the paths
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
SPLITS_FILE = os.path.join(DATASET_DIR, 'splits.json')

# Load splits.json
with open(SPLITS_FILE, 'r') as f:
    splits = json.load(f)

train_files = splits.get('train', [])
test_files = splits.get('test', [])

print(f"Number of training samples: {len(train_files)}")
print(f"Number of test samples: {len(test_files)}")

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
                'xmin': int(bndbox.find('xmin').text),
                'ymin': int(bndbox.find('ymin').text),
                'xmax': int(bndbox.find('xmax').text),
                'ymax': int(bndbox.find('ymax').text)
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

# def resize_image(image, max_width=800, max_height=800):
#     """
#     Resizes an image to fit within max_width and max_height while maintaining aspect ratio.
#     """
#     height, width = image.shape[:2]
#     scaling_factor = min(max_width / width, max_height / height, 1)  # Prevent upscaling

#     if scaling_factor < 1:
#         new_size = (int(width * scaling_factor), int(height * scaling_factor))
#         resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
#         return resized_image
#     return image

def save_object_proposals(proposals, filepath):
    """
    Saves object proposals to a JSON file.
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(proposals, f, indent=4)
        print(f"Object proposals successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving object proposals to {filepath}: {e}")

def load_object_proposals(filepath):
    """
    Loads object proposals from a JSON file.
    """
    try:
        with open(filepath, 'r') as f:
            proposals = json.load(f)
        return proposals
    except FileNotFoundError:
        print(f"Proposals file not found at {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading proposals from {filepath}: {e}")
        return {}

def extract_selective_search_proposals(image_path, ss, max_width=800, max_height=600, max_proposals=1000):
    """
    Extracts object proposals from an image using Selective Search.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return []
    
    # Resize image
    # resized_image = resize_image(image, max_width, max_height)
    
    # Set base image
    ss.setBaseImage(image)
    
    # Run selective search
    rects = ss.process()
    
    proposals = []
    for (x, y, w, h) in rects[:max_proposals]:
        bbox = {
            'xmin': int(x),
            'ymin': int(y),
            'xmax': int(x + w),
            'ymax': int(y + h)
        }
        proposals.append(bbox)
    
    return proposals

def visualize_proposals(image, proposals, num_proposals=50, figsize=(10, 8)):
    """
    Visualizes a subset of object proposals on an image.
    """
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for (x, y, w, h) in proposals[:num_proposals]:
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()

# Initialize Selective Search Segmentation
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
print("Selective Search Segmentation object created.")


# Prepare to store proposals
object_proposals = {}

# Get all image filenames from train and test sets
all_files = train_files + test_files

# Determine image extensions
possible_extensions = ['.jpg', '.png', '.jpeg']

# Process each image
print("\nExtracting Selective Search Proposals for All Images...\n")
for xml_filename in tqdm(all_files, desc="Processing Images"):
    # Derive the corresponding image filename
    base_filename = os.path.splitext(xml_filename)[0]
    
    # Find the image file with a valid extension
    image_found = False
    for ext in possible_extensions:
        image_filename = base_filename + ext
        image_path = os.path.join(ANNOTATED_IMAGES_DIR, image_filename)
        if os.path.exists(image_path):
            image_found = True
            break
    
    # Read the image to ensure it's readable
    image = cv2.imread(image_path)

    
    # Set base image and switch to Fast Mode
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    
    # Extract proposals
    rects = ss.process()
    
    # Convert rects to bounding boxes
    proposals = []
    for (x, y, w, h) in rects[:1000]:  # Limit to first 1000 proposals
        bbox = {
            'xmin': int(x),
            'ymin': int(y),
            'xmax': int(x + w),
            'ymax': int(y + h)
        }
        proposals.append(bbox)
    
    # Store the proposals
    object_proposals[image_filename] = proposals

# Save all proposals to a JSON file
PROPOSALS_FILE = os.path.join(DATASET_DIR, 'selective_search_proposals.json')
save_object_proposals(object_proposals, PROPOSALS_FILE)
print(f"\nObject proposals saved to {PROPOSALS_FILE}")
