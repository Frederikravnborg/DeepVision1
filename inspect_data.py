import os
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import random


# Define the paths
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
SPLITS_FILE = os.path.join(DATASET_DIR, 'splits.json')

# Load splits.json
with open(SPLITS_FILE, 'r') as f:
    splits = json.load(f)

train_files = splits['train']
test_files = splits['test']

print(f"Number of training samples: {len(train_files)}")
print(f"Number of test samples: {len(test_files)}")

def parse_annotation(xml_file):
    """
    Parses a Pascal VOC XML file and extracts bounding box coordinates.

    Args:
        xml_file (str): Path to the XML annotation file.

    Returns:
        List of dictionaries containing bounding box coordinates.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        # Extract the bounding box
        bndbox = obj.find('bndbox')
        bbox = {
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        }
        boxes.append(bbox)
    return boxes


def visualize_image_with_boxes(image_path, boxes, figsize=(10, 8)):
    """
    Displays an image with bounding boxes.

    Args:
        image_path (str): Path to the image file.
        boxes (list): List of bounding boxes.
        figsize (tuple): Size of the plot.
    """
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()
    
    for box in boxes:
        # Calculate width and height of the bounding box
        width = box['xmax'] - box['xmin']
        height = box['ymax'] - box['ymin']
        # Create a Rectangle patch
        rect = Rectangle((box['xmin'], box['ymin']), width, height, linewidth=2, edgecolor='red', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.show()



def visualize_train_samples(num_samples=5):
    selected_train_files = random.sample(train_files, num_samples)

    for xml_filename in selected_train_files:
        # Derive the corresponding image filename
        base_filename = os.path.splitext(xml_filename)[0]
        
        # Assuming images have extensions like .jpg or .png
        # You may need to adjust this based on your dataset
        for ext in ['.jpg', '.png', '.jpeg']:
            image_filename = base_filename + ext
            image_path = os.path.join(ANNOTATED_IMAGES_DIR, image_filename)
            if os.path.exists(image_path):
                break
        else:
            print(f"Image file for {xml_filename} not found.")
            continue
        
        # Parse the annotation
        xml_path = os.path.join(ANNOTATED_IMAGES_DIR, xml_filename)
        boxes = parse_annotation(xml_path)
        
        # Visualize
        print(f"Visualizing {image_filename} with {len(boxes)} bounding box(es).")
        visualize_image_with_boxes(image_path, boxes)


def visualize_test_samples(num_samples=5):
    # Select random test samples
    selected_test_files = random.sample(test_files, num_samples)

    for xml_filename in selected_test_files:
        # Derive the corresponding image filename
        base_filename = os.path.splitext(xml_filename)[0]
        
        # Assuming images have extensions like .jpg or .png
        # You may need to adjust this based on your dataset
        for ext in ['.jpg', '.png', '.jpeg']:
            image_filename = base_filename + ext
            image_path = os.path.join(ANNOTATED_IMAGES_DIR, image_filename)
            if os.path.exists(image_path):
                break
        else:
            print(f"Image file for {xml_filename} not found.")
            continue
        
        # Parse the annotation
        xml_path = os.path.join(ANNOTATED_IMAGES_DIR, xml_filename)
        boxes = parse_annotation(xml_path)
        
        # Visualize
        print(f"Visualizing {image_filename} with {len(boxes)} bounding box(es).")
        visualize_image_with_boxes(image_path, boxes)


if __name__ == '__main__':
    # visualize_train_samples()
    visualize_test_samples()