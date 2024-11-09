import os
import json
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import cv2

# Paths (adjust as necessary)
DATASET_DIR = 'Potholes/'  # Replace with your dataset directory path
ANNOTATED_IMAGES_DIR = os.path.join(DATASET_DIR, 'annotated-images')
TRAINING_DATA_FILE = os.path.join(DATASET_DIR, 'training_data_with_gt.pkl')
SPLITS_FILE = os.path.join(DATASET_DIR, 'splits.json')

# Number of examples to visualize
NUM_EXAMPLES = 5

# Load splits.json to get training files
with open(SPLITS_FILE, 'r') as f:
    splits = json.load(f)

train_files = splits.get('train', [])

# Load labeled training data with ground truths
with open(TRAINING_DATA_FILE, 'rb') as f:
    combined_data = pickle.load(f)

training_data = combined_data['proposals']
ground_truth_data = combined_data['ground_truths']

# Extract unique image filenames from training_data
unique_images = list(set([item['image_filename'] for item in training_data]))
if len(unique_images) < NUM_EXAMPLES:
    print(f"Only {len(unique_images)} unique images available for visualization.")
    NUM_EXAMPLES = len(unique_images)

# Randomly select images to visualize
selected_images = random.sample(unique_images, NUM_EXAMPLES)

for image_filename in selected_images:
    # Find all proposals for this image
    proposals = [item for item in training_data if item['image_filename'] == image_filename]
    
    # Separate positive and negative proposals
    positive_proposals = [p['bbox'] for p in proposals if p['label'] == 1]
    negative_proposals = [p['bbox'] for p in proposals if p['label'] == 0]
    
    # Load the image
    image_path = os.path.join(ANNOTATED_IMAGES_DIR, image_filename)
    if not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist.")
        continue
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}.")
        continue
    
    # Convert BGR to RGB for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"Image: {image_filename}")
    ax.axis('off')
    
    # Load and plot ground truth boxes in green
    gt_boxes = ground_truth_data.get(image_filename, [])
    for gt in gt_boxes:
        rect = patches.Rectangle((gt['xmin'], gt['ymin']),
                                 gt['xmax'] - gt['xmin'],
                                 gt['ymax'] - gt['ymin'],
                                 linewidth=5, edgecolor='green', facecolor='none', label='Ground Truth')
        ax.add_patch(rect)
    
    # Plot positive proposals in blue
    for prop in positive_proposals:
        rect = patches.Rectangle((prop['xmin'], prop['ymin']),
                                 prop['xmax'] - prop['xmin'],
                                 prop['ymax'] - prop['ymin'],
                                 linewidth=3, edgecolor='blue', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
    
    # Plot negative proposals in red
    for prop in negative_proposals:
        rect = patches.Rectangle((prop['xmin'], prop['ymin']),
                                 prop['xmax'] - prop['xmin'],
                                 prop['ymax'] - prop['ymin'],
                                 linewidth=3, edgecolor='red', facecolor='none', alpha=0.3)
        ax.add_patch(rect)
    
    # Create custom legends
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Ground Truth'),
        Line2D([0], [0], color='blue', lw=4, label='Positive Proposals'),
        Line2D([0], [0], color='red', lw=4, label='Negative Proposals')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.show()