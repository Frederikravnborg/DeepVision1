import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

# ===============================
# Configuration
# ===============================

# Path to the trained model
MODEL_PATH = 'Potholes/fasterrcnn_pothole_model.pth'

# Confidence threshold for displaying boxes
CONFIDENCE_THRESHOLD = 0.5

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load Model
# ===============================
def load_model(model_path):
    # Load the pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace the box predictor to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # 2 classes (background + 1 object)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ===============================
# Prediction and Visualization
# ===============================

def predict_and_draw_boxes(model, image_path, save_path=None):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract the boxes, labels, and scores
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = box
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=3)
            draw.text((xmin, ymin - 10), f"Score: {score:.2f}", fill='red', font=font)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # Save the image if a save path is provided
    if save_path:
        image.save(save_path)
        print(f"Saved predicted image with bounding boxes at: {save_path}")

# ===============================
# Main Script
# ===============================
if __name__ == "__main__":
    # Load the model
    model = load_model(MODEL_PATH)

    # Specify the image file for prediction
    IMAGE_PATH = 'Potholes/annotated-images/img-6.jpg'  # Change to your image file path

    # Predict and draw boxes on the image
    predict_and_draw_boxes(model, IMAGE_PATH)
