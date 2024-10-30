import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_proposals(image, proposals, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for (x, y, w, h) in proposals[:50]:  # Visualize first 50 proposals
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()

# Replace this path with the path to a single image in your dataset
sample_image_path = 'Potholes/annotated-images/img-110.jpg'

# Read the image
image = cv2.imread(sample_image_path)
if image is None:
    print(f"Failed to read image: {sample_image_path}")
    exit(1)

# Initialize Selective Search
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Set the base image
ss.setBaseImage(image)

# Switch to Fast Mode
# ss.switchToSelectiveSearchFast()

# Run Selective Search
rects = ss.process()

print(f"Number of proposals: {len(rects)}")

# Visualize the first 50 proposals
visualize_proposals(image, rects)
