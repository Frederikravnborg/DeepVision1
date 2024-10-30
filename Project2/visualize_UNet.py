import torch
import torch.nn as nn
import torch.nn.functional as F
from models import UNet
from torchviz import make_dot
import matplotlib.pyplot as plt
import hiddenlayer as hl

# # Instantiate the model
# model = UNet()

# # Create a sample input tensor (e.g., batch size 1, 3 channels, 256x256 image)
# sample_input = torch.randn(1, 3, 256, 256)

# # Perform a forward pass
# output = model(sample_input)

# # Generate the visualization
# dot = make_dot(output, params=dict(model.named_parameters()))

# # Save the visualization to a file
# dot.format = 'png'
# dot.render('unet_architecture', cleanup=True)

# print("UNet architecture has been saved as 'unet_architecture.png'")


# Instantiate the model
model = UNet()

# Create a sample input tensor
sample_input = torch.randn(1, 3, 256, 256)

# Create a hiddenlayer graph
graph = hl.build_graph(model, sample_input)

# Render the graph to a file
graph.theme = hl.graph.THEMES["blue"].copy()
graph.save("unet_hiddenlayer", format="png")

print("UNet architecture has been saved as 'unet_hiddenlayer.png'")

