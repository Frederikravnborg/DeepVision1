import torch

#### Losses ####

def bce_loss_unstable(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))


def bce_loss(y_real, y_pred):
    # Clip predictions to prevent numerical instability
    eps = 1e-7
    y_pred = torch.clamp(y_pred, min=-100, max=100)  # Prevent extreme values
    
    # Apply sigmoid to get probabilities
    y_pred_sigmoid = torch.sigmoid(y_pred)
    y_pred_sigmoid = torch.clamp(y_pred_sigmoid, min=eps, max=1-eps)  # Prevent log(0)
    
    # Calculate BCE
    loss = -(y_real * torch.log(y_pred_sigmoid) + (1 - y_real) * torch.log(1 - y_pred_sigmoid))
    return torch.mean(loss)

def focal_loss(y_real, y_pred, alpha=0.25, gamma=2.0):
    bce_loss = - (y_real * torch.log(y_pred) + (1 - y_real) * torch.log(1 - y_pred))
    pt = torch.where(y_real == 1, y_pred, 1 - y_pred)  # pt is the predicted probability for the true class

    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return torch.mean(focal_loss)


def bce_total_variation(y_real, y_pred):
    return bce_loss(y_real, y_pred) + 0.1*...


#### Metrics ####

def dice(y_real, y_pred):
    return 1 - torch.mean(2 * y_real*y_pred + 1) / torch.mean(y_real + y_pred + 1)

def intersection_over_union(y_real, y_pred):
    # Calculate Intersection over Union (IoU)
    intersection = torch.sum(y_real * y_pred)
    union = torch.sum(y_real) + torch.sum(y_pred) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)  # Add small value to prevent division by zero
    return iou

def accuracy(y_real, y_pred):
    # Calculate Accuracy
    y_pred_bin = torch.round(y_pred)  # Convert predictions to binary (0 or 1)
    correct = torch.sum(y_real == y_pred_bin)
    total = torch.numel(y_real)
    return correct.float() / total

def sensitivity(y_real, y_pred):
    # Calculate Sensitivity (Recall or True Positive Rate)
    y_pred_bin = torch.round(y_pred)
    true_positive = torch.sum((y_real == 1) & (y_pred_bin == 1))
    false_negative = torch.sum((y_real == 1) & (y_pred_bin == 0))
    return (true_positive + 1e-6) / (true_positive + false_negative + 1e-6)

def specificity(y_real, y_pred):
    # Calculate Specificity (True Negative Rate)
    y_pred_bin = torch.round(y_pred)
    true_negative = torch.sum((y_real == 0) & (y_pred_bin == 0))
    false_positive = torch.sum((y_real == 0) & (y_pred_bin == 1))
    return (true_negative + 1e-6) / (true_negative + false_positive + 1e-6)


