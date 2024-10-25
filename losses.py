import torch


def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))


def bce_loss2(y_real, y_pred):
    # Clip predictions to prevent numerical instability
    eps = 1e-7
    y_pred = torch.clamp(y_pred, min=-100, max=100)  # Prevent extreme values
    
    # Apply sigmoid to get probabilities
    y_pred_sigmoid = torch.sigmoid(y_pred)
    y_pred_sigmoid = torch.clamp(y_pred_sigmoid, min=eps, max=1-eps)  # Prevent log(0)
    
    # Calculate BCE
    loss = -(y_real * torch.log(y_pred_sigmoid) + (1 - y_real) * torch.log(1 - y_pred_sigmoid))
    return torch.mean(loss)

def dice(y_real, y_pred):
    return 1 - torch.mean(2 * y_real*y_pred + 1) / torch.mean(y_real + y_pred + 1)

def intersection_over_union(y_real, y_pred):
    """Calculate Intersection over Union (IoU) between binary masks."""
    # Ensure inputs are binary
    y_pred = torch.sigmoid(y_pred)  # If predictions are logits
    y_pred = (y_pred > 0.5).float()
    
    # Calculate intersection and union
    intersection = torch.sum(y_real * y_pred, dim=(1,2,3))
    union = torch.sum(y_real, dim=(1,2,3)) + torch.sum(y_pred, dim=(1,2,3)) - intersection
    
    # Calculate IoU
    iou = (intersection + 1e-7) / (union + 1e-7)  # Increased epsilon for numerical stability
    return torch.mean(iou)  # Return mean IoU across batch

def accuracy(y_real, y_pred):
    """Calculate binary classification accuracy."""
    # Apply sigmoid if predictions are logits
    y_pred = torch.sigmoid(y_pred)
    y_pred_bin = (y_pred > 0.5).float()
    
    # Calculate accuracy
    correct = torch.sum(y_real == y_pred_bin, dim=(1,2,3))
    total = y_real.numel() / y_real.size(0)  # Account for batch size
    
    return torch.mean(correct.float() / total)

def sensitivity(y_real, y_pred):
    """Calculate sensitivity/recall (true positive rate)."""
    # Apply sigmoid if predictions are logits
    y_pred = torch.sigmoid(y_pred)
    y_pred_bin = (y_pred > 0.5).float()
    
    # Calculate true positives and false negatives
    true_positive = torch.sum((y_real == 1) & (y_pred_bin == 1), dim=(1,2,3))
    false_negative = torch.sum((y_real == 1) & (y_pred_bin == 0), dim=(1,2,3))
    
    # Calculate sensitivity
    sens = (true_positive + 1e-7) / (true_positive + false_negative + 1e-7)
    return torch.mean(sens)

def specificity(y_real, y_pred):
    """Calculate specificity (true negative rate)."""
    # Apply sigmoid if predictions are logits
    y_pred = torch.sigmoid(y_pred)
    y_pred_bin = (y_pred > 0.5).float()
    
    # Calculate true negatives and false positives
    true_negative = torch.sum((y_real == 0) & (y_pred_bin == 0), dim=(1,2,3))
    false_positive = torch.sum((y_real == 0) & (y_pred_bin == 1), dim=(1,2,3))
    
    # Calculate specificity
    spec = (true_negative + 1e-7) / (true_negative + false_positive + 1e-7)
    return torch.mean(spec)

def focal_loss(y_real, y_pred, alpha=0.25, gamma=2.0):
    
    # Apply sigmoid to get probabilities
    y_pred = torch.sigmoid(y_pred)
    
    # Clip predictions for numerical stability
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    
    # Calculate binary cross entropy
    bce_loss = -y_real * torch.log(y_pred) - (1 - y_real) * torch.log(1 - y_pred)
    
    # Calculate focal term
    pt = torch.where(y_real == 1, y_pred, 1 - y_pred)
    focal_term = (1 - pt) ** gamma
    
    # Apply class balancing
    alpha_t = torch.where(y_real == 1, alpha, 1 - alpha)
    
    # Combine all terms
    loss = alpha_t * focal_term * bce_loss
    
    return torch.mean(loss)

def bce_total_variation(y_real, y_pred, lambda_tv=0.1):

    # Apply sigmoid to get probabilities
    y_pred = torch.sigmoid(y_pred)
    
    # Calculate BCE loss
    bce = F.binary_cross_entropy(y_pred, y_real, reduction='mean')
    
    # Calculate total variation
    tv_h = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]).mean()
    tv_w = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]).mean()
    tv = tv_h + tv_w
    
    return bce + lambda_tv * tv
