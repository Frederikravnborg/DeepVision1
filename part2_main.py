import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
import csv

import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
from models import *
from dataloaders import *
from losses import *
PH2_path = '/dtu/datasets1/02516/PH2_Dataset_images/'

def point_level_loss(preds, pos_clicks, neg_clicks):
    # Convert the preds to a probability map using sigmoid (if not already)
    preds = torch.sigmoid(preds)  # Assumes preds are logits

    # Create tensors for storing loss
    pos_loss = 0.0
    neg_loss = 0.0

    # Process positive clicks
    if len(pos_clicks) > 0:
        pos_preds = preds[tuple(zip(*pos_clicks))]  # Get predictions at positive click points
        pos_loss = F.binary_cross_entropy(pos_preds, torch.ones_like(pos_preds))  # Loss at positive clicks

    # Process negative clicks
    if len(neg_clicks) > 0:
        neg_preds = preds[tuple(zip(*neg_clicks))]  # Get predictions at negative click points
        neg_loss = F.binary_cross_entropy(neg_preds, torch.zeros_like(neg_preds))  # Loss at negative clicks

    # Total loss is the sum of positive and negative click losses
    total_loss = pos_loss + neg_loss

    return total_loss

size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

batch_size = 10
ph2_train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
ph2_test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
ph2set = Ph2(transform=train_transform)
train_size = int(0.7 * len(ph2set))
val_size = int(0.1 * len(ph2set))
test_size = len(ph2set) - train_size - val_size
train_ph2, val_ph2, test_ph2 = random_split(ph2set, [train_size, val_size,test_size])
train_loader = DataLoader(train_ph2, batch_size=batch_size, shuffle=True, num_workers=3)
val_loader = DataLoader(val_ph2, batch_size=batch_size, shuffle=False, num_workers=3)
test_loader = DataLoader(test_ph2, batch_size=batch_size, shuffle=False, num_workers=3)

print('Loaded %d training images' % len(train_ph2))
print('Loaded %d validation images' % len(val_ph2))
print('Loaded %d test images' % len(test_ph2))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



def train(model, opt, loss_fn, epochs, train_loader, test_loader, clicks):
    X_test, Y_test = next(iter(test_loader))
    #print(len(X_test))
    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        # Initialize lists to save all positive and negative clicks for the entire epoch
        epoch_pos_clicks = []
        epoch_neg_clicks = []
        gambiarra = None
        for X_batch, Y_batch in train_loader:
            gambiarra = X_batch
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            all_pos_clicks, all_neg_clicks = [], []
            
            for i, mask in enumerate(Y_batch):
                pos_clicks, neg_clicks = generate_clicks(mask.cpu(), clicks, clicks)
                #print(f"Image {i} - Pos Clicks: {pos_clicks}, Neg Clicks: {neg_clicks}")
                all_pos_clicks.append(pos_clicks)
                all_neg_clicks.append(neg_clicks)
            # set parameter gradients to zero
            opt.zero_grad()
            # forward
            Y_pred = model(X_batch)
            epoch_pos_clicks.append(all_pos_clicks)
            epoch_neg_clicks.append(all_neg_clicks)
            if torch.isnan(Y_pred).any():
                print('Y_pred is nan')
                break

            # Calculate loss using only point-level annotations (clicks)
            loss = 0
            for i in range(len(Y_batch)):
                pos_clicks = all_pos_clicks[i]
                neg_clicks = all_neg_clicks[i]
                loss += loss_fn(Y_pred[i], pos_clicks, neg_clicks)  # Loss with clicks

            # Average loss over the batch
            loss /= len(Y_batch)
            
            if torch.isnan(loss):   
                print('Loss is nan')
                break

            if torch.isinf(loss):
                print('Loss is inf')
                break
            loss.backward()  # backward-pass
            
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        toc = time()
        print(f' - epoch loss: {loss}')
        print(' - loss: %f' % avg_loss)

        # Show intermediate results
        if epoch == 19:
            model.eval()  # testing mode
            Y_hat = torch.sigmoid(model(gambiarra.to(device))).detach().cpu()

            for k in range(10):
                # Get the real image and the corresponding generated clicks
                real_image = np.rollaxis(gambiarra[k].numpy(), 0, 3)  # Convert to HxWxC format
                pos_clicks = epoch_pos_clicks[-1][k]  # Retrieve the positive clicks
                neg_clicks = epoch_neg_clicks[-1][k]  # Retrieve the negative clicks
                # Plot the real image with overlayed clicks
                plt.subplot(2, 10, k + 1)
                plt.imshow(real_image, cmap='gray')
                plt.title('Real')
                plt.axis('off')
                # Overlay positive clicks (in green) and negative clicks (in red)
                for click in pos_clicks:
                    _, x, y = click  # Extract only the x and y coordinates
                    plt.plot(y, x, 'g.', markersize=1)  # Green dots for positive clicks
                for click in neg_clicks:
                    _, x, y = click  # Extract only the x and y coordinates
                    plt.plot(y, x, 'r.', markersize=1)  # Red dots for negative clicks

                # Plot the predicted output below
                plt.subplot(2, 10, k + 11)
                plt.imshow(Y_hat[k, 0], cmap='gray')
                plt.title('Output')
                plt.axis('off')

            # Display title and save the figure
            plt.suptitle('%d / %d - loss: %f' % (epoch + 1, epochs, avg_loss))
            savename = f'{model.name()}_{epochs}e_{clicks}clicks'
            plt.savefig(f'Results/Project2_Part2/{savename}.png')
            plt.close()


def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [F.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(Y_pred)


def get_model(model_name):
    """
    Factory function to create model instances
    """
    models = {
        'EncDec': EncDec,
        'Unet': UNet,
        'Unet2': UNet2,
        'DilatedNet': DilatedNet
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name]()


def save_predicted_vs_true_masks(X_batch, Y_batch, Y_pred, num_samples=5, file_path="./"):
    # Move tensors to CPU and convert them to numpy arrays
    X_batch = X_batch.cpu().detach().numpy()
    Y_batch = Y_batch.cpu().detach().numpy()
    #Y_batch = np.rollaxis(Y_batch, 0, 3)
    Y_pred = (Y_pred > 0.5).cpu().detach().numpy()  # Apply threshold to get binary mask (adjust if needed)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(X_batch[i].transpose(1, 2, 0), cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # True mask
        axes[i, 1].imshow(Y_batch[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("True Mask")
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(Y_pred[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close(fig)
    print(f"Saved predicted vs. true masks to {file_path}")


if __name__ == '__main__':
    epochs = 20

    # choose between these losses:
    'bce_loss, dice, intersection_over_union, accuracy, sensitivity, specificity, focal_loss, bce_total_variation'
    loss = point_level_loss

    # choose between these models:
    'EncDec(), UNet(), UNet2(), DilatedNet()'
    

    models_to_test = ['EncDec', 'Unet','Unet2', 'DilatedNed', 'Umax'] 

    models_to_test = ['Unet'] 
    click_count = [1,2,4,6,8,10,12,14,16]
    click_count = [1,2]
    for m in models_to_test:
        results_csv = f"./Results/Project2_Part2/{m}.csv"
        with open(results_csv,mode='w',newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Click count','Loss', 'Dice overlap', 'Intersection over Union', 'Accuracy', 'Sensibility', 'Specificity'])
            for clicks in click_count:
                model = get_model(m)
                model = model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                train(model, optimizer, loss, epochs, train_loader, val_loader, clicks)

                total_eval_loss = 0
                total_dice = 0
                total_iou = 0
                total_acc = 0
                total_sensitivity = 0
                total_specificity = 0
                with torch.no_grad():
                    for X_batch, Y_batch in test_loader:
                        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                        
                        # Forward pass
                        Y_pred = model(X_batch)  # Use sigmoid if using BCE loss

                        # Compute mask-based evaluation loss
                        eval_loss = bce_loss2(Y_batch, Y_pred)
                        total_eval_loss += eval_loss

                        Y_pred = torch.sigmoid(Y_pred)
                        # Compute dice
                        dice_metric = dice(Y_batch,Y_pred)
                        total_dice += dice_metric
                        # Compute intersection over union
                        IoU = intersection_over_union(Y_batch,Y_pred)
                        total_iou += IoU
                        # Compute accuracy
                        acc = accuracy(Y_batch,Y_pred)
                        total_acc += acc
                        # Compute sensitivity
                        sens = sensitivity(Y_batch,Y_pred)
                        total_sensitivity += sens
                        # Compute specificity
                        spec = specificity(Y_batch,Y_pred)
                        total_specificity += spec


                        evaluation_img_result = f"./Results/Project2_Part2/{m}_{clicks}.png"
                        save_predicted_vs_true_masks(X_batch, Y_batch, Y_pred, file_path=evaluation_img_result)
                
                avg_dice = total_dice / len(test_loader)
                avg_iou = total_iou / len(test_loader)
                avg_acc = total_acc / len(test_loader)
                avg_sensitivity = total_sensitivity / len(test_loader)
                avg_specificity = total_specificity / len(test_loader)
                avg_eval_loss = total_eval_loss / len(test_loader)
                writer.writerow([clicks,avg_eval_loss.item(), avg_dice.item(), avg_iou.item(), avg_acc.item(), avg_sensitivity.item(), avg_specificity.item()])
                print(f"Evaluation Mask-Based Loss: {avg_eval_loss}")
                print(f"Average Dice Overlap: {avg_dice}")
                print(f"Average Intersection over Union: {avg_iou}")
                print(f"Average Accuracy: {avg_acc}")
                print(f"Average Sensitivity: {avg_sensitivity}")
                print(f"Average Specificity: {avg_specificity}")
