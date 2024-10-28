import os
import numpy as np
import glob
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from models import *
from losses_metrics import *
from project2DataLoaders import ph2_loaders, drive_loaders


def compute_metrics(model, data_loader):
    model.eval()
    total_dice = 0
    total_iou = 0
    total_accuracy = 0
    total_sensitivity = 0
    total_specificity = 0
    num_batches = 0

    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            Y_pred = model(X_batch)
            Y_pred = torch.sigmoid(Y_pred)
            
            # Compute metrics using your functions
            batch_dice = 1 - dice(Y_batch, Y_pred)  # Adjusted since your dice function returns 1 - Dice coefficient
            batch_iou = intersection_over_union(Y_batch, Y_pred)
            batch_accuracy = accuracy(Y_batch, Y_pred)
            batch_sensitivity = sensitivity(Y_batch, Y_pred)
            batch_specificity = specificity(Y_batch, Y_pred)

            total_dice += batch_dice.item()
            total_iou += batch_iou.item()
            total_accuracy += batch_accuracy.item()
            total_sensitivity += batch_sensitivity.item()
            total_specificity += batch_specificity.item()
            num_batches += 1

    # Compute averages
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_sensitivity = total_sensitivity / num_batches
    avg_specificity = total_specificity / num_batches

    metrics = {
        'Dice Coefficient': avg_dice,
        'IoU': avg_iou,
        'Accuracy': avg_accuracy,
        'Sensitivity': avg_sensitivity,
        'Specificity': avg_specificity
    }
    return metrics

def print_metrics(metrics, dataset_name):
    print(f"\n{dataset_name} set metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

def train(model, opt, loss_fn, epochs, train_loader, val_loader, test_loader):
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)

            if torch.isnan(Y_pred).any():
                print('Y_pred is nan')
                break
            loss = loss_fn(Y_pred, Y_batch)  # forward-pass
            
            if torch.isnan(loss):   
                print('Loss is nan')
                break

            if torch.isinf(loss):
                print('Loss is inf')
                break
            loss.backward()  # backward-pass

            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss.item() / len(train_loader)
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = torch.sigmoid(model(X_test.to(device))).detach().cpu()
        batch_size = X_test.size(0)
        for k in range(batch_size):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        savename = f'{model.name()}, {epochs}e'
        plt.savefig(f'Results/{savename}.png')
        plt.close()

        # Evaluate metrics on training and validation sets during the last epoch
        if epoch == epochs - 1:
            train_metrics = compute_metrics(model, train_loader)
            print_metrics(train_metrics, "Training")
            val_metrics = compute_metrics(model, val_loader)
            print_metrics(val_metrics, "Validation")

    # Save the model after the last epoch
    model_save_name = f'Models/{model.name()}_{epochs}e.pth'
    torch.save(model.state_dict(), model_save_name)
    print(f'Model saved at {model_save_name}')

def evaluate(model, data_loader, dataset_name="Test"):
    metrics = compute_metrics(model, data_loader)
    print_metrics(metrics, dataset_name)

def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [torch.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(Y_pred)


if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    epochs = 75

    # choose between these losses:
    ' bce_loss, focal_loss, nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device)) '
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))

    # choose between these models:
    'EncDec(), UNet(), UNet2(), DilatedNet()'
    model = UNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader, val_loader, test_loader = ph2_loaders()
    # train_loader, val_loader, test_loader = drive_loaders()
    train(model, optimizer, loss, epochs, train_loader, val_loader, test_loader)
    evaluate(model, test_loader, "Test")
