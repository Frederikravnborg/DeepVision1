import os
import numpy as np
import glob
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from time import time
import matplotlib.pyplot as plt
from models import *
from losses import *
from dataloaders import *
data_path = '/dtu/datasets1/02516/phc_data'

# Custom Dataset class
class PhC(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path=data_path):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.jpg'))
        self.label_paths = sorted(glob.glob(data_path + '/labels/*.png'))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y

# Training function
def train(model, opt, loss_fn, epochs, train_loader, test_loader, size):
    X_test, Y_test = next(iter(test_loader))
    
    for epoch in range(epochs):
        print(f'* Epoch {epoch+1}/{epochs}')
        avg_loss = 0
        model.train()
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            opt.zero_grad()
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)
            loss.backward()
            opt.step()
            avg_loss += loss / len(train_loader)
        
        print(f' - loss: {avg_loss:.4f}')
        
        # Displaying intermediate results
        model.eval()
        Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')
            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle(f'{model.name()} - size: {size} - epoch: {epoch+1} - loss: {avg_loss:.4f}')
        plt.savefig(f'Results/{model.name()}_{size}_{epoch+1}.png')

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    test_accuracy = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X_batch)
            test_accuracy += accuracy(Y_batch, Y_pred).item()
    test_accuracy /= len(test_loader)
    return test_accuracy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment configurations
models = [UNet2()]
image_sizes = [64, 128, 256, 512]
epochs = 75  # Adjust as needed for testing

# Run experiments
if __name__ == '__main__':
    for model in models:
        model_name = model.__class__.__name__
        
        for size in image_sizes:
            # Update transformations for the current image size
            train_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

            # Create dataset loaders for the specified size
            trainset = PhC(train=True, transform=train_transform)
            testset = PhC(train=False, transform=test_transform)
            train_loader = DataLoader(trainset, batch_size=6, shuffle=True)
            test_loader = DataLoader(testset, batch_size=6, shuffle=False)

            # Reset model and optimizer
            model = model.to(device)
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = bce_loss2  # Replace with your actual loss function

            # Train and evaluate
            print(f"\nTraining {model_name} with image size {size}")
            train(model, optimizer, loss_fn, epochs, train_loader, test_loader, size)
            final_test_accuracy = evaluate_model(model, test_loader)
            print(f"Test accuracy for {model_name} with size {size}: {final_test_accuracy:.4f}\n")
