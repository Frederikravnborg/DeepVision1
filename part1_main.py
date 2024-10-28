import os
import numpy as np
import glob
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
import matplotlib.pyplot as plt
# from IPython.display import clear_output
from models import *
from losses import *
from dataloaders import *
data_path = '/dtu/datasets1/02516/phc_data'
# data_path = '/Users/fredmac/Documents/DTU-FredMac/Deep Vision/Poster 2/phc_data'

class PhC(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path=data_path):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.jpg'))
        self.label_paths = sorted(glob.glob(data_path + '/labels/*.png'))
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y


size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

batch_size = 6
trainset = PhC(train=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testset = PhC(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# %%
print('Loaded %d training images' % len(trainset))
print('Loaded %d test images' % len(testset))


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps')
print(device)



def train(model, opt, loss_fn, epochs, train_loader, test_loader):
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        tic = time()
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
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            
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
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
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
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        savename = f'{model.name()}, {loss_fn.__name__}, {epochs}e'
        plt.savefig(f'Results/{savename}.png')


def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [F.sigmoid(model(X_batch.to(device))) for X_batch, _ in data]
    return np.array(Y_pred)


def evaluate_model(model, test_loader):
    """Evaluate the model on the test set and return the accuracy."""
    model.eval()  # Set model to evaluation mode
    test_accuracy = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X_batch)
            test_accuracy += accuracy(Y_batch, Y_pred).item()

    # Average accuracy over the test set
    test_accuracy /= len(test_loader)
    return test_accuracy

# from torchviz import make_dot
# make_dot(model(torch.randn(20, 3, 256, 256)), params=dict(model.named_parameters()))


models = [EncDec(), UNet(), UNet2(), DilatedNet()]
pos_weights = [1.0, 2.0, calculate_pos_weight(train_loader), 10.0]  # Define the weights to test

if __name__ == '__main__':
    epochs = 75

    # Loop over each model
    for model in models:
        model_name = model.__class__.__name__
        model = model.to(device)

        # Loop over each weight for the weighted BCE loss
        for pos_weight in pos_weights:
            # Define the optimizer and reset the model
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

            # Define the loss function with the current weight
            def loss_fn(y_real, y_pred):
                return weighted_bce_loss(y_real, y_pred, pos_weight=pos_weight)
            
            print(f"\nTraining {model_name} with pos_weight={pos_weight}")

            # Train the model
            train(model, optimizer, loss_fn, epochs, train_loader, test_loader)

            # Get the final test accuracy
            final_test_accuracy = evaluate_model(model, test_loader)
            print(f"Test accuracy for {model_name} with pos_weight={pos_weight}: {final_test_accuracy:.4f}\n")