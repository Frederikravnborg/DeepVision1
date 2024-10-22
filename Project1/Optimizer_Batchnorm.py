import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#####################  CONFIGURATION  #####################

_optimizer = "SGD Momentum"  # "Adam", "SGD", "SGD Momentum"
_lr = 1e-2
_use_batchnorm = False

num_epochs = 10
size = 128
batch_size = 64

###########################################################


if torch.backends.mps.is_available():
    print("The code will run on MPS.")
    device = torch.device('mps')
elif torch.cuda.is_available():
    print("The code will run on GPU.")
    device = torch.device('cuda')
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cpu')



# Define the Hotdog_NotHotdog dataset class
class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='hotdog_nothotdog/'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path + '/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y

# Define the SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if _use_batchnorm else None
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if _use_batchnorm else None
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if _use_batchnorm else None
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.bn_fc1 = nn.BatchNorm1d(64) if _use_batchnorm else None
        
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        if _use_batchnorm:
            x = self.bn1(x)
        x = self.pool(F.relu(x))
        
        x = self.conv2(x)
        if _use_batchnorm:
            x = self.bn2(x)
        x = self.pool(F.relu(x))
        
        x = self.conv3(x)
        if _use_batchnorm:
            x = self.bn3(x)
        x = self.pool(F.relu(x))
        
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc1(x)
        if _use_batchnorm:
            x = self.bn_fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        return torch.sigmoid(x)


train_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

# Load the train dataset and split it into training and validation sets
trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

# Create DataLoaders for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

testset = Hotdog_NotHotdog(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.BCELoss()
if _optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=_lr)
elif _optimizer == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=_lr)
elif _optimizer == "SGD Momentum":
    optimizer = optim.SGD(model.parameters(), lr=_lr, momentum=0.9)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).float().squeeze() 
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Training loop
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for i, data in enumerate(progress_bar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track the loss
        running_loss += loss.item()
        progress_bar.set_postfix({'Loss': running_loss / (i+1)})

    # Calculate train and validation accuracy
    train_acc = calculate_accuracy(train_loader, model)
    val_acc = calculate_accuracy(val_loader, model)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
          f'Train Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%')

# Evaluate the model on the test set
test_accuracy = calculate_accuracy(test_loader, model)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Config: optimizer={_optimizer}, lr={_lr}, batchnorm={_use_batchnorm}')

# Plot train and validation accuracy
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Train vs Validation Accuracy')
plt.show()