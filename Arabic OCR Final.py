import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

#%% Define dataset class to read, preprocess, and label images
class ArabicPNGDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(label_file, encoding='utf-16')
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.labels.loc[index, 'Image'])
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        label = self.labels.loc[index, 'Label']
        return img, int(label)


#%% Define custom model
# Define Gradient Reversal Layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.alpha
        return grad_input, None

# Modify the CustomModel
# Define a custom model with some convolutional layers, pooling layers, and fully connected layers
class CustomModel(nn.Module):
    def __init__(self, num_classes=14):
        super(CustomModel, self).__init__()

        # Define the feature extraction layers using convolutional and pooling operations
        self.features = nn.Sequential(
            # Convolutional Layer 1: Input channels=1, output channels=32, kernel size=3x3, stride=1, padding=1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # Apply ReLU activation function element-wise in-place
            nn.MaxPool2d(kernel_size=2, stride=2),  # Perform max pooling with a 2x2 kernel and stride of 2

            # Convolutional Layer 2: Input channels=32, output channels=64, kernel size=3x3, stride=1, padding=1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Convolutional Layer 3: Input channels=64, output channels=128, kernel size=3x3, stride=1, padding=1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive average pooling layer to convert any input size to (16, 16)
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))

        # Classifier: Fully connected layers for diacritic recognition
        self.classifier = nn.Sequential(
            nn.Dropout(),  # Apply dropout regularization to prevent overfitting
            nn.Linear(128 * 16 * 16, 512),  # Fully connected layer with input size 128x16x16 and output size 512
            nn.ReLU(inplace=True),
            nn.Dropout(),  # Apply dropout regularization again
            nn.Linear(512, num_classes),  # Final fully connected layer with output size equal to the number of classes
        )

        # Domain classifier: Another fully connected layer for domain adaptation (optional)
        self.domain_classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 100),  # Fully connected layer with input size 128x16x16 and output size 100
            nn.ReLU(),  # Apply ReLU activation function element-wise
            nn.Linear(100, 1),  # Final fully connected layer with output size 1 (binary classification)
        )

    def forward(self, x, alpha=None):
        # Perform forward pass through the feature extraction layers
        x = self.features(x)

        # Apply adaptive average pooling to convert any input size to (16, 16)
        x = self.avgpool(x)

        # Flatten the output from the feature extraction layers to prepare for the fully connected layers
        x = torch.flatten(x, 1)

        # Classifier for diacritic recognition: Perform forward pass through the classifier layers
        class_output = self.classifier(x)

        # Domain classifier (adaptation): Perform forward pass through the domain classifier layers (if alpha is provided)
        if alpha is not None:
            # Apply a gradient reversal operation on the feature representation (used in domain adaptation)
            reverse_feature = GradReverse.apply(x, alpha)

            # Perform forward pass through the domain classifier layers
            domain_output = self.domain_classifier(reverse_feature)

            # Return both the class_output and domain_output if alpha is provided (used in domain adaptation)
            return class_output, domain_output
        else:
            # Return only the class_output if alpha is not provided (regular diacritic recognition)
            return class_output



#%% Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Set paths and parameters
image_dir = "Synthetic Dataset/images"
label_file = "Synthetic Dataset/labels/labels.csv"
batch_size = 16
num_epochs = 20
learning_rate = 0.001
image_size = 128

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# Create dataset
dataset = ArabicPNGDataset(image_dir, label_file, transform=transform)

# Split dataset into train and test sets FOR ALL FONTS EVENLY
# train_set, test_set = train_test_split(dataset, test_size=0.33, random_state=0)

# Split dataset into train and test sets SPECIFY FONTS
# train_indices = []
# test_indices = []
# for i in range(len(dataset)):
#     filename = dataset.labels.loc[i, 'Image']
#     if filename.startswith("Font2_") or filename.startswith("Font3_"):
#         test_indices.append(i)
#     else:
#         train_indices.append(i)

# Get unique font numbers from the dataset
font_numbers = set()
for i in range(len(dataset)):
    filename = dataset.labels.loc[i, 'Image']
    font_number = int(filename.split('_')[0].replace('Font', ''))
    font_numbers.add(font_number)

# Randomly select 15 fonts from 2 to 48
test_fonts_range = random.sample(range(2, 49), 15)

# Randomly select 15 fonts from other fonts excluding the range 2 to 48
test_fonts_other = random.sample(font_numbers - set(range(2, 49)), 15)

# Combine the selected fonts for testing
test_fonts = test_fonts_range + test_fonts_other

# Split dataset into train and test sets
train_indices = []
test_indices = []
for i in range(len(dataset)):
    filename = dataset.labels.loc[i, 'Image']
    font_number = int(filename.split('_')[0].replace('Font', ''))
    if font_number in test_fonts:
        test_indices.append(i)
    else:
        train_indices.append(i)

train_set = torch.utils.data.Subset(dataset, train_indices)
test_set = torch.utils.data.Subset(dataset, test_indices)

# Count the number of unique fonts in the testing dataset
unique_test_fonts = set()
for index in test_indices:
    filename = dataset.labels.loc[index, 'Image']
    font_number = int(filename.split('_')[0].replace('Font', ''))
    unique_test_fonts.add(font_number)

# Print the number of unique fonts in the testing dataset
print("Number of unique fonts in the testing dataset:", len(unique_test_fonts))
print(unique_test_fonts)

# Count the number of unique fonts in the training dataset
unique_train_fonts = set()
for index in train_indices:
    filename = dataset.labels.loc[index, 'Image']
    font_number = int(filename.split('_')[0].replace('Font', ''))
    unique_train_fonts.add(font_number)

# Print the number of unique fonts in the training dataset
print("Number of unique fonts in the training dataset:", len(unique_train_fonts))
print(unique_train_fonts)

# Print filenames in the testing dataset FILENAMES **************************************************
# print("Filenames in the testing dataset:")
# for i in test_indices:
#     filename = dataset.labels.loc[i, 'Image']
#     print(filename)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Create the model
model = CustomModel().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create two empty lists to store accuracy and loss values
train_accuracy_list = []
test_accuracy_list = []
train_loss_list = []
test_loss_list = []

#%% Training loop
# Training loop with domain adaptation
lambda_domain = 0.1
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass with domain adaptation (alpha > 0)
        class_output, domain_output = model(images, alpha=0.1)

        # Calculate classification loss and domain adaptation loss
        class_loss = criterion(class_output, labels)
        domain_labels = torch.zeros(len(images)).long().to(device)  # Assuming source domain is labeled as 0
        domain_loss = criterion(domain_output, domain_labels)

        # Total loss with a trade-off parameter lambda_domain
        total_loss = class_loss + lambda_domain * domain_loss

        total_loss.backward()
        optimizer.step()

        # Update training accuracy and loss
        train_loss += total_loss.item()
        _, predicted = torch.max(class_output, 1)
        train_correct += (predicted == labels).sum().item()

    # Calculate average training loss and accuracy for this epoch
    train_loss /= len(train_loader)
    train_accuracy = train_correct / len(train_loader.dataset)

    # Validation and testing (no domain adaptation, alpha = None)
    model.eval()
    test_loss = 0.0
    test_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass without domain adaptation
            class_output = model(images, alpha=None)

            # Calculate validation or test loss and accuracy
            loss = criterion(class_output, labels)
            test_loss += loss.item()
            _, predicted = torch.max(class_output, 1)
            test_correct += (predicted == labels).sum().item()

    # Calculate average validation or test loss and accuracy for this epoch
    test_loss /= len(test_loader)
    test_accuracy = test_correct / len(test_loader.dataset)

    # Update the lists for plotting
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

    # Print training and validation/test metrics for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print()

#%% Save the trained model
torch.save(model.state_dict(), "arabic_diacritic_recognition_model.pth")

#%% Evaluation on test set
# Final evaluation on the test set
model.eval()
true_labels = []
predicted_labels = []
test_loss = 0.0
test_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass without domain adaptation
        class_output = model(images, alpha=None)

        # Calculate test loss and accuracy
        loss = criterion(class_output, labels)
        test_loss += loss.item()
        _, predicted = torch.max(class_output, 1)
        test_correct += (predicted == labels).sum().item()

        # Store true and predicted labels for later use
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Calculate average test loss and accuracy
test_loss /= len(test_loader)
test_accuracy = test_correct / len(test_loader.dataset)

# Print final evaluation results
print("Final Evaluation on Test Set:")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

#%% Generate classification report
class_names = ['Fatha', 'Damma', 'Kasra', 'Fathatan', 'Dammatan', 'Kasratan', 'Shadda', 'Sukun', 'Shadda with Alef', 'Shadda with Fatha', 'Shadda with Damma', 'Shadda with Kasra', 'Shadda with Dammatan', 'Shadda with Kasratan']
classification_rep = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Classification Report:")
print(classification_rep)

#%% Generate confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

#%%
# Plot accuracy
epochs = range(1, num_epochs+1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy_list, 'b-', label='Train Accuracy')
plt.plot(epochs, test_accuracy_list, 'r-', label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_list, 'b-', label='Train Loss')
plt.plot(epochs, test_loss_list, 'r-', label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

