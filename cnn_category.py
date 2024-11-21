import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the directory where images are stored
image_dir = '/Users/giodano/Desktop/College/2024_Fall/CSC_871/project/images/'

# Load CSV metadata
csv_file = 'dataset.csv'
data = pd.read_csv(csv_file)

# Assuming your CSV has 'img_name' and 'category' columns
image_paths = data['img_name'].tolist()
labels = data['category'].tolist()

# Encode string labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save the mapping for decoding predictions later
label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label Mapping:", label_mapping)

# Define a custom PyTorch Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, image_dir, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure RGB format

        # Load label
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# Split into train, validation, and test sets
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, encoded_labels, test_size=0.3, stratify=encoded_labels, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

#hyperparameters
batch_size = 4
num_epochs = 100
learning_rate = 0.001

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Create datasets
train_dataset = ImageDataset(train_paths, train_labels, image_dir, transform=transform)
val_dataset = ImageDataset(val_paths, val_labels, image_dir, transform=transform)
test_dataset = ImageDataset(test_paths, test_labels, image_dir, transform=transform)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

labels = ('sports', 'occupation', 'water activities', 'home activities',
 'lawn and garden', 'miscellaneous', 'religious activities',
 'winter activities', 'conditioning exercise', 'bicycling',
 'fishing and hunting', 'walking', 'running', 'self care', 'music playing',
 'home repair', 'transportation', 'dancing', 'inactivity quiet/light',
 'volunteer activities', 'nan')



# Define CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 224, 224)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.num_classes = num_classes

        # Automatically initialize fully connected layers
        self._initialize_fc_layers(input_shape)

    def _compute_flattened_size(self, x):
        """Helper to compute flattened size for dynamically sized input."""
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.conv3(x)))
        return x.view(x.size(0), -1).size(1)

    def _initialize_fc_layers(self, input_shape):
        """Initialize fully connected layers dynamically."""
        dummy_input = torch.zeros(1, *input_shape)
        self.flattened_size = self._compute_flattened_size(dummy_input)
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



model = CNN(len(set(labels))).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_progress = tqdm(train_loader, desc=f"Training - Epoch {epoch+1}/{num_epochs}")

    # Gradient accumulation setup
    accumulation_steps = 4
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item()

    # Learning rate scheduling
    scheduler.step()

    # Print epoch summary
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {running_loss/len(train_loader):.4f}")
