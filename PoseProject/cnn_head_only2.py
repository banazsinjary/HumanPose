# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:54:06 2024

@author: Charl
"""

import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# set project and image directories
proj_dir = r"C:\Users\banazsinjary\Desktop\871\PoseProject\mpii_human_pose\\"
img_dir = proj_dir + "images"

# load label data
image_labels = pd.read_csv(proj_dir + "labels.csv", index_col=0)


class HeadDataset(Dataset):
    def __init__(self, images, loc_and_scale, head_positions):
        self.images = images
        self.loc_and_scale = loc_and_scale
        self.head_positions = head_positions  # Head (x, y) positions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        loc_and_scale = self.loc_and_scale[idx]
        head_position = self.head_positions[idx]  # (head_x, head_y)
        return image, loc_and_scale, head_position


# Define transformations
H = 360; W = 640
transform = transforms.Compose([
    transforms.Resize((H, W)),  # Resize to target size
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

class FeatureExtraction(nn.Module):
    def __init__(self, input_channels):
        super(FeatureExtraction, self).__init__()
        
        # input_shape: height, width, and # of channels of images
        # num_joints: number of joints to be iden

        # Shared Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
    
    
    def forward(self, x):
        # Shared Convolutional Layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Global Average Pooling to reduce spatial dimensions
        x = self.global_avg_pool(x)
        #x = torch.flatten(x, 1)  # Flatten to (batch_size, 128)
        x = x.view(x.size(0), -1)

        return x
    
# CNN model for head position detection
class HeadDetectionCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(HeadDetectionCNN, self).__init__()
        
        self.cnn = FeatureExtraction(input_channels)
        
        self.fc1 = nn.Linear(128 + 3, 64)
        self.fc2 = nn.Linear(64,32)
        
        self.joint_coordinates = nn.Linear(32, 2)
        
        #self.fc_input_size = H*W

    def forward(self, image, loc_and_scale):
        
        image_features = self.cnn(image)
        
        combined_features = torch.cat((image_features, loc_and_scale), dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        
        # Separate outputs for coordinates and visibility
        joint_coords = self.joint_coordinates(x)
        
        return joint_coords

# Load images
def load_images(folder_path, file_list):
    images = []
    incl_files = []
    for filename in file_list: 
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(img_path).convert('RGB') 
            except FileNotFoundError:
                print(filename)
                continue
            image = transform(image)  
            images.append(image)
            incl_files = incl_files + [filename]
    return torch.stack(images), incl_files  


# create dataset from labels, images and filenames
def getDataset(labels, images, files):
    label_set = labels[labels['img_name'].isin(files)] 
    loc_and_scale = torch.tensor(label_set[['objpos_x', 'objpos_y', 'scale']].values, dtype=torch.float32)
    head_loc = torch.tensor(label_set[['head_x', 'head_y']].values, dtype=torch.float32)

    dataset = HeadDataset(images=images, 
                          loc_and_scale=loc_and_scale, 
                          head_positions=head_loc)
    return(dataset)

### Load data and create DataLoaders

##### TODO -->
# load data in parts to conserve memory

# get list of filenames
img_filenames = image_labels['img_name'].values

# load images (780 sec)
start = time.time()
image_data, incl_files = load_images(img_dir, img_filenames) # too much, need to do in parts
end = time.time()
end - start

# split train/validation/test (360 sec)
start = time.time()
image_train, image_test, files_train, files_test = train_test_split(
    image_data, incl_files, test_size=0.2, random_state=1)
image_train, image_val, files_train, files_val = train_test_split(
    image_train, files_train, test_size=0.25, random_state=1)
end = time.time()
end - start


# load training data
train_dataset = getDataset(image_labels, image_train, files_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# load validation data
val_dataset = getDataset(image_labels, image_val, files_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# load test data
test_dataset = getDataset(image_labels, image_test, files_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

###

# Initialize model, loss, and optimizer
model = HeadDetectionCNN()
criterion = nn.MSELoss()  # Mean Squared Error Loss for coordinate prediction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (5 epochs -> 8796 sec)
epochs = 5
start = time.time()
for epoch in range(epochs):
    
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    running_val_loss = 0.0
        
    for images, locs_and_scales, head_positions in train_dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, locs_and_scales)
        loss = criterion(outputs, head_positions)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    train_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] ({train_time-epoch_start}), Training Loss: {running_loss/len(train_dataloader):.2f}")
    
    for images, locs_and_scales, head_positions in val_dataloader:
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = model(images, locs_and_scales)
        
        val_loss = criterion(outputs, head_positions)
        
        running_val_loss += val_loss.item()

    val_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] ({val_time-train_time}), Validation Loss: {running_val_loss/len(val_dataloader):.2f}")
    
end = time.time()
print(f"Training completed ({end - start})")


model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    running_test_loss = 0.0
        
    for images, locs_and_scales, head_positions in test_dataloader:
        
        joint_loc_preds = model(images, locs_and_scales)
        test_loss = criterion(joint_loc_preds, head_positions)
        
        running_test_loss += test_loss.item()

    print(f"Test Loss: {running_test_loss/len(test_dataloader):.2f}")


##### TODO -->
# create a set of test outputs and visualize


# save model
torch.save(model, proj_dir + 'model.pth') 
torch.save(model.state_dict(), proj_dir + 'model_state_dict.pth')
