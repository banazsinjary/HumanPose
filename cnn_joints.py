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
proj_dir = r"C:\Users\Charl\Documents\College\2 - CSC 871\project\mpii_human_pose\\"
img_dir = proj_dir + "images"

# load label data
image_labels = pd.read_csv(proj_dir + "labels.csv", index_col=0)

class JointDataset(Dataset):
    def __init__(self, files, scales, images, loc_and_scale, joint_visibility, joint_positions):
        self.files = files
        self.scales = scales
        self.images = images
        self.loc_and_scale = loc_and_scale
        self.joint_visibility = joint_visibility
        self.joint_positions = joint_positions  # Head (x, y) positions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file = self.files[idx]
        scale = self.scales[idx]
        image = self.images[idx]
        loc_and_scale = self.loc_and_scale[idx]
        joint_vis = self.joint_visibility[idx]
        joint_position = self.joint_positions[idx]  # (head_x, head_y)
        return file, scale, image, loc_and_scale, joint_vis, joint_position

# Define transformations
H = 360; W = 640
transform = transforms.Compose([
    transforms.Resize((H, W)),  # Resize to target size
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    
    ##### TODO --> find mean and sd for image layers
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

class FeatureExtraction(nn.Module):
    def __init__(self, input_channels, dropout_rate=.5):
        super(FeatureExtraction, self).__init__()
        
        # input_shape: height, width, and # of channels of images
        # num_joints: number of joints to be iden

        # Shared Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        #self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Dropout after convolutional layers
        self.dropout_conv1 = nn.Dropout2d(dropout_rate)
        self.dropout_conv2 = nn.Dropout2d(dropout_rate)
        self.dropout_conv3 = nn.Dropout2d(dropout_rate)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        # Shared Convolutional Layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_conv1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout_conv3(x)
        
        #x = self.pool(F.relu(self.conv4(x)))
        
        # Global Average Pooling to reduce spatial dimensions
        x = self.global_avg_pool(x)
        #x = torch.flatten(x, 1)  # Flatten to (batch_size, 128)
        x = x.view(x.size(0), -1)

        return x
    
# CNN model for head position detection
class JointDetectionCNN(nn.Module):
    def __init__(self, input_channels=3, dropout_rate=.5):
        super(JointDetectionCNN, self).__init__()
        
        self.cnn = FeatureExtraction(input_channels)
        
        self.fc1 = nn.Linear(256 + 3, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Dropout after fully connected layers
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        
        self.joint_coordinates = nn.Linear(64, 32)
        
        #self.fc_input_size = H*W

    def forward(self, image, loc_and_scale):
        
        image_features = self.cnn(image)
        
        combined_features = torch.cat((image_features, loc_and_scale), dim=1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        
        # Separate outputs for coordinates and visibility
        joint_coords = self.joint_coordinates(x)
        
        return joint_coords

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path=proj_dir + 'checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)

# Load images
def load_images(folder_path, file_list):
    images = []
    incl_files = []
    sizes = []
    for filename in file_list: 
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(img_path).convert('RGB') 
                size = image.size
            except FileNotFoundError:
                print(filename)
                continue
            
            image = transform(image)  
            images.append(image)
            incl_files = incl_files + [filename]
            sizes = sizes + [[filename, *size]]
    return torch.stack(images), incl_files, sizes  

# create dataset from labels, images and filenames
def getDataset(labels, images, files):
    label_set = labels[labels['img_name'].isin(files)] 
    scales = torch.tensor(labels[['image_size_w','image_size_h']].values, dtype=torch.float32)
    loc_and_scale = torch.tensor(label_set[['objpos_x', 'objpos_y', 'scale']].values, dtype=torch.float32)
    joint_vis = torch.tensor(label_set[['rankl', 'rknee', 'rhip', 'lhip', 'lknee', 'lankl', 
                                        'pelvis', 'thorax', 'upper_neck', 'head', 'rwri', 'relb',
                                        'rsho', 'lsho', 'lelb', 'lwri']].values, dtype=torch.float32)
    joint_locs = torch.tensor(label_set[['rankl_x', 'rankl_y', 'rknee_x', 'rknee_y', 'rhip_x', 'rhip_y',
                                         'lhip_x', 'lhip_y', 'lknee_x', 'lknee_y', 'lankl_x', 'lankl_y',
                                         'pelvis_x', 'pelvis_y', 'thorax_x', 'thorax_y', 'upper_neck_x', 
                                         'upper_neck_y', 'head_x', 'head_y', 'rwri_x', 'rwri_y',
                                         'relb_x', 'relb_y', 'rsho_x', 'rsho_y', 'lsho_x', 'lsho_y',
                                         'lelb_x', 'lelb_y', 'lwri_x', 'lwri_y']].values, dtype=torch.float32)
    dataset = JointDataset(files=files,
                           scales=scales,
                           images=images, 
                           loc_and_scale=loc_and_scale, 
                           joint_visibility=joint_vis,
                           joint_positions=joint_locs)
    return(dataset)

### Load data and create DataLoaders

##### TODO -->
# load data in parts to conserve memory

# get list of filenames
img_filenames = image_labels['img_name'].values

# load images (780 sec for all, 118 sec for 3000 images)
start = time.time()
image_data, incl_files, image_sizes = load_images(img_dir, img_filenames) # too much, need to do in parts
end = time.time()
end - start


# scale labels by image size
# merge image sizes with labels
image_sizes_df = pd.DataFrame(image_sizes, columns=['img_name', 'image_size_w', 'image_size_h'])
scaled_labels = pd.merge(image_sizes_df, image_labels, on='img_name', how='left')

# scale x values
scale_x = 1 / scaled_labels['image_size_w']
# select fields that contain _x and multiply by scale_x
x_cols = scaled_labels.filter(like='_x')
for col in x_cols.columns: scaled_labels[col] = scaled_labels[col] * scale_x

# scale y values
scale_y = 1 / scaled_labels['image_size_h']
# select fields that contain _y and multiply by scale_y
y_cols = scaled_labels.filter(like='_y')
for col in y_cols.columns: scaled_labels[col] = scaled_labels[col] * scale_y


# split train/validation/test (360 sec for all, 14.3 sec for 3000 images)
start = time.time()
image_train, image_test, files_train, files_test = train_test_split(
    image_data, incl_files, test_size=0.2, random_state=1)
image_train, image_val, files_train, files_val = train_test_split(
    image_train, files_train, test_size=0.25, random_state=1)
end = time.time()
end - start


# load training data
train_dataset = getDataset(scaled_labels, image_train, files_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# load validation data
val_dataset = getDataset(scaled_labels, image_val, files_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# load test data
test_dataset = getDataset(scaled_labels, image_test, files_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

###

# Initialize model, loss, and optimizer
model = JointDetectionCNN()
early_stopping = EarlyStopping(patience=5, verbose=True)
criterion = nn.MSELoss(reduction='none')  # Mean Squared Error Loss for coordinate prediction
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop (5 epochs -> 8796 sec for all, sec for 3000 images)
epochs = 100
start = time.time()
for epoch in range(epochs):
    
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    running_val_loss = 0.0
        
    iter_count = 0
    for image_names, scales, images, locs_and_scales, joint_visibility, joint_positions in train_dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, locs_and_scales)
        
        # mask outputs
        visibility_mask = joint_visibility.repeat_interleave(2, dim=1)
        
        # calculate loss with mask and take average (sum / number of visibile joint coords)
        loss = criterion(outputs, joint_positions)
        masked_loss = loss * visibility_mask
        actual_loss = masked_loss.sum() / visibility_mask.sum()
        
        # Backward pass and optimization
        actual_loss.backward()
        optimizer.step()
        
        # keep tally of running loss
        running_loss += actual_loss.item()
        
        # regular updates on progress
        iter_count += 1
        if iter_count % 10 == 0: print(f"{iter_count}/{len(train_dataloader)} completed")

    train_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] Training complete ({train_time-epoch_start:.1f} seconds) Loss: {running_loss/len(train_dataloader):.2f}")
    
    for image_names, scales, images, locs_and_scales, joint_visibility, joint_positions in val_dataloader:
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = model(images, locs_and_scales)
        
        # mask outputs
        val_visibility_mask = joint_visibility.repeat_interleave(2, dim=1)

        # calculate loss with mask and take average (sum / number of visibile joint coords)
        val_loss = criterion(outputs, joint_positions)
        val_masked_loss = val_loss * val_visibility_mask
        val_actual_loss = val_masked_loss.sum() / val_visibility_mask.sum()
        
        # keep tally of running loss
        running_val_loss += val_actual_loss.item()

    val_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] Validation complete ({val_time-train_time:.1f} seconds) Loss: {running_val_loss/len(val_dataloader):.2f}")
    
    early_stopping(running_val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Exiting training loop.")
        break
   
end = time.time()
print(f"Training completed ({end - start:.1f} seconds)")

#model.load_state_dict(torch.load(proj_dir + 'checkpoint.pth'))

joint_loc_preds_df = pd.DataFrame(columns=['filename'])
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    running_val_loss = 0.0
    
    batch_count = 0
    for image_names, scales, images, locs_and_scales, joint_visibility, joint_positions in val_dataloader:
        
        joint_loc_preds = model(images, locs_and_scales)
        joint_loc_preds_df = pd.concat([joint_loc_preds_df, 
                                        pd.DataFrame({'filename': list(image_names),
                                                      'scale_image_w': scales[:, 0].tolist(),
                                                      'scale_image_h': scales[:, 1].tolist(),
                                                      'rankl_x': joint_loc_preds[:, 0].tolist(), 
                                                      'rankl_y': joint_loc_preds[:, 1].tolist(),
                                                      'rknee_x': joint_loc_preds[:, 2].tolist(), 
                                                      'rknee_y': joint_loc_preds[:, 3].tolist(),
                                                      'rhip_x': joint_loc_preds[:, 4].tolist(), 
                                                      'rhip_y': joint_loc_preds[:, 5].tolist(),
                                                      'lhip_x': joint_loc_preds[:, 6].tolist(), 
                                                      'lhip_y': joint_loc_preds[:, 7].tolist(),
                                                      'lknee_x': joint_loc_preds[:, 8].tolist(), 
                                                      'lknee_y': joint_loc_preds[:, 9].tolist(),
                                                      'lankl_x': joint_loc_preds[:, 10].tolist(), 
                                                      'lankl_y': joint_loc_preds[:, 11].tolist(),
                                                      'pelvis_x': joint_loc_preds[:, 12].tolist(), 
                                                      'pelvis_y': joint_loc_preds[:, 13].tolist(),
                                                      'thorax_x': joint_loc_preds[:, 14].tolist(), 
                                                      'thorax_y': joint_loc_preds[:, 15].tolist(),
                                                      'upper_neck_x': joint_loc_preds[:, 16].tolist(),
                                                      'upper_neck_y': joint_loc_preds[:, 17].tolist(),
                                                      'head_top_x': joint_loc_preds[:, 18].tolist(), 
                                                      'head_top_y': joint_loc_preds[:, 19].tolist(),
                                                      'rwri_x': joint_loc_preds[:, 20].tolist(), 
                                                      'rwri_y': joint_loc_preds[:, 21].tolist(),
                                                      'relb_x': joint_loc_preds[:, 22].tolist(), 
                                                      'relb_y': joint_loc_preds[:, 23].tolist(),
                                                      'rsho_x': joint_loc_preds[:, 24].tolist(), 
                                                      'rsho_y': joint_loc_preds[:, 25].tolist(),
                                                      'lsho_x': joint_loc_preds[:, 26].tolist(), 
                                                      'lsho_y': joint_loc_preds[:, 27].tolist(),
                                                      'lelb_x': joint_loc_preds[:, 28].tolist(), 
                                                      'lelb_y': joint_loc_preds[:, 29].tolist(),
                                                      'lwri_x': joint_loc_preds[:, 30].tolist(), 
                                                      'lwri_y': joint_loc_preds[:, 31].tolist()})])
        # mask outputs
        #val_visibility_mask = torch.cat((joint_visibility, joint_visibility), dim=1)
        
        # calculate loss with mask and take average (sum / number of visibile joint coords)
        #val_loss = criterion(joint_loc_preds, joint_positions)
        #val_masked_loss = val_loss * val_visibility_mask
        #val_actual_loss = val_masked_loss.sum() / val_visibility_mask.sum()
        
        #running_val_loss += val_actual_loss.item()
        batch_count += 1
        if batch_count > 0: break

    #print(f"Val Loss: {running_val_loss/len(val_dataloader):.2f}")


##### TODO -->
# create a set of test outputs and visualize
joint_loc_preds_df.to_csv(proj_dir + 'joint_val_preds_df.csv', index=False)

# save model
torch.save(model, proj_dir + 'jt_model.pth') 
torch.save(model.state_dict(), proj_dir + 'jt_model_state_dict.pth')

# load model
model = torch.load(proj_dir + "jt_model.pth")
