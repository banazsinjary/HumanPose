# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:54:06 2024

@author: Charl
"""

import pandas as pd
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import sys
import ast

code_dir = r"C:\Users\ckite\OneDrive\Documents\GitHub\HumanPose\\"
sys.path.append(code_dir)

import data_loading as dl
import model as cnn
import cnn_category as cat_cnn

# run on gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set project and image directories
proj_dir = r"C:\Users\ckite\Documents\Project\mpii_human_pose\\"
img_dir = proj_dir + "images"

# load label data
image_labels = pd.read_csv(proj_dir + "labels.csv", index_col=0)
image_labels['activity'] = image_labels['activity'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Define transformations
H = 480; W = 720
transform = transforms.Compose([
    transforms.Resize((H, W)),  # Resize to target size
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ##### TODO --> find mean and sd for image layers
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])


### Load data and create DataLoaders

set(image_labels['category'])
# get list of filenames
img_filenames = image_labels['img_name'][image_labels['category'].isin(["dancing", "bicycling", "running"])].values
# [image_labels['category'].isin(["dancing", "bicycling", "fishing and hunting"])]
# bad -> "fishing and hunting", "sports", "water activities", "winter activities"

# create encoders
category_encoder = LabelEncoder()
activity_encoder = MultiLabelBinarizer()

# fit encoders
category_encoder.fit(image_labels['category'][image_labels['img_name'].isin(img_filenames)])
activity_encoder.fit(image_labels['activity'][image_labels['img_name'].isin(img_filenames)])
category_encoder.classes_
activity_encoder.classes_

# encode categories and activities
enc_cat = category_encoder.transform(image_labels['category'][image_labels['img_name'].isin(img_filenames)])
enc_act = activity_encoder.transform(image_labels['activity'][image_labels['img_name'].isin(img_filenames)])

# split train/validation/test
files_train, files_test = train_test_split(img_filenames, test_size=0.2, random_state=1)
files_train, files_val = train_test_split(
    files_train, test_size=0.25, random_state=1)

##### TODO -->
# save files train and files test

# Get training labels
label_train = image_labels[image_labels['img_name'].isin(files_train)]
cat_train = torch.tensor(enc_cat[image_labels['img_name'][image_labels['img_name'].isin(img_filenames)].isin(files_train)], dtype=torch.long)
act_train = torch.tensor(enc_act[image_labels['img_name'][image_labels['img_name'].isin(img_filenames)].isin(files_train)], dtype=torch.float32)

### load training images ()
start = time.time()
images, labels, filenames, sizes = dl.load_images_in_chunks(img_dir, label_train, transform, 500)
time.time() - start

# update training labels
size_train = pd.DataFrame(sizes, columns=['img_name', 'image_size_w', 'image_size_h'])
label_train = pd.DataFrame()
for i in range(len(labels)):
    label_train = pd.concat([label_train, pd.DataFrame(labels[i])])
label_train = pd.merge(label_train, size_train)

# crate dataset and dataloader
train_dataset = dl.getCatDataset(label_train, images, filenames, cat_train, act_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


### load validation images
label_val = image_labels[image_labels['img_name'].isin(files_val)]
cat_val = torch.tensor(enc_cat[image_labels['img_name'][image_labels['img_name'].isin(img_filenames)].isin(files_val)], dtype=torch.long)
act_val = torch.tensor(enc_act[image_labels['img_name'][image_labels['img_name'].isin(img_filenames)].isin(files_val)], dtype=torch.float32)

start = time.time()
images, labels, filenames, sizes = dl.load_images_in_chunks(img_dir, label_val, transform, 500)
time.time() - start

# Get validation labels
size_val = pd.DataFrame(sizes, columns=['img_name', 'image_size_w', 'image_size_h'])
label_val = pd.DataFrame()
for i in range(len(labels)):
    label_val = pd.concat([label_val, pd.DataFrame(labels[i])])
label_val = pd.merge(label_val, size_val)

# load validation data
val_dataset = dl.getCatDataset(label_val, images, filenames, cat_val, act_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# early stopping and loss criteria
early_stopping = cnn.EarlyStopping(patience=5, verbose=True, delta=-0.0, path=proj_dir + 'checkpoint.pth')
cat_criterion = nn.CrossEntropyLoss()

# Loaded joint detection model
#pretrained_model = cnn.JointDetectionCNN()
pretrained_model = cnn.JointDetectionCNN_resnet()
#pretrained_model.load_state_dict(torch.load(proj_dir + 'joint_model.pth'))
pretrained_model.load_state_dict(torch.load(proj_dir + 'joint_model_resnet_fe.pth'))
print(pretrained_model)

# Create category classification model
len(category_encoder.classes_)
len(activity_encoder.classes_)
num_classes = 21
classification_model = cat_cnn.ClassificationCNN(pretrained_model, num_classes)

# Print model summary
print(classification_model)

# Initialize model and optimizer
optimizer = torch.optim.Adam([
    {'params': classification_model.feature_extractor.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
    {'params': classification_model.fc1.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
    {'params': classification_model.fc2.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5},
    {'params': classification_model.output.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5}
])

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# move model to gpu if available
pretrained_model.to(device)
classification_model.to(device)


# Training loop (5 epochs -> 8796 sec for all, sec for 3000 images)
epochs = 10
start = time.time()
for epoch in range(epochs):
    
    epoch_start = time.time()
    classification_model.train()
    running_loss = 0.0
    running_val_loss = 0.0
    correct_classes = 0.0
        
    iter_count = 0
    for image_names, images, locs_and_scales, categories, activities in train_dataloader:
        
        # move data to gpu if available
        images, locs_and_scales = images.to(device), locs_and_scales.to(device)
        categories, activities = categories.to(device), activities.to(device)
        locs_and_scales
        
        # set gradient mode
        optimizer.zero_grad()

        # Forward pass
        pred_class = classification_model(images, locs_and_scales, categories.unsqueeze(1))

        # calculate joint location loss with mask and take average (sum / number of visibile joint coords)
        loss = cat_criterion(pred_class, activities)      

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # keep tally of running loss
        running_loss += loss.item()
        
        # regular updates on progress
        iter_count += 1
        if iter_count % 10 == 0: print(f"{iter_count}/{len(train_dataloader)} completed")

    train_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] Training complete ({train_time-epoch_start:.1f} seconds) Loss: {running_loss:.2f}")
    
    for image_names, images, locs_and_scales, categories, activities in val_dataloader:
        
        #activity_encoder.inverse_transform(activities)
        # move data to gpu if available
        images, locs_and_scales = images.to(device), locs_and_scales.to(device)
        categories, activities = categories.to(device), activities.to(device)

        classification_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            pred_class = classification_model(images, locs_and_scales, categories.unsqueeze(1))
                       
        #activity_encoder.inverse_transform(activities)
        correct_class_preds = sum(torch.argmax(pred_class, dim=-1)==torch.argmax(activities, dim=1) )
        correct_classes += correct_class_preds.item()
        
        # calculate loss with mask and take average (sum / number of visibile joint coords)
        val_loss = cat_criterion(pred_class, activities)
        
        # keep tally of running loss
        running_val_loss += val_loss.item()

    val_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] Validation complete ({val_time-train_time:.1f} seconds) Loss: {running_val_loss:.2f}")
    print(f"Correct activity predictions: {correct_classes}/{val_dataset.__len__()}")

    early_stopping(running_val_loss, classification_model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Exiting training loop.")
        break
   
end = time.time()
print(f"Training completed ({end - start:.1f} seconds)")


torch.save(classification_model.state_dict(), proj_dir + 'act_model.pth')
classification_model.load_state_dict(torch.load(proj_dir + 'checkpoint.pth'))


# run validation set to get predictions for visualization
joint_loc_preds_df = pd.DataFrame(columns=['filename'])
classification_model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    running_val_loss = 0.0
    
    batch_count = 0
    for image_names, scales, images, locs_and_scales, joint_visibility, joint_positions in val_dataloader:
        
        # move data to gpu if available
        images, locs_and_scales = images.to(device), locs_and_scales.to(device)
        categories, activities = categories.to(device), activities.to(device)

        pred_class = classification_model(images, locs_and_scales, categories[0].unsqueeze(1))

        ##### TODO -->
        # store pred_classifications for comparison
        
        batch_count += 1
        if batch_count > 0: break

    #print(f"Val Loss: {running_val_loss/len(val_dataloader):.2f}")


# create a set of test outputs and visualize
joint_loc_preds_df.to_csv(proj_dir + 'joint_val_preds_df.csv', index=False)

# save model
#torch.save(classification_model, proj_dir + 'jt_model.pth') 
#torch.save(classification_model.state_dict(), proj_dir + 'jt_model_state_dict.pth')

# load model
#model = torch.load(proj_dir + "jt_model.pth")



# ### load test images
# start = time.time()
# images, filenames, sizes = dl.load_images_in_chunks(img_dir, files_test, transform, 500)
# time.time() - start

# # Get test labels
# label_test = image_labels[image_labels['img_name'].isin(filenames)]
# size_test = pd.DataFrame(sizes, columns=['img_name', 'image_size_w', 'image_size_h'])
# label_test = pd.merge(label_test, size_test)

# # load test data
# test_dataset = dl.getDataset(label_test, images, filenames)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)