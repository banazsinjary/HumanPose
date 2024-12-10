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
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys

code_dir = r"C:\Users\ckite\OneDrive\Documents\GitHub\HumanPose\\"
sys.path.append(code_dir)

import data_loading as dl
import model as cnn

# run on gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set project and image directories
proj_dir = r"C:\Users\ckite\Documents\Project\mpii_human_pose\\"
img_dir = proj_dir + "images"

# load label data
image_labels = pd.read_csv(proj_dir + "labels.csv", index_col=0)
image_labels['img_name'] = [eval(name)[0] for name in image_labels['img_name']]

# Define transformations
H = 448; W = 448
transform = transforms.Compose([
    transforms.Resize((H, W)),  # Resize to target size
    transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])


### Load data and create DataLoaders

# get list of filenames
img_filenames = image_labels['img_name'].values

# split train/validation/test
files_train, files_test = train_test_split(img_filenames, test_size=0.2, random_state=1)
files_train, files_val = train_test_split(
    files_train, test_size=0.25, random_state=1)

##### TODO -->
# save files train and files test

# Get training labels
label_train = image_labels[image_labels['img_name'].isin(files_train)]

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
train_dataset = dl.getDataset(label_train, images, filenames)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


### load validation images
label_val = image_labels[image_labels['img_name'].isin(files_val)]

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
val_dataset = dl.getDataset(label_val, images, filenames)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# early stopping and loss criteria
early_stopping = cnn.EarlyStopping(patience=5, verbose=True, delta=-0.01, path=proj_dir + 'checkpoint.pth')
loc_criterion = nn.MSELoss(reduction='none')  # Mean Squared Error Loss for coordinate prediction

# Initialize model and optimizer
#model = cnn.JointDetectionCNN()
model = cnn.JointDetectionCNN_resnet()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        
# move model to gpu if available
model.to(device)

torch.cuda.empty_cache()
#%%
# Training loop
# Train on vis only first, then coords with weighted vis
epochs = 100
vis_weight = .25
training_losses_vis = []
val_losses_vis = []
start = time.time()
for epoch in range(epochs):
    
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    running_val_vis_loss = 0.0
    running_val_coord_loss = 0.0
    running_val_loss = 0.0
        
    iter_count = 0
    for image_names, scales, images, locs_and_scales, joint_visibility, joint_positions in train_dataloader:
        
        # move data to gpu if available
        images, locs_and_scales = images.to(device), locs_and_scales.to(device)
        joint_positions, joint_visibility = joint_positions.to(device), joint_visibility.to(device)
        
        # set gradient mode
        optimizer.zero_grad()
        
        # Forward pass
        pred_coords, pred_vis = model(images, locs_and_scales)

        # mask outputs
        visibility_mask = joint_visibility.repeat_interleave(2, dim=1)
        visibility_mask = visibility_mask.to(device)
        
        # calculate joint location loss with mask and take average (sum / number of visibile joint coords)
        loss = loc_criterion(pred_coords, joint_positions)
        masked_loss = loss * visibility_mask
        actual_loss = masked_loss.sum() / visibility_mask.sum()
        
        # calculate visibility loss
        vis_loss = nn.functional.binary_cross_entropy_with_logits(pred_vis, joint_visibility)

        # total loss
        total_loss = actual_loss + vis_weight * vis_loss        
        #total_loss = vis_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        # keep tally of running loss
        running_loss += total_loss.item()
        # regular updates on progress
        iter_count += 1
        if iter_count % 100 == 0: print(f"{iter_count}/{len(train_dataloader)} completed")

    training_losses_vis = training_losses_vis + [running_loss]
    train_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] Training complete ({train_time-epoch_start:.1f} seconds) Loss: {running_loss:.2f}")
    
    for image_names, scales, images, locs_and_scales, joint_visibility, joint_positions in val_dataloader:
        
        # move data to gpu if available
        images, locs_and_scales = images.to(device), locs_and_scales.to(device)
        joint_positions, joint_visibility = joint_positions.to(device), joint_visibility.to(device)

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            pred_coords, pred_vis = model(images, locs_and_scales)
        
        # mask outputs
        val_visibility_mask = joint_visibility.repeat_interleave(2, dim=1)
        val_visibility_mask = val_visibility_mask.to(device)

        # calculate loss with mask and take average (sum / number of visibile joint coords)
        val_loss = loc_criterion(pred_coords, joint_positions)
        val_masked_loss = val_loss * val_visibility_mask
        val_actual_loss = val_masked_loss.sum() / val_visibility_mask.sum()
        
        # calculate visibility loss
        val_vis_loss = nn.functional.binary_cross_entropy_with_logits(pred_vis, joint_visibility)

        # total loss
        val_total_loss = val_actual_loss + vis_weight * val_vis_loss        
        #val_total_loss = val_vis_loss
        
        # keep tally of running loss
        running_val_vis_loss += vis_loss.item()
        running_val_coord_loss += val_actual_loss.item()
        running_val_loss += val_total_loss.item()

    val_losses_vis = val_losses_vis + [running_val_loss]
    val_time = time.time()
    print(f"running_val_vis_loss={running_val_vis_loss:.6f}")
    print(f"running_val_coord_loss={running_val_coord_loss:.6f}")
    print(f"Epoch [{epoch+1}/{epochs}] Validation complete ({val_time-train_time:.1f} seconds) Loss: {running_val_loss:.2f}")
    
    early_stopping(running_val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Exiting training loop.")
        break
   
end = time.time()
print(f"Training completed ({end - start:.1f} seconds)")
#%%

###
model.load_state_dict(torch.load(proj_dir + 'checkpoint.pth'))
torch.save(model.state_dict(), proj_dir + 'joint_model_resnet_20241207_multi_conv_redo.pth')
###

#%%
# run validation set to get predictions for visualization
joint_loc_preds_df = pd.DataFrame(columns=['filename'])
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    running_val_loss = 0.0
    
    batch_count = 0
    for image_names, scales, images, locs_and_scales, joint_visibility, joint_positions in val_dataloader:
        
        # move data to gpu if available
        images, locs_and_scales, joint_positions = images.to(device), locs_and_scales.to(device), joint_positions.to(device)

        joint_loc_preds, joint_vis_preds = model(images, locs_and_scales)

        joint_loc_preds_df = pd.concat([joint_loc_preds_df, 
                                        pd.DataFrame({'filename': list(image_names),
                                                      'scale_image_w': scales[:, 0].tolist(),
                                                      'scale_image_h': scales[:, 1].tolist(),
                                                      'rankl_vis': (joint_vis_preds[:, 0]>= 0).int().tolist(),
                                                      'rknee_vis': (joint_vis_preds[:, 1]>= 0).int().tolist(),
                                                      'rhip_vis': (joint_vis_preds[:, 2]>= 0).int().tolist(),
                                                      'lhip_vis': (joint_vis_preds[:, 3]>= 0).int().tolist(),
                                                      'lknee_vis': (joint_vis_preds[:, 4]>= 0).int().tolist(),
                                                      'lankl_vis': (joint_vis_preds[:, 5]>= 0).int().tolist(),
                                                      'pelvis_vis': (joint_vis_preds[:, 6]>= 0).int().tolist(),
                                                      'thorax_vis': (joint_vis_preds[:, 7]>= 0).int().tolist(),
                                                      'upper_neck_vis': (joint_vis_preds[:, 8]>= 0).int().tolist(),
                                                      'head_top_vis': (joint_vis_preds[:, 9]>= 0).int().tolist(),
                                                      'rwri_vis': (joint_vis_preds[:, 10]>= 0).int().tolist(),
                                                      'relb_vis': (joint_vis_preds[:, 11]>= 0).int().tolist(),
                                                      'rsho_vis': (joint_vis_preds[:, 12]>= 0).int().tolist(),
                                                      'lsho_vis': (joint_vis_preds[:, 13]>= 0).int().tolist(),
                                                      'lelb_vis': (joint_vis_preds[:, 14]>= 0).int().tolist(),
                                                      'lwri_vis': (joint_vis_preds[:, 15]>= 0).int().tolist(),
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


# create a set of test outputs and visualize
joint_loc_preds_df.to_csv(proj_dir + 'joint_val_preds_df.csv', index=False)

#%%
# save model
torch.save(model, proj_dir + 'jt_model.pth') 
torch.save(model.state_dict(), proj_dir + 'jt_model_state_dict.pth')

# load model
model = torch.load(proj_dir + "jt_model.pth")



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