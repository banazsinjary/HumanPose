# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:54:06 2024

@author: Charl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                
        # Global Average Pooling to reduce spatial dimensions
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        return x
    
# CNN model for head position detection
class JointDetectionCNN(nn.Module):
    def __init__(self, input_channels=3, dropout_rate=.5):
        super(JointDetectionCNN, self).__init__()
        
        self.cnn = FeatureExtraction(input_channels, dropout_rate)
        
        self.fc1 = nn.Linear(256 + 3, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        
        self.joint_coordinates = nn.Linear(64, 32)
        self.joint_visibility = nn.Linear(64, 16)

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
        joint_visibility = self.joint_visibility(x)
        
        return joint_coords, joint_visibility

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
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
            self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
