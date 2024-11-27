# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:54:06 2024

@author: Charl
"""

import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import transform_images as ti


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


# Load images
def load_images(folder_path, chunk_labels, transform):
    t_images = []
    t_labels = []
    incl_files = []
    sizes = []
    for index, label in chunk_labels.iterrows():
        filename = label['img_name']
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(img_path).convert('RGB') 
                size = image.size
            except FileNotFoundError:
                print(filename)
                continue
            
            t_image, t_label = ti.transform_annotations(image, label)
            t_image = transform(t_image)  
            t_images.append(t_image)
            t_labels.append(t_label)
            incl_files = incl_files + [filename]
            sizes = sizes + [[filename, *size]]
            
    return torch.stack(t_images), t_labels, incl_files, sizes  


def load_images_in_chunks(folder_path, data, transform, chunk_size=100):
    all_images = []
    all_labels = []
    all_files = []
    all_sizes = []
    
    # Process the file list in chunks
    for i in range(0, len(data['img_name']), chunk_size):
        chunk_labels = data.iloc[i:i+chunk_size]
        images, labels, incl_files, sizes = load_images(folder_path, chunk_labels, transform)
        all_images.append(images)
        all_labels.append(labels)
        all_files.extend(incl_files)
        all_sizes.extend(sizes)
        print(f"{i + chunk_size} completed")
    
    # Combine the results from all chunks
    final_images = torch.cat(all_images)  # Combine all image tensors
    return final_images, all_labels, all_files, all_sizes


# create dataset from labels, images and filenames
def getDataset(labels, images, files):
    
    # scale x values
    scale_x = 1 / labels['image_size_w']
    x_cols = labels.filter(like='_x')
    for col in x_cols.columns: labels[col] = labels[col] * scale_x

    # scale y values
    scale_y = 1 / labels['image_size_h']
    # select fields that contain _y and multiply by scale_y
    y_cols = labels.filter(like='_y')
    for col in y_cols.columns: labels[col] = labels[col] * scale_y
    
    scales = torch.tensor(labels[['image_size_w','image_size_h']].values, dtype=torch.float32)
    loc_and_scale = torch.tensor(labels[['objpos_x', 'objpos_y', 'scale']].values, dtype=torch.float32)
    joint_vis = torch.tensor(labels[['rankl', 'rknee', 'rhip', 'lhip', 'lknee', 'lankl', 
                                        'pelvis', 'thorax', 'upper_neck', 'head', 'rwri', 'relb',
                                        'rsho', 'lsho', 'lelb', 'lwri']].values, dtype=torch.float32)
    joint_locs = torch.tensor(labels[['rankl_x', 'rankl_y', 'rknee_x', 'rknee_y', 'rhip_x', 'rhip_y',
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

