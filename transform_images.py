# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 08:54:32 2024

@author: ckite
"""

import pandas as pd
import numpy as np
from PIL import Image, ImageOps

# # Test annotations DataFrame
# df_annotations = pd.DataFrame({
#     'lwri_x': [0],
#     'lwri_y': [0],
#     'rwri_x': [772],
#     'rwri_y': [294]
# })

def add_padding(image, annotations, padding_ratio=0.2):
    
    # Calculate padding size (e.g., 20% of the larger dimension)
    W, H = image.size
    padding = int(max(W, H) * padding_ratio)

    # Add padding to the image
    padded_image = ImageOps.expand(image, border=padding, fill=(0, 0, 0))  # Fill with black padding
    
    # Adjust annotations
    for col in annotations.columns:
        if col.endswith('_x') or col.endswith('_y'):
            annotations.loc[:,col] += padding

    return padded_image, annotations

# Transformation functions for annotations
def rotate_annotations(annotations, angle, W, H):
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Center of the image
    cx, cy = W / 2, H / 2

    # Rotate each coordinate around the center
    for joint in annotations.index:
        if '_x' in joint:
            # Get corresponding y-coordinate column
            joint_y = joint.replace('_x', '_y')
            
            # Extract current x and y
            x = annotations[joint]
            y = annotations[joint_y]

            # Translate center of the image
            x_shifted = x - cx
            y_shifted = y - cy

            # Apply rotation matrix
            x_rotated = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
            y_rotated = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)

            # Translate back
            annotations[joint] = x_rotated + cx
            annotations[joint_y] = y_rotated + cy

    return annotations

# Applying transformations
def transform_annotations(image, annotations, apply_rotation=True):
    W, H = image.size

    # Apply random rotations
    if apply_rotation:
        angle = np.random.uniform(-30, 30)  # Random angle between -30 and 30 degrees
        image = image.rotate(angle)
        annotations = rotate_annotations(annotations, angle, W, H)

    return image, annotations

# # Image dimensions test
# image = Image.open(r"C:\Users\ckite\Documents\Project\mpii_human_pose\images\060111501.jpg")
# p_image, p_annotations = add_padding(image, df_annotations)

# transformed_image, transformed_annotations = transform_annotations(p_image, df_annotations)

# print(transformed_annotations)
# transformed_image.show()
