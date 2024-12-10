# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:37:18 2024

"""

import os
import scipy.io
import pandas as pd
import copy

# set directory and get filenames
proj_dir = r"C:\Users\ckite\Documents\Project\mpii_human_pose\\"
filenames = os.listdir(proj_dir + "images")

# index to joint name conversion
mpii_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 3: 'lhip', 4: 'lknee', 5: 'lankl', 
                   6: 'pelvis', 7: 'thorax', 8: 'upper_neck', 9: 'head', 10: 'rwri', 
                   11: 'relb', 12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

# load mat file
matlab_mpii = scipy.io.loadmat(os.path.join(proj_dir, 'mpii_human_pose_v1_u12_1.mat'), 
                               struct_as_record=False)['RELEASE'][0, 0]

#matlab_mpii.__dict__
#matlab_mpii.__dict__['annolist'][0, 0].__dict__
#annolist_train = matlab_mpii.__dict__['single_person'][matlab_mpii.__dict__['img_train'][0]==1]
#annolist_test = matlab_mpii.__dict__['single_person'][matlab_mpii.__dict__['img_train'][0]==0]


# get number of images
num_images = matlab_mpii.__dict__['annolist'][0].shape[0]

# names of label columns
label_colnames = ['img_name', 'person', 'category', 'activity',
                  'objpos_x', 'objpos_y', 'scale',
                  'rankl', 'rankl_x', 'rankl_y', 
                  'rknee', 'rknee_x', 'rknee_y',
                  'rhip', 'rhip_x', 'rhip_y',
                  'lhip', 'lhip_x', 'lhip_y',
                  'lknee', 'lknee_x', 'lknee_y', 
                  'lankl', 'lankl_x', 'lankl_y',
                  'pelvis', 'pelvis_x', 'pelvis_y',
                  'thorax', 'thorax_x', 'thorax_y',
                  'upper_neck', 'upper_neck_x', 'upper_neck_y',
                  'head', 'head_x', 'head_y',
                  'rwri', 'rwri_x', 'rwri_y',
                  'relb', 'relb_x', 'relb_y',
                  'rsho', 'rsho_x', 'rsho_y',
                  'lsho', 'lsho_x', 'lsho_y',
                  'lelb', 'lelb_x', 'lelb_y', 
                  'lwri', 'lwri_x', 'lwri_y',
                  'max_x', 'min_x', 'max_y', 'min_y']

# template for joint dictionary
joint_dict_template = {}
for i in range(16): joint_dict_template[i] = (0,0,0)

# dataframes for storing labels
training_labels = pd.DataFrame(columns=label_colnames)
test_labels = pd.DataFrame(columns=label_colnames)

# iterate through images to extract labels of images with 0 or 1 person in them
for i in range(num_images):

    # check # of annotated single persons in image
    person_id = matlab_mpii.__dict__['single_person'][i][0].flatten()    
    if len(person_id) > 1: continue # exclude images with more than 1 single person in annotations
    #if len(person_id) > 1: person_id = person_id[0] # include only 1st single person in annotations
        
    # get annotations, training/test allocation, name, category and activity of image
    annotations = matlab_mpii.__dict__['annolist'][0, i]
    train_test_mpii = matlab_mpii.__dict__['img_train'][0, i].flatten()[0]
    img_name = annotations.__dict__['image'][0, 0].__dict__['name'][0]

    act = matlab_mpii.__dict__['act'][i, 0]    
    img_cat = act.__dict__['cat_name'].tolist()
    if len(img_cat) == 0:
        img_cat = ""
    else:
        img_cat = img_cat[0]
    img_act = act.__dict__['act_name'].tolist()
    
    ### dropping images with no annotated person(s) in them
    # set labels of images without an annotated person in them
    # if len(person_id) < 1:
    #     # drop image without an annotated single person in them
    #     img_labels = [img_name] + [0]*52
    #     if train_test_mpii == 1:
    #         training_labels = pd.concat([training_labels, pd.DataFrame([img_labels], columns = label_colnames)])
    #     elif train_test_mpii == 0:
    #         test_labels = pd.concat([test_labels, pd.DataFrame([img_labels], columns = label_colnames)])
    #     else:
    #         print([train_test_mpii, img_name])
    
    # set labels of images with 1 person in them
    if len(person_id) == 1:
        joint_dict = copy.deepcopy(joint_dict_template)
        # use try/catch to handle key and index errors
        try:
            # get annotated points and number of joints in image
            annorect_img_mpii = annotations.__dict__['annorect'][0, person_id-1][0]
            obj_pos = annorect_img_mpii.__dict__['objpos'][0, 0]
            scale = annorect_img_mpii.__dict__['scale'][0][0]
            annopoints_img_mpii = annorect_img_mpii.__dict__['annopoints'][0, 0]
            num_joints = annopoints_img_mpii.__dict__['point'][0].shape[0]
            
            # create base for image labels for this image
            img_labels = [img_name, 1, img_cat, img_act, 
                          obj_pos.__dict__['x'].flatten()[0], 
                          obj_pos.__dict__['y'].flatten()[0], scale]
            max_x, min_x, max_y, min_y = 0, 10000, 0, 10000
            for j in range(num_joints):
                # check visibility of joint
                vis = annopoints_img_mpii.__dict__['point'][0, j].__dict__['is_visible'].flatten()
                # head and neck have no entry for vis (it is an empty array)
                # so only drop joints that are set as not visible
                if vis.size > 0 and vis[0] == 0: continue
            
                # get id and location of joint and store in dictionary
                x = annopoints_img_mpii.__dict__['point'][0, j].__dict__['x'].flatten()[0]
                if x > max_x: max_x = x
                if x < min_x: min_x = x
                y = annopoints_img_mpii.__dict__['point'][0, j].__dict__['y'].flatten()[0]
                if y > max_y: max_y = y
                if y < min_y: min_y = y
                if y < 0: print(y)
                id_ = annopoints_img_mpii.__dict__['point'][0, j].__dict__['id'][0][0]
                joint_dict[id_] = (1, x, y)
                
            # add joint visibility and locations to labels for this image
            for j_id in range(16):
                img_labels = img_labels + [*joint_dict[j_id]]
            
            img_labels = img_labels + [max_x , min_x, max_y, min_y]
                
                
            ##### TODO ->
            # Get max x and add padding
            
            # Get min x and add padding
            # Get max y and add padding
            # Get min y and add padding
            
            # add image labels to appropriate df
            if train_test_mpii == 1:
                training_labels = pd.concat([training_labels, pd.DataFrame([img_labels], columns = label_colnames)])
            elif train_test_mpii == 0:
                test_labels = pd.concat([test_labels, pd.DataFrame([img_labels], columns = label_colnames)])
            else:
                print([train_test_mpii, img_name])
        
        # catch errors
        except KeyError:
            ### no annotated joints: drop from dataset
            # print('Key Error: image #' + str(i) + ", " + img_name)
            # img_labels = [img_name, 1] + [0]*51
            # if train_test_mpii == 1:
            #     training_labels = pd.concat([training_labels, pd.DataFrame([img_labels], columns = label_colnames)])
            # elif train_test_mpii == 0:
            #     test_labels = pd.concat([test_labels, pd.DataFrame([img_labels], columns = label_colnames)])
            # else:
            #     print([train_test_mpii, img_name])
            continue
        except IndexError:
            # images with more than 1 person but only 1 is annotated
            # print(i, img_name)
            continue
    
training_labels.to_csv(proj_dir + "labels.csv")
#test_labels.to_csv(proj_dir + "test_labels.csv")
