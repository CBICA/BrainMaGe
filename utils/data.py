#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:59:07 2020

@author: siddhesh
"""


import nibabel as nib
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import os
import torchvision.transforms.functional as TF
import random

class WholeTumorDataset(Dataset):
    def __init__(self, csv_file, params):
        self.df = pd.read_csv(csv_file, header = 0)
        self.params = params
        
    def __len__(self):
        return len(self.df)
    
    def transform(self, image, mask):
        """[Tkaes the images and converts them info transforms]
        
        [description]
        
        Arguments:
            image {[type]} -- [description]
            mask {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
         # Random horizontal flipping

        if random.random() > 0.5:
             image = np.fliplr(image)
             mask = np.fliplr(mask)
    
         # Random vertical flipping
        if random.random() > 0.5:
             image = np.flipud(image)
             mask = np.flipud(mask)
        
         # Add random rotation
        if random.random() > 0.5:
             image = np.rot90(image, k=1)
             mask = np.rot90(mask, k=1)

        return image, mask
    
    def __getitem__(self, patient_id):
        image_name = self.df.iloc[patient_id, 0]
        gt_path = os.path.join(self.df.iloc[patient_id, 1])
        image_path = os.path.join(self.df.iloc[patient_id, 2])
        gt = np.load(gt_path)['arr_0']
        image = np.load(image_path)['arr_0']
        if random.random() > 0.5:
            image, gt = self.transform(image, gt)
        image = np.reshape(image.astype(np.float32), (4, 128, 128, 128))
        gt_data = np.reshape(gt.astype(np.float32), (1, 128, 128, 128))
        # print("Loader: ", image.shape, gt_data.shape)
        sample = {'image_name':image_name, 'image_data':image, 'gt_data':gt_data}
        return sample