#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 06:35:05 2020

@author: siddhesh
"""

import os
import nibabel as nib
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd


class SkullStripDataset(Dataset):
    """
    Skull strip dataloader
    """
    def __init__(self, csv_file, params, test=False):
        self.df = pd.read_csv(csv_file, header=0)
        self.params = params
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, patient_id):
        image_name = self.df.iloc[patient_id, 0]
        if not self.test:
            ground_truth_path = os.path.join(self.df.iloc[patient_id, 1])
            ground_truth = nib.load(ground_truth_path)
            nmods = self.params['num_modalities']
            stack = np.zeros([int(nmods), 128, 128, 128], dtype=np.float32)
            for i in range(int(nmods)):
                image_path = os.path.join(self.df.iloc[patient_id, i+2])
                image = nib.load(image_path)
                image_data = image.get_fdata().astype(np.float32)[np.newaxis, ...]
                stack[i] = image_data
            ground_truth_data = ground_truth.get_data().astype(np.float32)[np.newaxis, ...]
            affine = image.affine
            sample = {'image_name': image_name, 'image_data': stack,
                      'ground_truth_data': ground_truth_data, 'affine': affine}
        else:
            nmods = self.params['num_modalities']
            stack = np.zeros([int(nmods), 128, 128, 128], dtype=np.float32)
            for i in range(int(nmods)):
                image_path = os.path.join(self.df.iloc[patient_id, i+1])
                image = nib.load(image_path)
                image_data = image.get_fdata().astype(np.float32)[np.newaxis, ...]
                stack[i] = image_data
            affine = image.affine
            sample = {'image_name': image_name, 'image_data': stack,
                      'affine': affine}
        return sample
