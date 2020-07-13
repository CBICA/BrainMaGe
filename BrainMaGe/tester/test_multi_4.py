#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 03:18:40 2020

@author: siddhesh
"""

from __future__ import print_function, division
import os
import sys
import time
import pandas as pd
import torch
import nibabel as nib
import numpy as np
import tqdm
from scipy.ndimage.morphology import binary_fill_holes
from BrainMaGe.models.networks import fetch_model
from BrainMaGe.utils import csv_creator_adv
from BrainMaGe.utils.utils_test import interpolate_image, unpad_image
from BrainMaGe.utils.preprocess import preprocess_image


def infer_multi_4(cfg, device, save_brain, weights):
    """
    Inference using multi modality network

    Parameters
    ----------
    cfg : string
        Location of the config file
    device : int/str
        device to be run on
    save_brain : int
        whether to save brain or not

    Returns
    -------
    None.

    """
    cfg = os.path.abspath(cfg)

    if os.path.isfile(cfg):
        params_df = pd.read_csv(cfg, sep=' = ', names=['param_name', 'param_value'],
                                comment='#', skip_blank_lines=True,
                                engine='python').fillna(' ')
    else:
        print('Missing test_params.cfg file? Please give one!')
        sys.exit(0)
    params = {}
    params['weights'] = weights
    for i in range(params_df.shape[0]):
        params[params_df.iloc[i, 0]] = params_df.iloc[i, 1]
    start = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(start))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()

    print("Generating Test csv")
    if not os.path.exists(os.path.join(params['results_dir'])):
        os.mkdir(params['results_dir'])
    if not params['csv_provided'] == 'True':
        print('Since CSV were not provided, we are gonna create for you')
        csv_creator_adv.generate_csv(params['test_dir'],
                                     to_save=params['results_dir'],
                                     mode=params['mode'], ftype='test',
                                     modalities=params['modalities'])
        test_csv = os.path.join(params['results_dir'], 'test.csv')
    else:
        test_csv = params['test_csv']

    test_df = pd.read_csv(test_csv)

    model = fetch_model(params['model'],
                        int(params['num_modalities']),
                        int(params['num_classes']),
                        int(params['base_filters']))
    if device != 'cpu':
        model.cuda()

    checkpoint = torch.load(str(params['weights']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for patient in tqdm.tqdm(test_df.values):
        os.makedirs(os.path.join(params['results_dir'], patient[0]), exist_ok=True)
        nmods = params['num_modalities']
        stack = np.zeros([int(nmods), 128, 128, 128], dtype=np.float32)
        for i in range(int(nmods)):
            image_path = patient[i+1]
            patient_nib = nib.load(image_path)
            image = patient_nib.get_fdata()
            image = preprocess_image(patient_nib)
            stack[i] = image
        stack = stack[np.newaxis, ...]
        stack = torch.FloatTensor(stack)
        if device != 'cpu':
            image = stack.cuda()
        with torch.no_grad():
            output = model(image)
            output = output.cpu().numpy()[0][0]
            to_save = interpolate_image(output, (240, 240, 160))
            to_save = unpad_image(to_save)
            to_save[to_save >= 0.9] = 1
            to_save[to_save < 0.9] = 0
            for i in range(to_save.shape[2]):
                if np.any(to_save[:, :, i]):
                    to_save[:, :, i] = binary_fill_holes(to_save[:, :, i])
            to_save_mask = nib.Nifti1Image(to_save, patient_nib.affine)
            nib.save(to_save_mask, os.path.join(params['results_dir'], patient[0],
                                                patient[0]+'_mask.nii.gz'))
    print("Done with running the model.")
    if save_brain:
        print("You chose to save the brain. We are now saving it with the masks.")
        for patient in tqdm.tqdm(test_df.values):
            nmods = params['num_modalities']
            mask_nib = nib.load(os.path.join(params['results_dir'], patient[0],
                                             patient[0]+'_mask.nii.gz'))
            mask_data = mask_nib.get_fdata().astype(np.int8)
            for i in range(int(nmods)):
                image_name = os.path.basename(patient[i+1]).strip('.nii.gz')
                image_path = patient[i+1]
                patient_nib = nib.load(image_path)
                image_data = patient_nib.get_fdata()
                image_data[mask_data == 0] = 0
                to_save_image = nib.Nifti1Image(image_data, patient_nib.affine)
                nib.save(to_save_image, os.path.join(params['results_dir'],
                                                     patient[0],
                                                     image_name+'_brain.nii.gz'))

    print("Final output stored in : %s" % (params['results_dir']))
    print("Thank you for using BrainMaGe")
    print('*'*60)
