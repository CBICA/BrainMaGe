#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:18:35 2020

@author: siddhesh
"""

from __future__ import print_function, division
import os
import sys
import time
import pandas as pd
import torch
import nibabel as nib
import tqdm
import numpy as np
from skimage.transform import resize
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from BrainMaGe.models.networks import fetch_model
from BrainMaGe.utils import csv_creator_adv
from BrainMaGe.utils.utils_test import pad_image, process_image, interpolate_image,\
    padder_and_cropper

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC

def infer_ma(cfg, device, save_brain, weights):

    cfg = os.path.abspath(cfg)

    if os.path.isfile(cfg):
        params_df = pd.read_csv(cfg, sep=' = ', names=['param_name', 'param_value'],
                                comment='#', skip_blank_lines=True,
                                engine='python').fillna(' ')
    else:
        print('Missing test_params.cfg file? Please give one!')
        sys.exit(0)
    params = {}
    for i in range(params_df.shape[0]):
        params[params_df.iloc[i, 0]] = params_df.iloc[i, 1]
    params['weights'] = weights
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
    temp_dir = os.path.join(params['results_dir'], 'Temp')
    os.makedirs(temp_dir, exist_ok=True)

    patients_dict = {}

    print("Resampling the images to isotropic resolution of 1mm x 1mm x 1mm")
    print("Also Converting the images to RAI and brats for smarter use.")
    for patient in tqdm.tqdm(test_df.values):
        os.makedirs(os.path.join(temp_dir, patient[0]), exist_ok=True)
        patient_path = patient[1]
        image = nib.load(patient_path)
        old_spacing = image.header.get_zooms()
        old_affine = image.affine
        old_shape = image.header.get_data_shape()
        new_spacing = (1, 1, 1)
        new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                     int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                     int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
        image_data = image.get_fdata()
        new_image = resize(image_data, new_shape, order=3, mode='edge', cval=0,
                           anti_aliasing=False)
        new_affine = np.eye(4)
        new_affine = np.array(old_affine)
        for i in range(3):
            for j in range(3):
                if old_affine[i, j] != 0:
                    new_affine[i, j] = old_affine[i, j]*(1/old_affine[i, j])
                    if old_affine[i, j] <= 0:
                        new_affine[i, j] = -1*(old_affine[i, j]*(1/old_affine[i, j]))
        temp_image = nib.Nifti1Image(new_image, new_affine)
        nib.save(temp_image, os.path.join(temp_dir, patient[0],
                                          patient[0]+'_resamp111.nii.gz'))

        temp_dict = {}
        temp_dict['name'] = patient[0]
        temp_dict['old_spacing'] = old_spacing
        temp_dict['old_affine'] = old_affine
        temp_dict['old_shape'] = old_shape
        temp_dict['new_spacing'] = new_spacing
        temp_dict['new_affine'] = new_affine
        temp_dict['new_shape'] = new_shape

        patient_path = os.path.join(temp_dir, patient[0],
                                    patient[0]+'_resamp111.nii.gz')
        patient_nib = nib.load(patient_path)
        patient_data = patient_nib.get_fdata()
        patient_data, pad_info = pad_image(patient_data)
        patient_affine = patient_nib.affine
        temp_image = nib.Nifti1Image(patient_data, patient_affine)
        nib.save(temp_image, os.path.join(temp_dir, patient[0], patient[0]+'_bratsized.nii.gz'))
        temp_dict['pad_info'] = pad_info
        patients_dict[patient[0]] = temp_dict

    model = fetch_model(params['model'],
                        int(params['num_modalities']),
                        int(params['num_classes']),
                        int(params['base_filters']))
    checkpoint = torch.load(str(params['weights']))
    model.load_state_dict(checkpoint['model_state_dict'])

    if device != 'cpu':
        model.cuda()
    model.eval()


    print("Done Resampling the Data.\n")
    print("--"*30)
    print("Running the model on the subjects")
    for patient in tqdm.tqdm(test_df.values):
        patient_path = os.path.join(temp_dir, patient[0],
                                    patient[0]+'_bratsized.nii.gz')
        patient_nib = nib.load(patient_path)
        image = patient_nib.get_fdata()
        image = process_image(image)
        image = resize(image, (128, 128, 128), order=3, mode='edge', cval=0,
                       anti_aliasing=False)
        image = image[np.newaxis, np.newaxis, ...]
        image = torch.FloatTensor(image)
        if device != 'cpu':
            image = image.cuda()
        with torch.no_grad():
            output = model(image)
            output = output.cpu().numpy()[0][0]
            to_save = interpolate_image(output, patient_nib.shape)
            to_save[to_save >= 0.9] = 1
            to_save[to_save < 0.9] = 0
            to_save_nib = nib.Nifti1Image(to_save, patient_nib.affine)
            nib.save(to_save_nib, os.path.join(temp_dir,
                                               patient[0],
                                               patient[0]+'_bratsized_mask.nii.gz'))
            current_patient_dict = patients_dict[patient[0]]
            new_image = padder_and_cropper(to_save, current_patient_dict['pad_info'])
            to_save_new_nib = nib.Nifti1Image(new_image, patient_nib.affine)
            nib.save(to_save_new_nib, os.path.join(temp_dir,
                                                   patient[0],
                                                   patient[0]+'_resample111_mask.nii.gz'))
            to_save_final = resize(new_image, current_patient_dict['old_shape'], order=0,
                                   mode='edge', cval=0)
            to_save_final[to_save_final > 0] = 1
            for i in range(to_save_final.shape[2]):
                if np.any(to_save_final[:, :, i]):
                    to_save_final[:, :, i] = binary_fill_holes(to_save_final[:, :, i])
            to_save_final = getLargestCC(to_save_final)
            to_save_final_nib = nib.Nifti1Image(to_save_final,
                                                current_patient_dict['old_affine'])

            os.makedirs(os.path.join(params['results_dir'], patient[0]), exist_ok=True)

            nib.save(to_save_final_nib, os.path.join(params['results_dir'],
                                                     patient[0],
                                                     patient[0]+'_mask.nii.gz'))

    print("Done with running the model.")
    if save_brain:
        print("You chose to save the brain. We are now saving it with the masks.")
        for patient in tqdm.tqdm(test_df.values):
            image = nib.load(patient[1])
            image_data = image.get_fdata()
            mask = nib.load(os.path.join(params['results_dir'],
                                         patient[0],
                                         patient[0]+'_mask.nii.gz'))
            mask_data = mask.get_fdata().astype(np.int8)
            image_data[mask_data == 0] = 0
            to_save_brain = nib.Nifti1Image(image_data, image.affine)
            nib.save(to_save_brain, os.path.join(params['results_dir'],
                                                 patient[0],
                                                 patient[0]+'_brain.nii.gz'))

    print("Please check the %s folder for the intermediate outputs if you\"+\
          would like to see some intermediate steps." % (os.path.join(params['results_dir'], 'Temp')))
    print("Final output stored in : %s" % (params['results_dir']))
    print("Thank you for using BrainMaGe")
    print('*'*60)
