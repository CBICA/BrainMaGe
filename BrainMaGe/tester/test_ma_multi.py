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
from multiprocessing import cpu_count, Pool


def postprocess_prediction(seg):
    try:
        seg = (seg > 0.5).astype(np.uint8)
        mask = seg != 0
        lbls = label(mask, connectivity=3)
        lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
        largest_region = np.argmax(lbls_sizes[1:]) + 1
        seg[lbls != largest_region] = 0
    except Exception:
        pass
    return seg

def _intensity_standardize(image):
    new_image_temp = image[image >= image.mean()]
    p1 = np.percentile(new_image_temp, 2)
    p2 = np.percentile(new_image_temp, 95)
    image[image > p2] = p2
    new_image = (image - p1)/p2
    return new_image.astype(np.float32)

def _preprocess_data(image_src):
    image = nib.load(image_src).get_fdata()
    image = resize(image, (128, 128, 128), order=3, mode='edge', cval=0,
                   anti_aliasing=False)
    image = _intensity_standardize(image)
    return image

def preprocess_batch_works(k):
    sub_patients_path = [patients_path[i] for i in range(len(patients_path)) if i % n_processes == k]
    sub_patients = [patients[i] for i in range(len(patients)) if i % n_processes == k]
    for patient, patient_path in zip(sub_patients, sub_patients_path):
        patient_output_src = os.path.join(preprocessed_output_dir, patient)
        print(patient, patient_path, patient_output_src)
        preprocessed_data = _preprocess_data(patient_path)
        np.savez_compressed(patient_output_src, image=preprocessed_data)


def _save_post_hard(patient_output, output_shape, output_affine, output_dir):
    patient_output = patient_output[0]
    try:
        patient_output = postprocess_prediction(patient_output)
    except:
        pass
    patient_output = np.array(patient_output, dtype=np.float32)
    patient_mask = resize(patient_output, output_shape, order=3,
                          preserve_range=True)
    patient_mask_post = (patient_mask > 0.5).astype(np.int8)
    to_save_post = nib.Nifti1Image(patient_mask_post, output_affine)
    nib.save(to_save_post, os.path.join(output_dir,
                                        os.path.basename(output_dir)+'_mask.nii.gz'))

def process_output(patient_output_src, patient_orig_path, output_dir):
    patient_nib = nib.load(patient_orig_path)
    patient_orig_shape = patient_nib.shape
    patient_orig_affine = patient_nib.affine
    patient_output = np.load(patient_output_src)['output']
    _save_post_hard(patient_output, patient_orig_shape, patient_orig_affine,
                    output_dir)

def postprocess_batch_works(k):
    sub_patients_path = [patients_path[i] for i in range(len(patients_path)) if i % n_processes == k]
    sub_patients = [patients[i] for i in range(len(patients)) if i % n_processes == k]
    for patient, patient_path in zip(sub_patients, sub_patients_path):
        patient_dir = os.path.join(model_dir, patient)
        os.makedirs(patient_dir, exist_ok=True)
        patient_temp_output_src = os.path.join(temp_output_dir, patient+'.npz')
        print(patient, patient_temp_output_src, patient_path)
        process_output(patient_temp_output_src, patient_path, patient_dir)

def infer_ma(hparams):
    global patients, patients_path, n_processes, model_dir, preprocessed_output_dir, temp_output_dir
    model_dir = hparams.model_dir
    training_start_time = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(training_start_time))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()
    # Parsing the number of CPU's used
    n_processes = int(hparams.threads)
    print("Number of CPU's used : ", n_processes)

    # PRINT PARSED HPARAMS
    print("\n\n")
    print("Model Dir               :", hparams.model_dir)
    print("Test CSV                :", hparams.test_csv)
    print("Number of channels      :", hparams.num_channels)
    print("Model Name              :", hparams.model)
    print("Modalities              :", hparams.modalities)
    print("Number of classes       :", hparams.num_classes)
    print("Base Filters            :", hparams.base_filters)
    print("Load Weights            :", hparams.weights)
    
    print("Generating Test csv")
    if not os.path.exists(hparams['results_dir']):
        os.mkdir(hparams.results_dir)
    if not hparams.csv_provided == 'True':
        print('Since CSV were not provided, we are gonna create for you')
        csv_creator_adv.generate_csv(hparams.test_dir,
                                     to_save=hparams.results_dir,
                                     mode=hparams.mode,
                                     ftype='test',
                                     modalities=hparams.modalities)
        test_csv = os.path.join(hparams.results_dir, 'test.csv')
    else:
        test_csv = hparams.test_csv

    n_processes = int(hparams.threads)
    model = fetch_model(hparams.model,
                        int(hparams.num_modalities),
                        int(hparams.num_classes),
                        int(hparams.base_filters))
    checkpoint = torch.load(str(hparams.weights))
    model.load_state_dict(checkpoint.model_state_dict)

    if hparams.device != 'cpu':
        model.cuda()
    model.eval()

    test_df = pd.read_csv(test_csv)
    preprocessed_output_dir = os.path.join(hparams.model_dir, 'preprocessed')
    os.makedirs(preprocessed_output_dir, exist_ok=True)
    patients = test_df.iloc[:, 0].astype(str)
    patients_path = test_df.iloc[:, 1]
    n_processes = int(hparams.threads)
    if len(patients) < n_processes:
        print("\n*********** WARNING ***********")
        print("You are processing less number of patients as compared to the\n"+
              "threads provided, which means you are asking for more resources than \n"+
              "necessary which is not a great practice. Anyway, we have accounted for that \n"+
              "and reduced the number of threads to the maximum number of patients for \n"+
              "better resource management!\n")
        n_processes = len(patients)
    print('*'*80)
    print('Intializing preprocessing')
    print('*'*80)
    print("Initiating the CPU workload on %d threads.\n\n"%n_processes)
    print("Currently processing the following patients : ")
    START = time.time()
    pool = Pool(processes=n_processes)
    pool.map(preprocess_batch_works, range(n_processes))
    END = time.time()
    print("\n\n Preprocessing time taken : {} seconds".format(END - START))

    # Load the preprocessed patients to the dataloader
    print('*'*80)
    print('Intializing Deep Neural Network')
    print('*'*80)
    START = time.time()
    print("Initiating the GPU workload on CUDA threads.\n\n")
    print("Currently processing the following patients : ")
    preprocessed_data_dir = os.path.join(hparams.model_dir, 'preprocessed')
    temp_output_dir = os.path.join(hparams.model_dir, 'temp_output')
    os.makedirs(temp_output_dir, exist_ok=True)
    dataset_infer = VolSegDatasetInfer(preprocessed_data_dir)
    infer_loader = DataLoader(dataset_infer, batch_size=int(hparams.batch_size),
                              shuffle=False, num_workers=int(hparams.threads),
                              pin_memory=False)