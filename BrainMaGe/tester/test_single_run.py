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
import torch
import nibabel as nib
import tqdm
import numpy as np
from skimage.transform import resize
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from BrainMaGe.models.networks import fetch_model
from BrainMaGe.utils import csv_creator_adv
from BrainMaGe.utils.utils_test import (
    pad_image,
    process_image,
    interpolate_image,
    padder_and_cropper,
)


def postprocess_prediction(seg):
    mask = seg != 0
    lbls = label(mask, connectivity=3)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region] = 0
    return seg


def infer_single_ma(input_path, output_path, weights, mask_path=None, device="cpu"):
    start = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(start))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()
    print("Generating Test csv")

    model = fetch_model(
        modelname="resunet", num_channels=1, num_classes=2, num_filters=16
    )
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device != "cpu":
        model.cuda()
    model.eval()

    patient_nib = nib.load(input_path)
    image = patient_nib.get_fdata()
    old_shape = patient_nib.shape
    image = process_image(image)
    image = resize(
        image, (128, 128, 128), order=3, mode="edge", cval=0, anti_aliasing=False
    )
    image = image[np.newaxis, np.newaxis, ...]
    image = torch.FloatTensor(image)
    if device != "cpu":
        image = image.cuda()
    with torch.no_grad():
        output = model(image)
        output = output.cpu().numpy()[0][0]
        to_save = interpolate_image(output, patient_nib.shape)
        to_save[to_save >= 0.9] = 1
        to_save[to_save < 0.9] = 0
        to_save = postprocess_prediction(to_save)
        to_save_nib = nib.Nifti1Image(to_save, patient_nib.affine)
        nib.save(to_save_nib, os.path.join(output_path))

    print("Done with running the model.")

    if mask_path is not None:
        print("You chose to save the brain. We are now saving it with the masks.")
        brain_data = image_data
        brain_data[to_save == 0] = 0
        to_save_brain = nib.Nifti1Image(brain_data, image.affine)
        nib.save(to_save_brain, os.path.join(mask_path))

    print("Thank you for using BrainMaGe")
    print("*" * 60)
