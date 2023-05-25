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
from skimage.measure import label
from BrainMaGe.models.networks import fetch_model
from BrainMaGe.utils import csv_creator_adv
from BrainMaGe.utils.utils_test import interpolate_image, unpad_image
from BrainMaGe.utils.preprocess import preprocess_image


def postprocess_prediction(seg):
    mask = seg != 0
    lbls = label(mask, connectivity=3)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region] = 0
    return seg


def infer_single_multi_4(
    input_paths, output_path, weights, mask_path=None, device="cpu"
):
    """
    Inference using multi modality network

    Parameters [TODO]
    ----------
    input_paths : list
        path to all input images following T1_path,T2_path,T1ce_path,Flair_path
    output_path : str
        path of the mask to be generated (prediction)
    weights : str
        path to the weights of the model used
    device : int/str
        device to be run on

    Returns
    -------
    None.

    """
    assert all([os.path.exists(image_path) for image_path in input_paths])

    start = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(start))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()

    # default config for multi-4 as from config/test_params_multi_4.cfg
    model = fetch_model(
        modelname="resunet",
        num_channels=4,
        num_classes=2,
        num_filters=16,
    )

    checkpoint = torch.load(str(weights), map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    if device != "cpu":
        model.cuda()
    model.eval()

    stack = np.zeros([4, 128, 128, 128], dtype=np.float32)
    for i, image_path in enumerate(input_paths):
        patient_nib = nib.load(image_path)
        image = patient_nib.get_fdata()
        image = preprocess_image(patient_nib)
        stack[i] = image
    stack = stack[np.newaxis, ...]
    image = torch.FloatTensor(stack)

    if device != "cpu":
        image = image.cuda()

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
        to_save = postprocess_prediction(to_save).astype(np.uint8)
        to_save_nib = nib.Nifti1Image(to_save, patient_nib.affine)
        nib.save(to_save_nib, os.path.join(output_path))

    print("Done with running the model.")

    if mask_path is not None:
        raise NotImplementedError("Sorry, masking is not implemented (yet).")

    print("Final output stored in : %s" % (output_path))
    print("Thank you for using BrainMaGe")
    print("*" * 60)
