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
from pathlib import Path
import matplotlib.pyplot as plt


def postprocess_prediction(seg):
    mask = seg != 0
    lbls = label(mask, connectivity=3)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region] = 0
    return seg


def postprocess_output(output, patient_nib_shape):
    to_save = interpolate_image(output, patient_nib_shape)
    to_save[to_save >= 0.9] = 1
    to_save[to_save < 0.9] = 0
    to_save = postprocess_prediction(to_save)

    return to_save


def postprocess_save_output(output, patient_nib, output_path, save_nib_to_disk=False):
    to_save = interpolate_image(output, patient_nib.shape)
    to_save[to_save >= 0.9] = 1
    to_save[to_save < 0.9] = 0
    to_save = postprocess_prediction(to_save)

    to_save_nib = None
    if save_nib_to_disk:
        to_save_nib = nib.Nifti1Image(to_save, patient_nib.affine)
        nib.save(to_save_nib, os.path.join(output_path))
        print(f"to_save Image size: {to_save.shape} , dtype: {to_save.dtype} ")
        print("Output saved at: ", output_path)

    return to_save, to_save_nib


def dice(inp, target):
    smooth = 1e-7
    iflat = inp.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    dice_score = (2 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice_score


def get_mask_image(mask_path):
    mask = nib.load(mask_path)
    mask_data = mask.get_fdata().astype(np.float32)
    # Make the mask_data binary
    mask_data[mask_data > 0] = 1
    return mask_data


def get_input_image(input_path):
    patient_nib = nib.load(input_path)
    image_data = patient_nib.get_fdata()
    old_shape = patient_nib.shape
    image = process_image(image_data)
    image = resize(
        image, (128, 128, 128), order=3, mode="edge", cval=0, anti_aliasing=False
    )
    image = image[np.newaxis, np.newaxis, ...]
    image = torch.FloatTensor(image)
    return image, patient_nib
