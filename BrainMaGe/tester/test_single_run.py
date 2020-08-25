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


def postprocess_prediction(seg):
    mask = seg != 0
    lbls = label(mask, connectivity=3)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region] = 0
    return seg

def infer_single_ma(hparams):
    start = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(start))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()
    print("Generating Test csv")
    if not os.path.exists(os.path.join(hparams.results_dir)):
        os.mkdir(hparams.results_dir)
    temp_dir = os.path.join(hparams.results_dir, 'Temp')

    subjects = hparams.subjects