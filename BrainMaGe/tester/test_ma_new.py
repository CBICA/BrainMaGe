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


def infer_ma(cfg, device, save_brain, weights):
    cfg = os.path.abspath(cfg)

    if os.path.isfile(cfg):
        params_df = pd.read_csv(
            cfg,
            sep=" = ",
            names=["param_name", "param_value"],
            comment="#",
            skip_blank_lines=True,
            engine="python",
        ).fillna(" ")
    else:
        print("Missing test_params.cfg file? Please give one!")
        sys.exit(0)
    params = {}
    for i in range(params_df.shape[0]):
        params[params_df.iloc[i, 0]] = params_df.iloc[i, 1]
    params["weights"] = weights
    start = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(start))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()

    print("Generating Test csv")
    if not os.path.exists(os.path.join(params["results_dir"])):
        os.mkdir(params["results_dir"])
    if not params["csv_provided"] == "True":
        print("Since CSV were not provided, we are gonna create for you")
        csv_creator_adv.generate_csv(
            params["test_dir"],
            to_save=params["results_dir"],
            mode=params["mode"],
            ftype="test",
            modalities=params["modalities"],
        )
        test_csv = os.path.join(params["results_dir"], "test.csv")
    else:
        test_csv = params["test_csv"]

    model = fetch_model(
        params["model"],
        int(params["num_modalities"]),
        int(params["num_classes"]),
        int(params["base_filters"]),
    )
    checkpoint = torch.load(str(params["weights"]), map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    if device != "cpu":
        model.cuda()
    model.eval()

    test_df = pd.read_csv(test_csv)
    test_df.ID = test_df.ID.astype(str)
    temp_dir = os.path.join(params["results_dir"], "Temp")
    os.makedirs(temp_dir, exist_ok=True)

    print("Resampling the images to isotropic resolution of 1mm x 1mm x 1mm")
    print("Also Converting the images to RAI and brats for smarter use.")

    for index, patient in tqdm.tqdm(test_df.iterrows()):
        os.makedirs(os.path.join(temp_dir, patient["ID"]), exist_ok=True)
        patient_path = patient["Image_path"]

        patient_nib = nib.load(patient_path)

        image_data = patient_nib.get_fdata()
        image = process_image(image_data)
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
            for i in range(to_save.shape[2]):
                if np.any(to_save[:, :, i]):
                    to_save[:, :, i] = binary_fill_holes(to_save[:, :, i])
            to_save = postprocess_prediction(to_save).astype(np.uint8)
            to_save_nib = nib.Nifti1Image(to_save, patient_nib.affine)

            os.makedirs(
                os.path.join(params["results_dir"], patient["ID"]), exist_ok=True
            )

            output_path = os.path.join(
                params["results_dir"], patient["ID"], patient["ID"] + "_mask.nii.gz"
            )

            nib.save(to_save_nib, output_path)

        if save_brain:
            image = nib.load(patient["Image_path"])
            image_data = image.get_fdata()
            mask = nib.load(
                os.path.join(
                    params["results_dir"],
                    patient["ID"],
                    patient["ID"] + "_mask.nii.gz",
                )
            )
            mask_data = mask.get_fdata().astype(np.int8)
            image_data[mask_data == 0] = 0
            to_save_brain = nib.Nifti1Image(image_data, image.affine)
            nib.save(
                to_save_brain,
                os.path.join(
                    params["results_dir"],
                    patient["ID"],
                    patient["ID"] + "_brain.nii.gz",
                ),
            )

    print("*" * 60)
    print("Final output stored in : %s" % (params["results_dir"]))
    print("Thank you for using BrainMaGe")
    print("*" * 60)
