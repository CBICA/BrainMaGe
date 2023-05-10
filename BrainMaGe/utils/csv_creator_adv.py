#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 06:33:50 2020

@author: siddhesh
"""


import os
import sys
import glob
import re
from bids import BIDSLayout


def rex_o4a_csv(folder_path, save_folder, file_type, modalities):
    """
    CSV generation for OneForAll.
    This function generates a csv for OneForAll mode and creates a csv.

    Args:
        folder_path (str): The folder to see where to look for the different modalities
        save_folder (str): The folder to save the csv
        file_type (str): train, validation or test. If file_type is set to test, it does not look for ground truths.
        modalities (list of str): The modalities to include in the csv.
    """
    modalities = modalities[1:-1]
    modalities = re.findall("[^, ']+", modalities)
    if not modalities:
        raise ValueError("Could not find modalities! Are you sure you have put in something in the modalities field?")
        
    if file_type == "test":
        csv_path = os.path.join(save_folder, file_type + ".csv")
        with open(csv_path, "w+") as csv_file:
            csv_file.write("ID,Image_Path\n")
    else:
        csv_path = os.path.join(save_folder, file_type + ".csv")
        with open(csv_path, "w+") as csv_file:
            csv_file.write("ID,gt_path,Image_path\n")
            
    folders = os.listdir(folder_path)
    for folder in folders:
        for modality in modalities:
            csv_file.write(folder + "_" + modality + ",")
            
            if file_type != "test":
                ground_truth_path = glob.glob(os.path.join(folder_path, folder, "*mask.nii.gz"))[0]
                csv_file.write(ground_truth_path)
                csv_file.write(",")
            
            image_path = glob.glob(os.path.join(folder_path, folder, "*" + modality + ".nii.gz"))[0]
            csv_file.write(image_path)
            csv_file.write("\n")
            
    print("CSV file saved successfully at:", csv_path)


def rex_sin_csv(folder_path, save_folder, file_type, modalities):
    """
    CSV generation for Single Modalities.
    This function generates a csv for Single mode and creates a csv.

    Args:
        folder_path (str): The folder to see where to look for the different modalities
        save_folder (str): The folder to save the csv
        file_type (str): train, validation or test. If file_type is set to test, it does not look for ground truths.
        modalities (list of str): The modalities to include in the csv.
    """
    modalities = modalities[1:-1]
    modalities = re.findall("[^, ']+", modalities)
    if len(modalities) > 1:
        raise ValueError("Found more than one modality, exiting!")
    if not modalities:
        raise ValueError("Could not find modalities! Are you sure you have put in something in the modalities field?")
        
    if file_type == "test":
        csv_path = os.path.join(save_folder, file_type + ".csv")
        with open(csv_path, "w+") as csv_file:
            csv_file.write("ID,")
    else:
        csv_path = os.path.join(save_folder, file_type + ".csv")
        with open(csv_path, "w+") as csv_file:
            csv_file.write("ID,gt_path,")
            
    modality = modalities[0]
    csv_file.write(modality + "_path\n")
    folders = os.listdir(folder_path)
    for folder in folders:
        csv_file.write(folder)
        csv_file.write(",")
        if file_type != "test":
            ground_truth_path = glob.glob(os.path.join(folder_path, folder, "*mask.nii.gz"))[0]
            csv_file.write(ground_truth_path)
            csv_file.write(",")
            
        image_path = glob.glob(os.path.join(folder_path, folder, "*" + modality + ".nii.gz"))[0]
        csv_file.write(image_path)
        csv_file.write("\n")
            
    print("CSV file saved successfully at:", csv_path)


def rex_mul_csv(folder_path, save_folder, file_type, modalities):
    """
    CSV generation for Multi Modalities.
    This function generates a csv for multi mode and creates a csv.

    Args:
        folder_path (str): The folder to see where to look for the different modalities
        save_folder (str): The folder to save the csv
        file_type (str): train, validation or test. If file_type is set to test, it does not look for ground truths.
        modalities (list of str): The modalities to include in the csv.
    """
    modalities = modalities[1:-1]
    modalities = re.findall("[^, ']+", modalities)
    if not modalities:
        raise ValueError("Could not find modalities! Are you sure you have put in something in the modalities field?")
        
    if file_type == "test":
        csv_path = os.path.join(save_folder, file_type + ".csv")
        with open(csv_path, "w+") as csv_file:
            csv_file.write("ID,")
    else:
        csv_path = os.path.join(save_folder, file_type + ".csv")
        with open(csv_path, "w+") as csv_file:
            csv_file.write("ID,gt_path,")
            
    csv_file.write(",".join([f"{modality}_path" for modality in modalities]))
    csv_file.write("\n")
    
    folders = os.listdir(folder_path)
    for folder in folders:
        csv_file.write(folder)
        csv_file.write(",")
        
        if file_type != "test":
            ground_truth_path = glob.glob(os.path.join(folder_path, folder, "*mask.nii.gz"))[0]
            csv_file.write(ground_truth_path)
            csv_file.write(",")
            
        for modality in modalities[:-1]:
            image_path = glob.glob(os.path.join(folder_path, folder, "*" + modality + ".nii.gz"))[0]
            csv_file.write(image_path)
            csv_file.write(",")
            
        image_path = glob.glob(os.path.join(folder_path, folder, "*" + modalities[-1] + ".nii.gz"))[0]
        csv_file.write(image_path)
        csv_file.write("\n")
            
    print("CSV file saved successfully at:", csv_path)


def rex_bids_csv(folder_path, save_folder, file_type):
    """
    CSV generation for BIDS datasets.
    This function generates a csv for BIDS datasets.

    Args:
        folder_path (str): The folder to see where to look for the different modalities
        save_folder (str): The folder to save the csv
        file_type (str): train, validation or test. If file_type is set to test, it does not look for ground truths.
    """
    if file_type == "test":
        csv_path = os.path.join(save_folder, file_type + ".csv")
        with open(csv_path, "w+") as csv_file:
            csv_file.write("ID,")
    else:
        csv_path = os.path.join(save_folder, file_type + ".csv")
        with open(csv_path, "w+") as csv_file:
            csv_file.write("ID,gt_path,")

    layout = BIDSLayout(folder_path)
    modalities = ["t1", "t2", "flair", "t1ce"]
    modalities = [modality for modality in modalities if layout.get(suffix=f"{modality}w")]
    
    csv_file.write(",".join([f"{modality}_path" for modality in modalities]))
    csv_file.write("\n")
    
    for sub in layout.get_subjects():
        csv_file.write(sub)
        csv_file.write(",")
        
        if file_type != "test":
            ground_truth_path = layout.get(subject=sub, suffix="mask")[0].filename
            csv_file.write(ground_truth_path)
            csv_file.write(",")
            
        for modality in modalities[:-1]:
            image_path = layout.get(subject=sub, suffix=f"{modality}w")[0].filename
            csv_file.write(image_path)
            csv_file.write(",")
            
        image_path = layout.get(subject=sub, suffix=f"{modalities[-1]}w")[0].filename
        csv_file.write(image_path)
        csv_file.write("\n")
            
    print("CSV file saved successfully at:", csv_path)


def generate_csv(folder_path, to_save, mode, ftype, modalities):
    """[Function to generate CSV]
    [This function takes a look at the data directory and the modes and generates a csv]
    Arguments:
        folder_path {[strin]} -- [description]
        to_save {[strin]} -- [description]
        mode {[string]} -- [description]
        ftype {[string]} -- [description]
        modalities {[string]} -- [description]
    """
    print(f"Generating {ftype}.csv")
    modes = {
        "ma": rex_o4a_csv,
        "single": rex_sin_csv,
        "multi": rex_mul_csv,
        "bids": rex_bids_csv
    }
    func = modes.get(mode.lower())
    if func:
        func(folder_path, to_save, ftype, modalities)
    else:
        print("Sorry, this mode is not supported")
        sys.exit(0)
