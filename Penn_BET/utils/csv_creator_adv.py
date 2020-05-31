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


def rex_o4a_csv(folder_path, to_save, ftype, modalities):
    """[CSV generation for OneForAll]
    [This function is used to generate a csv for OneForAll mode and creates a
    csv]
    Arguments:
        folder_path {[string]} -- [Takes the folder to see where to look for
                                   the different modaliies]
        to_save {[string]} -- [Takes the folder as a string to save the csv]
        ftype {[string]} -- [Are you trying to save train, validation or test,
                             if file type is set to test, it does not look for
                             ground truths]
        modalities {[string]} -- [usually a string which looks like this
                                  :  ['t1', 't2', 't1ce']]
    """
    modalities = modalities[1:-1]
    modalities = re.findall('[^, \']+', modalities)
    if not modalities:
        print("Could not find modalities! Are you sure you have put in \
              something in the modalities field?")
        sys.exit(0)
    if ftype == 'test':
        csv_file = open(os.path.join(to_save, ftype+'.csv'), 'w+')
        csv_file.write('ID,Image_Path\n')
    else:
        csv_file = open(os.path.join(to_save, ftype+'.csv'), 'w+')
        csv_file.write('ID,gt_path,Image_path\n')
    folders = os.listdir(folder_path)
    for folder in folders:
        for modality in modalities:
            csv_file.write(folder+'_'+modality+',')
            if ftype != 'test':
                ground_truth = glob.glob(os.path.join(folder_path, folder,
                                                      '*maskFinal*.nii.gz'))[0]
                csv_file.write(ground_truth)
                csv_file.write(',')
            img = glob.glob(os.path.join(folder_path, folder,
                                         '*'+modality+'*.nii.gz'))[0]
            csv_file.write(img)
            csv_file.write('\n')
    csv_file.close()


def rex_sin_csv(folder_path, to_save, ftype, modalities):
    """[CSV generation for Single Modalities]
    [This function is used to generate a csv for Single mode and creates a csv]
    Arguments:
        folder_path {[string]} -- [Takes the folder to see where to look for
                                   the different modaliies]
        to_save {[string]} -- [Takes the folder as a string to save the csv]
        ftype {[string]} -- [Are you trying to save train, validation or test,
                             if file type is set to test, it does not look for
                             ground truths]
        modalities {[string]} -- [usually a string which looks like this
                                  :  ['t1']]
    """
    modalities = modalities[1:-1]
    modalities = re.findall('[^, \']+', modalities)
    if len(modalities) > 1:
        print("Found more than one modality, exiting!")
        sys.exit(0)
    if not modalities:
        print("Could not find modalities! Are you sure you have put in \
              something in the modalities field?")
        sys.exit(0)
    if ftype == 'test':
        csv_file = open(os.path.join(to_save, ftype+'.csv'), 'w+')
        csv_file.write('ID,')
    else:
        csv_file = open(os.path.join(to_save, ftype+'.csv'), 'w+')
        csv_file.write('ID,gt_path,')
    modality = modalities[0]
    csv_file.write(modality+'_path\n')
    folders = os.listdir(folder_path)
    for folder in folders:
        csv_file.write(folder)
        csv_file.write(',')
        if ftype != 'test':
            ground_truth = glob.glob(os.path.join(folder_path, folder,
                                                  '*maskFinal*.nii.gz'))[0]
            csv_file.write(ground_truth)
            csv_file.write(',')
        img = glob.glob(os.path.join(folder_path, folder,
                                     '*'+modality+'*.nii.gz'))[0]
        csv_file.write(img)
        csv_file.write('\n')
    csv_file.close()


def rex_mul_csv(folder_path, to_save, ftype, modalities):
    """[CSV generation for Multi Modalities]
    [This function is used to generate a csv for multi mode and creates a csv]
    Arguments:
        folder_path {[string]} -- [Takes the folder to see where to look for
                                   the different modaliies]
        to_save {[string]} -- [Takes the folder as a string to save the csv]
        ftype {[string]} -- [Are you trying to save train, validation or test,
                             if file type is set to test, it does not look for
                             ground truths]
        modalities {[string]} -- [usually a string which looks like this
                                  :  ['t1']]
    """
    modalities = modalities[1:-1]
    modalities = re.findall('[^, \']+', modalities)
    if not modalities:
        print("Could not find modalities! Are you sure you have put in \
              something in the modalities field?")
        sys.exit(0)
    if ftype == 'test':
        csv_file = open(os.path.join(to_save, ftype+'.csv'), 'w+')
        csv_file.write('ID,')
    else:
        csv_file = open(os.path.join(to_save, ftype+'.csv'), 'w+')
        csv_file.write('ID,gt_path,')
    for modality in modalities[:-1]:
        csv_file.write(modality+'_path,')
    modality = modalities[-1]
    csv_file.write(modality+'_path\n')
    folders = os.listdir(folder_path)
    for folder in folders:
        csv_file.write(folder)
        csv_file.write(',')
        if ftype != 'test':
            ground_truth = glob.glob(os.path.join(folder_path, folder,
                                                  '*maskFinal*.nii.gz'))[0]
            csv_file.write(ground_truth)
            csv_file.write(',')
        for modality in modalities[:-1]:
            img = glob.glob(os.path.join(folder_path, folder,
                                         '*'+modality+'*.nii.gz'))[0]
            csv_file.write(img)
            csv_file.write(',')
        modality = modalities[-1]
        img = glob.glob(os.path.join(folder_path, folder,
                                     '*'+modality+'*.nii.gz'))[0]
        csv_file.write(img)
        csv_file.write('\n')
    csv_file.close()


def generate_csv(folder_path, to_save, mode, ftype, modalities):
    """[Function to generate CSV]
    [This function takes a look at the data directory and the modes and
     generates a csv]
    Arguments:
        folder_path {[strin]} -- [description]
        to_save {[strin]} -- [description]
        mode {[string]} -- [description]
        ftype {[string]} -- [description]
        modalities {[string]} -- [description]
    """
    print("Generating", ftype, '.csv')
    if mode == 'one4all':
        rex_o4a_csv(folder_path, to_save, ftype, modalities)
    elif mode == 'single':
        rex_sin_csv(folder_path, to_save, ftype, modalities)
    elif mode == 'multi':
        rex_mul_csv(folder_path, to_save, ftype, modalities)
    else:
        print("Sorry, this mode is not supported")
        sys.exit(0)
