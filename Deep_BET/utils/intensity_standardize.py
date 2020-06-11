#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:49:24 2020

@author: siddhesh
"""


import numpy as np
import os
import glob
import nibabel as nib
import argparse
from utils.preprocess_data import preprocess_image
from multiprocessing import Pool, cpu_count


# """You can change the folder name here. The folders should be in the following
#    format.
# ---main_folder
#     |---Patient1
#          |---something_t1.nii.gz
#          |---something_t2.nii.gz
#          |---something_t1ce.nii.gz
#          |---something_flair.nii.gz
#     |---Patient2
#          |---something_t1.nii.gz
#          |---something_t2.nii.gz
#          |---something_t1ce.nii.gz
#          |---something_flair.nii.gz
#     |---Patient3
#     .
#     .
#     .
#     |---Patient(N)
#
#     *** The files will be generated as something_roimask.nii.gz ***
# """


def normalize(folder, dest_folder, patient_name, test=False):
    """[Function used to pre-process files]
    [This function is used for the skull stripping preprocessing,
     for more details, please visit the paper at : arxiv.org]
    Arguments:
        folder {[string]} -- [The Root folder to look into]
        dest_folder {[type]} -- [The folder to store preprocessed files into]
        test {[type]} -- [If doing it for the testing, we don't want to check
                          for ground truths]
    """
    dest_folder = os.path.join(dest_folder, patient_name)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    t1 = glob.glob(os.path.join(folder, '*t1.nii.gz'))[0]
    t2 = glob.glob(os.path.join(folder, '*t2.nii.gz'))[0]
    t1ce = glob.glob(os.path.join(folder, '*t1ce.nii.gz'))[0]
    flair = glob.glob(os.path.join(folder, '*flair.nii.gz'))[0]
    if not test:
        gt = glob.glob(os.path.join(folder, '*_mask.nii.gz'))[0]

    os.mkdir(os.path.join(dest_folder, folder))
    new_affine = np.array([[1.875, 0, 0],
                           [0, 1.875, 0],
                           [0, 0, 1.25]])

    # Reading T1 image and storing it
    t1_image = nib.load(t1)
    resized_t1_image = preprocess_image(t1_image, is_mask=False)
    temp_affine = t1_image.affine
    temp_affine[:3, :3] = new_affine
    resized_t1_image = nib.Nifti1Image(resized_t1_image, temp_affine)
    nib.save(resized_t1_image, os.path.join(dest_folder, folder, folder +
                                            "_t1.nii.gz"))

    t2_image = nib.load(t2)
    resized_t2_image = preprocess_image(t2_image, is_mask=False)
    temp_affine = t2_image.affine
    temp_affine[:3, :3] = new_affine
    resized_t2_image = nib.Nifti1Image(resized_t2_image, temp_affine)
    nib.save(resized_t2_image, os.path.join(dest_folder, folder, folder +
                                            "_t2.nii.gz"))

    t1ce_image = nib.load(t1ce)
    resized_t1ce_image = preprocess_image(t1ce_image, is_mask=False)
    temp_affine = t1ce_image.affine
    temp_affine[:3, :3] = new_affine
    resized_t1ce_image = nib.Nifti1Image(resized_t1ce_image,
                                         t1ce_image.affine)
    nib.save(resized_t1ce_image, os.path.join(dest_folder, folder, folder +
                                              "_t1ce.nii.gz"))

    flair_image = nib.load(flair)
    resized_flair_image = preprocess_image(flair_image, is_mask=False)
    temp_affine = flair_image.affine
    temp_affine[:3, :3] = new_affine
    resized_flair_image = nib.Nifti1Image(resized_flair_image,
                                          flair_image.affine)
    nib.save(resized_flair_image, os.path.join(dest_folder, folder, folder
                                               +
                                               "_flair.nii.gz"))

    if not test:
        gt_image = nib.load(gt)
        resized_gt_image = preprocess_image(gt_image, is_mask=True)
        resized_gt_image = nib.Nifti1Image(resized_gt_image,
                                           gt_image.affine)
        nib.save(resized_gt_image, os.path.join(dest_folder, folder, folder
                                                +"_mask.nii.gz"))
    return


def batch_works(k):
    if k == n_processes - 1:
        sub_patients = patients[k * int(len(patients) / n_processes):]
    else:
        sub_patients = patients[k * int(len(patients) / n_processes):
                                (k + 1) * int(len(patients) / n_processes)]
    for patient in sub_patients:
        patient_name, _ = os.path.splitext(os.path.basename(patient))
        patient_name.strip('.nii.gz')
        normalize(patient, output_path, patient_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', dest='input_path',
                        help="input path for the tissues", required=True)
    parser.add_argument('-o', '--output_path', dest='output_path',
                        help="output path for saving the files", required=True)
    parser.add_argument('-t', '--threads', dest='threads',
                        help="number of threads, by default will use all")
    args = parser.parse_args()

    if args.threads:
        n_processes = int(args.threads)
    else:
        n_processes = cpu_count()
    print("Number of CPU's used : ", n_processes)

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    patients = glob.glob(os.path.abspath(args.input_path)+'/*')
    n_processes = cpu_count()
    pool = Pool(processes=n_processes)
    pool.map(batch_works, range(n_processes))