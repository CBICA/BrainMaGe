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
from skimage.transform import resize
from multiprocessing import Pool, cpu_count
import pkg_resources


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

def pad_image(image):
    """[To pad the image to particular space]
    [This function will pad the image to a space of [240, 240, 160] and will
    automatically pad everything with zeros to this size]
    Arguments:
        image {[numpy image]} -- [Image of any standard shape[x, y. z]]
    Returns:
        [padded_image] -- [returns a padded image]
    """
    padded_image = image
    # Padding on X axes
    if image.shape[0] < 240:
        # print("Image was padded on the X-axis on both sides")
        padded_image = np.pad(padded_image, ((int((240-image.shape[0])/2),
                                              int((240-image.shape[0])/2)),
                                             (0, 0), (0, 0)),
                              mode='constant', constant_values=0)
    # Padding on Y axes
    if image.shape[1] < 240:
        # print("Image was padded on the Y-axis on both sides")
        padded_image = np.pad(padded_image, ((0, 0),
                                             (int((240-image.shape[1])/2),
                                              int((240-image.shape[1])/2)),
                                             (0, 0)),
                              mode='constant', constant_values=0)
    # Padding on Z axes
    if image.shape[2] < 160:
        # print("Image was padded on the Z-axis on top only")
        padded_image = np.pad(padded_image, ((0, 0), (0, 0),
                                             (0, int(160-image.shape[2]))),
                              'constant', constant_values=0)
    return padded_image


def preprocess_image(image, is_mask=False,
                     target_spacing=(1.875, 1.875, 1.25)):
    """[To preprocess an image depending on whether it a mask image or not]
    [This function in general will try to preprocess a given image to a partic-
    -ular image resolution and try to return a preprocessed image]
    Arguments:
        image {[nibabel image]} -- [Expecting a nibabel image to be handled]
    Keyword Arguments:
        is_mask {bool} -- [If the incoming image is a mask] (default: {False})
        target_spacing {tuple} -- [What should be a current given target
                                   spacing to be used]
                                  (default: {(1.875, 1.875, 1.25)})
    Returns:
        [preprocessed image] -- [Returning a properly preprocessed and a norma-
        -lized image]
    """
    old_spacing = image.header.get_zooms()
    shape = image.header.get_data_shape()
    new_image = image.get_fdata()
    new_spacing = (1, 1, 1)
    # Check if it is a normal image or a mask
    # If this thing is normal image
    if not is_mask:
        if old_spacing == (1.0, 1.0, 1.0):
            if shape == [240, 240, 160]:
                # print("Image is perfect?")
                """[Checking if it is an ideal image]
                _________________________________________
                ___________|_Correct_|_Incorrect_|______|
                shape      |   Yes   |     No    |     _|
                resolution_|___Yes___|_____No____|______|
                pad________|_________|___________|__No__|
                [An ideal image would be to have a shape of (240, 240, 160)
                with an isotropic resolution of (1.0, 1.0, 1.0), then we would
                just resize the image to (128, 128, 128)]
                """
                new_image = resize(new_image, (128, 128, 128), order=3,
                                   mode='edge', cval=0, anti_aliasing=False)
            else:
                """[Checking if it is an isotropic image with need to incorrect
                shape]
                ________________________________________
                ___________|_Correct_|_Incorrect_|_____|
                shape      |   No    |     Yes   |     |
                resolution_|___Yes___|_____No____|_____|
                pad________|_________|___________|_Yes_|
                [An ideal image would be to have a shape of (240, 240, 160)
                with a isotropic resolution of (1.0, 1.0, 1.0), then we would
                just resize the image to (128, 128, 128)]
                """
                # print("Image shape wasn't perfect")
                new_image = pad_image(new_image)
                # print("Trying to pad the image now")
                new_image = resize(new_image, (128, 128, 128), order=3,
                                   mode='edge', cval=0, anti_aliasing=False)
        else:
            """[Checking if it is not isotropic image with resolution needed]
            ________________________________________
            ___________|_Correct_|_Incorrect_|_____|
            shape      |   No    |     Yes   |     |
            resolution_|___Yes___|_____No____|_____|
            pad________|_________|___________|_Yes_|
            [An ideal image would be to have a shape of (240, 240, 160) with
            a isotropic resolution of (1.0, 1.0, 1.0), then we would just
            resize the image to (128, 128, 128)]
            """
            new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                         int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                         int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
            new_image = resize(new_image, new_shape, order=1,
                               mode='edge', cval=0, anti_aliasing=False)
            if new_shape == [240, 240, 160]:
                new_image = resize(new_image, (128, 128, 128), order=3,
                                   mode='edge', cval=0, anti_aliasing=False)
            else:
                new_image = pad_image(new_image)
                new_image = resize(new_image, (128, 128, 128), order=3,
                                   mode='edge', cval=0, anti_aliasing=False)
    else:
        if old_spacing == (1.0, 1.0, 1.0):
            if shape == [240, 240, 160]:
                new_image = resize(new_image, (128, 128, 128), order=0,
                                   mode='edge', cval=0, anti_aliasing=False)
            else:
                new_image = pad_image(new_image)
                new_image = resize(new_image, (128, 128, 128), order=0,
                                   mode='edge', cval=0, anti_aliasing=False)
        else:
            new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                         int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                         int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
            new_image = resize(new_image, new_shape, order=0,
                               mode='edge', cval=0, anti_aliasing=False)
            if new_shape == [240, 240, 160]:
                new_image = resize(new_image, (128, 128, 128), order=0,
                                   mode='edge', cval=0, anti_aliasing=False)
            else:
                new_image = pad_image(new_image)
                new_image = resize(new_image, (128, 128, 128), order=0,
                                   mode='edge', cval=0, anti_aliasing=False)

    if is_mask:  # Retrun if mask
        return new_image.astype(np.int8)
    else:
        new_image_temp = new_image[new_image >= new_image.mean()]
        p1 = np.percentile(new_image_temp, 2)
        p2 = np.percentile(new_image_temp, 95)
        new_image[new_image > p2] = p2
        new_image = (new_image - p1)/p2
        return new_image.astype(np.float32)

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
    patient_dest_folder = os.path.join(dest_folder, patient_name)
    os.makedirs(patient_dest_folder, exist_ok=True)
    t1 = glob.glob(os.path.join(folder, '*t1.nii.gz'))[0]
    t2 = glob.glob(os.path.join(folder, '*t2.nii.gz'))[0]
    t1ce = glob.glob(os.path.join(folder, '*t1ce.nii.gz'))[0]
    flair = glob.glob(os.path.join(folder, '*flair.nii.gz'))[0]
    if not test:
        gt = glob.glob(os.path.join(folder, '*mask.nii.gz'))[0]

    new_affine = np.array([[1.875, 0, 0],
                           [0, 1.875, 0],
                           [0, 0, 1.25]])

    # Reading T1 image and storing it
    t1_image = nib.load(t1)
    resized_t1_image = preprocess_image(t1_image, is_mask=False)
    temp_affine = t1_image.affine
    temp_affine[:3, :3] = new_affine
    resized_t1_image = nib.Nifti1Image(resized_t1_image, temp_affine)
    print(patient_dest_folder)
    print("Saving T1 at : ", os.path.join(patient_dest_folder, patient_name +
                                            "_t1.nii.gz"))
    nib.save(resized_t1_image, os.path.join(patient_dest_folder, patient_name +
                                            "_t1.nii.gz"))

    t2_image = nib.load(t2)
    resized_t2_image = preprocess_image(t2_image, is_mask=False)
    temp_affine = t2_image.affine
    temp_affine[:3, :3] = new_affine
    resized_t2_image = nib.Nifti1Image(resized_t2_image, temp_affine)
    nib.save(resized_t2_image, os.path.join(patient_dest_folder, patient_name +
                                            "_t2.nii.gz"))

    t1ce_image = nib.load(t1ce)
    resized_t1ce_image = preprocess_image(t1ce_image, is_mask=False)
    temp_affine = t1ce_image.affine
    temp_affine[:3, :3] = new_affine
    resized_t1ce_image = nib.Nifti1Image(resized_t1ce_image,
                                         t1ce_image.affine)
    nib.save(resized_t1ce_image, os.path.join(patient_dest_folder, patient_name +
                                              "_t1ce.nii.gz"))

    flair_image = nib.load(flair)
    resized_flair_image = preprocess_image(flair_image, is_mask=False)
    temp_affine = flair_image.affine
    temp_affine[:3, :3] = new_affine
    resized_flair_image = nib.Nifti1Image(resized_flair_image,
                                          flair_image.affine)
    nib.save(resized_flair_image, os.path.join(patient_dest_folder, patient_name
                                               +
                                               "_flair.nii.gz"))

    if not test:
        gt_image = nib.load(gt)
        resized_gt_image = preprocess_image(gt_image, is_mask=True)
        resized_gt_image = nib.Nifti1Image(resized_gt_image,
                                           gt_image.affine)
        nib.save(resized_gt_image, os.path.join(patient_dest_folder, patient_name
                                                +"_mask.nii.gz"))
    return


def batch_works(k):
    if k == n_processes - 1:
        sub_patients = patients[k * int(len(patients) / n_processes):]
    else:
        sub_patients = patients[k * int(len(patients) / n_processes):
                                (k + 1) * int(len(patients) / n_processes)]
    for patient in sub_patients:
        patient_name = os.path.basename(patient)
        print(patient_name)
        normalize(patient, output_path, patient_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='intensity_standardize', formatter_class=argparse.RawTextHelpFormatter,
                                     description='\nThis code was implemented to standardize intensities for skull stripping\n'+ '\n'\
    'Copyright: Center for Biomedical Image Computing and Analytics (CBICA), University of Pennsylvania.\n'\
    'For questions and feedback contact: software@cbica.upenn.edu')

    parser.add_argument('-i', '--input_path', dest='input_path',
                        help="input path for the tissues", required=True)
    parser.add_argument('-o', '--output_path', dest='output_path',
                        help="output path for saving the files", required=True)
    parser.add_argument('-t', '--threads', dest='threads',
                        help="number of threads, by default will use all")
                        
    parser.add_argument('-v', '--version', action='version',
                        version=pkg_resources.require("BrainMaGe")[0].version + '\n\nCopyright: Center for Biomedical Image Computing and Analytics (CBICA), University of Pennsylvania.', help="Show program's version number and exit.")
                
    args = parser.parse_args()

    if args.threads:
        n_processes = int(args.threads)
    else:
        n_processes = cpu_count()
    print("Number of CPU's used : ", n_processes)

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    os.makedirs(output_path, exist_ok=True)
    patients = glob.glob(os.path.abspath(args.input_path)+'/*')
    n_processes = cpu_count()
    pool = Pool(processes=n_processes)
    pool.map(batch_works, range(n_processes))
