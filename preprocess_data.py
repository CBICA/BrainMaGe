#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:42:48 2019

@author: siddhesh
"""

import SimpleITK as sitk
import numpy as np
from skimage.transform import resize

from scipy.misc import imresize

def pad_image(image):
    padded_image = image
    #Padding on X axes
    if image.shape[0] < 240:
        padded_image = np.pad(padded_image, ((int((240-image.shape[0])/2), int((240-image.shape[0])/2)), (0, 0), (0, 0)), 'constant', constant_values = 0)
    #Padding on Y axes
    if image.shape[0] < 240:
        padded_image = np.pad(padded_image, ((0, 0), (int((240-image.shape[0])/2), int((240-image.shape[0])/2)), (0, 0)), 'constant', constant_values = 0)
    #Padding on Z axes
    if image.shape[0] < 160:
        padded_image = np.pad(padded_image, ((0, 0), (0, 0), (int(240-image.shape[0]), 0)), 'constant', constant_values = 0)
    return padded_image

def preprocess_image(image, is_mask = False, target_spacing = (1.875, 1.875, 1.25)):
    """
    Expecting a nibabel type image
    """
    #image_np = np.pad(image.get_fdata(), ((0, 0), (0, 0), (5, 0)))
    old_spacing = image.get_zooms()
    shape = image.get_data_shape()
    new_image = image.get_fdata()
    new_spacing = (1, 1, 1)
    if not is_mask:
        if old_spacing == (1.0, 1.0, 1.0):
            if shape == [240, 240, 160]:
                new_image = resize(new_image, (128, 128, 128), order = 3, mode = 'edge', cval =0, anti_aliasing = False)
            else:
                new_image = pad_image(new_image)
                new_image = resize(new_image, (128, 128, 128), order = 3, mode = 'edge', cval =0, anti_aliasing = False)
        else:
            new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                     int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                     int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
            new_image = resize(new_image, new_shape, interp = 'trilinear', mode = 'edge', cval =0, anti_aliasing = False)
            if new_shape == [240, 240, 160]:
                new_image = resize(new_image, (128, 128, 128), order = 3, mode = 'edge', cval =0, anti_aliasing = False)
            else:
                new_image = pad_image(new_image)
                new_image = resize(new_image, (128, 128, 128), order = 3, mode = 'edge', cval =0, anti_aliasing = False)
    else:
        if old_spacing == (1.0, 1.0, 1.0):
            if shape == [240, 240, 160]:
                new_image = resize(new_image, (128, 128, 128), order = 0, mode = 'edge', cval =0, anti_aliasing = False)
            else:
                new_image = pad_image(new_image)
                new_image = resize(new_image, (128, 128, 128), order = 0, mode = 'edge', cval =0, anti_aliasing = False)
        else:
            new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                     int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                     int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
            new_image = resize(new_image, new_shape, interp = 'trilinear', mode = 'edge', cval =0, anti_aliasing = False)
            if new_shape == [240, 240, 160]:
                new_image = resize(new_image, (128, 128, 128), order = 0, mode = 'edge', cval =0, anti_aliasing = False)
            else:
                new_image = pad_image(new_image)
                new_image = resize(new_image, (128, 128, 128), order = 0, mode = 'edge', cval =0, anti_aliasing = False)

    if is_mask: # Retrun if mask
        return new_image.astype(np.float32)
    else:
        new_image_temp = new_image[new_image>new_image.mean()]
        p1 = np.percentile(new_image_temp, 2)
        p2 = np.percentile(new_image_temp, 95)
        new_image_temp[new_image_temp<p1] = p1
        new_image_temp[new_image_temp>p2] = p2
        new_image_temp = (new_image_temp - p1)/p2
        return new_image_temp.astype(np.float32)
    
    #
#def pad_image()
def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation, new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            reshaped_multihot = resize((segmentation == c).astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge', cval=0, anti_aliasing=False)


def preprocess_image(itk_image, is_seg=False, spacing_target=(1, 0.5, 0.5)):
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    image = sitk.GetArrayFromImage(itk_image).astype(float)

    assert len(image.shape) == 3, "The image has unsupported number of dimensions. Only 3D images are allowed"

    if not is_seg:
        if np.any([[i != j] for i, j in zip(spacing, spacing_target)]):
            image = resize_image(image, spacing, spacing_target).astype(np.float32)

        image -= image.mean()
        image /= image.std()
    else:
        new_shape = (int(np.round(spacing[0] / spacing_target[0] * float(image.shape[0]))),
                     int(np.round(spacing[1] / spacing_target[1] * float(image.shape[1]))),
                     int(np.round(spacing[2] / spacing_target[2] * float(image.shape[2]))))
        image = resize_segmentation(image, new_shape, 1)
    return image