#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 00:37:15 2020

@author: siddhesh
"""

import numpy as np
from skimage.transform import resize


def pad_image(image):
    """

    Parameters
    ----------
    image : ndarray
        DESCRIPTION.

    Returns
    -------
    TYPE
        padded image and its information

    """
    padded_image = image
    pad_x1, pad_x2, pad_y1, pad_y2, pad_z1, pad_z2 = 0, 0, 0, 0, 0, 0
    # Padding on X axes
    if image.shape[0] <= 240:
        pad_x1 = (240 - image.shape[0]) // 2
        pad_x2 = 240 - image.shape[0] - pad_x1
        padded_image = np.pad(
            padded_image,
            ((pad_x1, pad_x2), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # Padding on Y axes
    if image.shape[1] <= 240:
        pad_y1 = (240 - image.shape[1]) // 2
        pad_y2 = 240 - image.shape[1] - pad_y1
        padded_image = np.pad(
            padded_image,
            ((0, 0), (pad_y1, pad_y2), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    # Padding on Z axes
    if image.shape[2] <= 160:
        pad_z2 = 160 - image.shape[2]
        padded_image = np.pad(
            padded_image,
            ((0, 0), (0, 0), (pad_z2, 0)),
            mode="constant",
            constant_values=0,
        )

    return padded_image, ((pad_x1, pad_x2), (pad_y1, pad_y2), (pad_z1, pad_z2))


def process_image(image):
    """
    special percentile based preprocessing and then apply the stuff on image,
    check paper

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    """
    new_image_temp = image[image >= image.mean()]
    p1 = np.percentile(new_image_temp, 2)
    p2 = np.percentile(new_image_temp, 95)
    image[image > p2] = p2
    image = (image - p1) / p2
    return image


def padder_and_cropper(image, pad_info):
    (pad_x1, pad_x2), (pad_y1, pad_y2), (pad_z1, pad_z2) = pad_info
    if pad_x2 == 0:
        pad_x2 = -image.shape[0]
    if pad_y2 == 0:
        pad_y2 = -image.shape[1]
    if pad_z2 == 0:
        pad_z2 = -image.shape[2]
    image = image[pad_x1:-pad_x2, pad_y1:-pad_y2, pad_z2:]
    return image


def unpad_image(image):
    image = image[:, :, :155]
    return image


def interpolate_image(image, output_shape):
    new_image = resize(image, (output_shape), order=3, mode="edge", cval=0)
    return new_image
