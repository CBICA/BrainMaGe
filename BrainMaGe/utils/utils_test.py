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
        Input image as a numpy array.

    Returns
    -------
    tuple
        Padded image and its padding information as a tuple.

    """
    pad_info = ((0, 0), (0, 0), (0, 0))
    for i, dim in enumerate(image.shape):
        if dim < (240 if i != 2 else 160):
            pad_size = (240 if i != 2 else 160) - dim
            pad_before = pad_size // 2
            pad_after = pad_size - pad_before
            pad_info[i] = (pad_before, pad_after)
    padded_image = np.pad(
        image, pad_info, mode="constant", constant_values=0)
    return padded_image, pad_info


def process_image(image):
    """
    special percentile based preprocessing and then apply the stuff on image,
    check paper

    Parameters
    ----------
    image : ndarray
        Input image as a numpy array.

    Returns
    -------
    ndarray
        Preprocessed image as a numpy array.

    """
    p1, p2 = np.percentile(image[image >= image.mean()], [2, 95])
    image = np.clip(image, None, p2)
    image = (image - p1) / p2
    return image


def padder_and_cropper(image, pad_info):
    (pad_x1, pad_x2), (pad_y1, pad_y2), (pad_z1, pad_z2) = pad_info
    x_start, x_end = pad_x1, image.shape[0] - pad_x2 if pad_x2 != 0 else None
    y_start, y_end = pad_y1, image.shape[1] - pad_y2 if pad_y2 != 0 else None
    z_start, z_end = pad_z1, image.shape[2] - pad_z2 if pad_z2 != 0 else None
    return image[x_start:x_end, y_start:y_end, z_start:z_end]


def unpad_image(image):
    return image[:, :, :155]


def interpolate_image(image, output_shape):
    return resize(image, output_shape, order=3, mode="edge", cval=0)
