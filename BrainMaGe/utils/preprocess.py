#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:26:17 2020

@author: siddhesh
"""


import numpy as np
from skimage.transform import resize


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
        padded_image = np.pad(
            padded_image,
            (
                (int((240 - image.shape[0]) / 2), int((240 - image.shape[0]) / 2)),
                (0, 0),
                (0, 0),
            ),
            mode="constant",
            constant_values=0,
        )
    # Padding on Y axes
    if image.shape[1] < 240:
        # print("Image was padded on the Y-axis on both sides")
        padded_image = np.pad(
            padded_image,
            (
                (0, 0),
                (int((240 - image.shape[1]) / 2), int((240 - image.shape[1]) / 2)),
                (0, 0),
            ),
            mode="constant",
            constant_values=0,
        )
    # Padding on Z axes
    if image.shape[2] < 160:
        # print("Image was padded on the Z-axis on top only")
        padded_image = np.pad(
            padded_image,
            ((0, 0), (0, 0), (0, int(160 - image.shape[2]))),
            "constant",
            constant_values=0,
        )
    return padded_image


def preprocess_image(image, is_mask=False, target_spacing=(1.875, 1.875, 1.25)):
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
                new_image = resize(
                    new_image,
                    (128, 128, 128),
                    order=3,
                    mode="edge",
                    cval=0,
                    anti_aliasing=False,
                )
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
                new_image = resize(
                    new_image,
                    (128, 128, 128),
                    order=3,
                    mode="edge",
                    cval=0,
                    anti_aliasing=False,
                )
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
            new_shape = (
                int(np.round(old_spacing[0] / new_spacing[0] * float(image.shape[0]))),
                int(np.round(old_spacing[1] / new_spacing[1] * float(image.shape[1]))),
                int(np.round(old_spacing[2] / new_spacing[2] * float(image.shape[2]))),
            )
            new_image = resize(
                new_image, new_shape, order=1, mode="edge", cval=0, anti_aliasing=False
            )
            if new_shape == [240, 240, 160]:
                new_image = resize(
                    new_image,
                    (128, 128, 128),
                    order=3,
                    mode="edge",
                    cval=0,
                    anti_aliasing=False,
                )
            else:
                new_image = pad_image(new_image)
                new_image = resize(
                    new_image,
                    (128, 128, 128),
                    order=3,
                    mode="edge",
                    cval=0,
                    anti_aliasing=False,
                )
    else:
        if old_spacing == (1.0, 1.0, 1.0):
            if shape == [240, 240, 160]:
                new_image = resize(
                    new_image,
                    (128, 128, 128),
                    order=0,
                    mode="edge",
                    cval=0,
                    anti_aliasing=False,
                )
            else:
                new_image = pad_image(new_image)
                new_image = resize(
                    new_image,
                    (128, 128, 128),
                    order=0,
                    mode="edge",
                    cval=0,
                    anti_aliasing=False,
                )
        else:
            new_shape = (
                int(np.round(old_spacing[0] / new_spacing[0] * float(image.shape[0]))),
                int(np.round(old_spacing[1] / new_spacing[1] * float(image.shape[1]))),
                int(np.round(old_spacing[2] / new_spacing[2] * float(image.shape[2]))),
            )
            new_image = resize(
                new_image, new_shape, order=0, mode="edge", cval=0, anti_aliasing=False
            )
            if new_shape == [240, 240, 160]:
                new_image = resize(
                    new_image,
                    (128, 128, 128),
                    order=0,
                    mode="edge",
                    cval=0,
                    anti_aliasing=False,
                )
            else:
                new_image = pad_image(new_image)
                new_image = resize(
                    new_image,
                    (128, 128, 128),
                    order=0,
                    mode="edge",
                    cval=0,
                    anti_aliasing=False,
                )

    if is_mask:  # Retrun if mask
        return new_image.astype(np.int8)
    else:
        new_image_temp = new_image[new_image >= new_image.mean()]
        p1 = np.percentile(new_image_temp, 2)
        p2 = np.percentile(new_image_temp, 95)
        new_image[new_image > p2] = p2
        new_image = (new_image - p1) / p2
        return new_image.astype(np.float32)
