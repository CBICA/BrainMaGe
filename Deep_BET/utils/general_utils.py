#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 04:37:59 2020

@author: siddhesh
"""

import os
from skimage.morphology import label
import numpy as np


def postprocess_prediction(seg):
    mask = seg != 0
    lbls = label(mask, 8)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region] = 0
    return seg
