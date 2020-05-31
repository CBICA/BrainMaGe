#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:15:25 2020

@author: siddhesh
"""

import torch.nn as nn

def dice_loss(inp, target):
    smooth = 1e-7
    iflat = inp.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def dice(inp, target):
    smooth = 1e-7
    iflat = inp.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2*intersection+smooth)/(iflat.sum()+tflat.sum()+smooth)

def tversky(inp, target, alpha):
    smooth = 1e-7
    iflat = inp.view(-1)
    tflat = target.view(-1)
    intersection = (iflat*tflat).sum()
    fps = (iflat * (1-tflat)).sum()
    fns = ((1-iflat) * tflat).sum()
    denominator = intersection + (alpha*fps) + ((1-alpha)*fns) + smooth
    return (intersection+smooth)/denominator

def tversky_loss(inp, target, alpha):
    smooth = 1e-7
    iflat = inp.view(-1)
    tflat = inp.view(-1)
    intersection = (iflat*tflat).sum()
    fps = (inp * (1-target)).sum()
    fns = (inp * (1-target)).sum()
    denominator = intersection + (alpha*fps) + ((1-alpha)*fns) + smooth
    return 1 - ((intersection+smooth)/denominator)

def power_loss(inp, target, power):
    return dice_loss(inp, target) ** power

def pointwise_loss(inp, target, alpha=20, beta=3):
    iflat = inp.view(-1)
    tflat = target[:, 0, :, :].contiguous().view(-1)
    intersection = (alpha*(iflat * tflat).pow(beta)).sum()
    union = iflat.sum() + tflat.sum()
    return 1 - intersection/union


def bce_loss(inp, target):
    return nn.BCELoss(inp.view(-1), target.view(-1))

def focal_tversky_loss(inp, target, alpha=0.3, gamma=4/3):
    tver = tversky(inp, target, alpha)
    tver_loss = 1-tver
    return (tver_loss)**(1/gamma)

def ft_bce_loss(inp, target, alpha=0.3, gamma=4/3):
    tver = tversky(inp, target, alpha)
    tver_loss = 1-tver
    return 0.5*((tver_loss)**(1/gamma) + nn.BCELoss(inp.view(-1), target.view(-1)))
