#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:44:32 2020

@author: siddhesh
"""


import torch.optim as optim
import sys


def fetch_optimizer(optimizer, lr, model):
    # Setting up the optimizer
    if optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=float(lr), momentum=0.9, nesterov=True
        )
    elif optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=float(lr), betas=(0.9, 0.999), weight_decay=0.00005
        )
    elif optimizer.lower() == "rms":
        optimizer = optim.RMSprop(
            model.parameters(), lr=float(lr), momentum=0.9, weight_decay=0.00005
        )
    elif optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(), lr=float(lr), weight_decay=0.00005
        )
    else:
        print(
            "Sorry, {} is not supported or some sort of spell error. Please\
               choose from the given options!".format(
                optimizer
            )
        )
        sys.stdout.flush()
        sys.exit(0)
    return optimizer
