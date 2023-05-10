#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:44:32 2020

@author: siddhesh
"""



import torch.optim as optim
import sys


def fetch_optimizer(optimizer, lr, model):
    # Mapping optimizer name to class
    optimizer_map = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "rms": optim.RMSprop,
        "adagrad": optim.Adagrad,
    }

    # Setting up the optimizer
    optimizer_class = optimizer_map.get(optimizer.lower())
    if optimizer_class is None:
        print(f"Sorry, {optimizer} is not supported. Please choose from the given options!")
        sys.exit(1)
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(lr),
        **({"momentum": 0.9, "nesterov": True} if optimizer.lower() == "sgd" else {}),
        **({"betas": (0.9, 0.999), "weight_decay": 0.00005} if optimizer.lower() == "adam" else {}),
        **({"momentum": 0.9, "weight_decay": 0.00005} if optimizer.lower() == "rms" else {}),
        **({"weight_decay": 0.00005} if optimizer.lower() == "adagrad" else {}),
    )
    return optimizer
