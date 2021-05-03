#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 01:05:59 2020

@author: siddhesh
"""


from setuptools import setup
import setuptools

setup(
    name="BrainMaGe",
    version="1.0.5-dev",
    description="Skull stripping using multiple and single modalities",
    url="https://github.com/CBICA/BrainMaGe",
    python_requires=">=3.6",
    author="Siddhesh Thakur",
    author_email="software@cbica.upenn.edu",
    license="BSD-3-Clause",
    zip_safe=False,
    install_requires=[
        "numpy",
        "torch>=1.5.1",
        "scikit-image",
        "nibabel",
        "pytorch-lightning==0.8.1",
        "bids"
    ],
    scripts=[
        "brain_mage_run",
        "brain_mage_single_run",
        "brain_mage_intensity_standardize",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
