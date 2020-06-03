#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 01:05:59 2020

@author: siddhesh
"""


from setuptools import setup

setup(name='Penn-BET',
      version='1.0.0.Alpha',
      description='Skull stripping using multiple and single modalities',
      url='https://github.com/CBICA/Penn-BET',
      python_requires='>=3.6',
      author='Siddhesh Thakur',
      author_email='software@cbica.upenn.edu',
      license='Apache 2.0',
      zip_safe=False,
      install_requires=[
      'numpy',
      'torch>=1.0.1',
      'scikit-image',
      'nibabel',
      'pytorch-lightning'
      ],
      scripts=['penn_bet_run'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Operating System :: Unix'
      ]
      )
