#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

import os
import numpy as np
import json
import cv2 as cv
import glob
from addict import Dict
from math import ceil

import sys
import torch
from BrainMaGe.models.networks import fetch_model
from pathlib import Path
import pandas as pd
from compare_utils import (
    postprocess_prediction,
    postprocess_output,
    dice,
    get_mask_image,
    get_input_image
)

from compression.api import Metric, DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline
from compression.utils.logger import init_logger
import compression

os.environ["KMP_WARNINGS"] = "FALSE"

import argparse

brainmage_root = "../"

parser = argparse.ArgumentParser(
    description="Quantizes an OpenVINO model to INT8.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--xml_file", default=brainmage_root+"BrainMaGe/weights/ov/fp32/resunet_ma.xml",
                    help="XML file for OpenVINO to quantize")
parser.add_argument("--bin_file", default=brainmage_root+"BrainMaGe/weights/ov/fp32/resunet_ma.bin",
                help="BIN file for OpenVINO to quantize")
parser.add_argument("--manifest", default=brainmage_root+"openvino/nfbs-dataset-test-preprocessed.csv",
                help="Manifest file (CSV with filenames of images and labels)")
parser.add_argument("--data_dir", default="./data",
                help="Data directory root")
parser.add_argument("--int8_directory", default="./int8_openvino_model",
                help="INT8 directory for calibrated OpenVINO model")
parser.add_argument("--maximum_metric_drop", default=0.03,
                help="AccuracyAwareQuantization: Maximum allowed drop in metric")
parser.add_argument("--accuracy_aware_quantization",
                    help="use accuracy aware quantization",
                    action="store_true", default=True)

args = parser.parse_args()

image_path_idx = 1 # 0th col is sub-id, 1st col is input path, 2nd col is mask_path
mask_path_idx = 2

class bcolors:
    """
    Just gives us some colors for the text
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MyDataLoader(DataLoader):

    def __init__(self, config):

        super().__init__(config)

        """
        The assumption here is that a manifest file (CSV file) is
        passed to the object. The manifest contains a comma-separated
        list of the image filenames and label filenames for the validation
        dataset. You can modify this code to fit your needs. (For example,
        you could have the image filename and the class if using a
        classification model or the image filename and the object
        bounding boxes if using a localization model).
        """

        self.manifest = config["manifest"]  # Filename for manifest file with image and label filenames
        self.images = []
        self.labels = []
        
        dataset_df = pd.read_csv(self.manifest, header = None)
        
        for i, row in dataset_df.iterrows():
            self.images.append(row[image_path_idx]) #image path
            self.labels.append(row[mask_path_idx]) #mask path
        
        self.items = np.arange(dataset_df.shape[0])
        self.batch_size = 1

        print(bcolors.UNDERLINE + "\nQuantizing FP32 OpenVINO model to INT8\n" + bcolors.ENDC)

        print(bcolors.OKBLUE + "There are {:,} samples in the test dataset ".format(len(self.items)) + \
            bcolors.OKGREEN + "{}\n".format(self.manifest) + bcolors.ENDC)



    def set_subset(self, indices):
        self._subset = None

    @property
    def batch_num(self):
        return ceil(self.size / self.batch_size)

    @property
    def size(self):
        return self.items.shape[0]

    def __len__(self):
        return self.size

    def myPreprocess(self, image_filename, label_filename):
        """
        Custom code to preprocess input data
        For this example, we show how to process the brain tumor data.
        Change this to preprocess you data as necessary.
        """
 

        """
        Load the image and label for this item
        """
        msk = get_mask_image(label_filename)
        img, patient_nib = get_input_image(image_filename)


        return img, msk

    def __getitem__(self, item):
        """
        Iterator to grab the data.
        If the data is too large to fit into memory, then
        you can have the item be a filename to load for the input
        and the label.

        In this example, we use the myPreprocess function above to
        do any custom preprocessing of the input.
        """

        # Load the iage and label files for this item
        image_filename = self.images[self.items[item]]
        label_filename = self.labels[self.items[item]]

        image, label = self.myPreprocess(image_filename, label_filename)

        # IMPORTANT!
        # OpenVINO expects channels first so transpose channels to first dimension
#         image = np.transpose(image, [3,0,1,2]) # Channels first
#         label = np.transpose(label, [3,0,1,2]) # Channels first

        return (item, label), image

class MyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.name = "custom Metric - Dice score"
        self._values = []
        self.round = 1

    @property
    def value(self):
        """ Returns accuracy metric value for the last model output. """
        return {self.name: [self._values[-1]]}

    @property
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """
        value = np.ravel(self._values).mean()
        print("Round #{}    Mean {} = {}".format(self.round, self.name, value))

        self.round += 1

        return {self.name: value}

    def update(self, outputs, labels):
        """ Updates prediction matches.

        Args:
            outputs: model output
            labels: annotations

        Put your post-processing code here.
        Put your custom metric code here.
        The metric gets appended to the list of metric values
        """
        try:
            pt_output = outputs[0][0][0]
            patient_nib_shape = labels[0].shape
            pt_to_save = postprocess_output(pt_output, patient_nib_shape)

            metric = dice(labels[0], pt_to_save)
            self._values.append(metric)
        except:
            print (f" Inference Failed. ")

    def reset(self):
        """ Resets collected matches """
        self._values = []

    @property
    def higher_better(self):
        """Attribute whether the metric should be increased"""
        return True

    def get_attributes(self):
        return {self.name: {"direction": "higher-better", "type": ""}}

model_config = Dict({
    "model_name": "resunet_ma",
    "model": args.xml_file,
    "weights": args.bin_file
})

engine_config = Dict({
    "device": "CPU",
    "stat_requests_number": 4,
    "eval_requests_number": 4
})

dataset_config = {
    "manifest": args.manifest,
    "images": "image",
    "labels": "label"
}

default_quantization_algorithm = [
    {
        "name": "DefaultQuantization",
        "params": {
            "target_device": "CPU",
            "preset": "performance",
            #"stat_subset_size": 10
        }
    }
]


# accuracy_aware_quantization_algorithm = [
#     {
#         "name": "AccuracyAwareQuantization", # compression algorithm name
#         "params": {
#             "target_device": "CPU",
#             "preset": "performance",
#             "stat_subset_size": 10,
#             "metric_subset_ratio": 0.5, # A part of the validation set that is used to compare full-precision and quantized models
#             "ranking_subset_size": 300, # A size of a subset which is used to rank layers by their contribution to the accuracy drop
#             "max_iter_num": 10,    # Maximum number of iterations of the algorithm (maximum of layers that may be reverted back to full-precision)
#             "maximal_drop": args.maximum_metric_drop,      # Maximum metric drop which has to be achieved after the quantization
#             "drop_type": "absolute",    # Drop type of the accuracy metric: relative or absolute (default)
#             "use_prev_if_drop_increase": True,     # Whether to use NN snapshot from the previous algorithm iteration in case if drop increases
#             "base_algorithm": "DefaultQuantization" # Base algorithm that is used to quantize model at the beginning
#         }
#     }
# ]
        
accuracy_aware_quantization_algorithm = [
    {
        "name": "AccuracyAwareQuantization", # compression algorithm name
        "params": {
            "target_device": "CPU",
            "preset": "performance",
            "ranking_subset_size": 300, # A size of a subset which is used to rank layers by their contribution to the accuracy drop
            "max_iter_num": 30,    # Maximum number of iterations of the algorithm (maximum of layers that may be reverted back to full-precision)
            "maximal_drop": args.maximum_metric_drop,      # Maximum accuracy drop which has to be achieved after the quantization
            "drop_type": "absolute",    # Drop type of the accuracy metric: relative or absolute (default)
            "use_prev_if_drop_increase": False,      # Whether to use NN snapshot from the previous algorithm iteration in case if drop increases
            "base_algorithm": "DefaultQuantization", # Base algorithm that is used to quantize model at the beginning
            "convert_to_mixed_preset": True,  # Whether to convert the model to mixed mode if the accuracy criteria 
                                               # of the symmetrically quantized model are not satisfied
            "metric_subset_ratio": 0.5  # A part of the validation set that is used to compare element-wise full-precision and 
                                        # quantized models in case of predefined metric values of the original model
         }
    }
   
]

class GraphAttrs(object):
    def __init__(self):
        self.keep_quantize_ops_in_IR = True
        self.keep_shape_ops = False
        self.data_type = "FP32"
        self.progress = False
        self.generate_experimental_IR_V10 = True
        self.blobs_as_inputs = True
        self.generate_deprecated_IR_V7 = False


model = load_model(model_config)

data_loader = MyDataLoader(dataset_config)
metric = MyMetric()


engine = IEEngine(engine_config, data_loader, metric)

if args.accuracy_aware_quantization:
    # https://docs.openvinotoolkit.org/latest/_compression_algorithms_quantization_accuracy_aware_README.html
    print(bcolors.BOLD + "Accuracy-aware quantization method" + bcolors.ENDC)
    pipeline = create_pipeline(accuracy_aware_quantization_algorithm, engine)
else:
    print(bcolors.BOLD + "Default quantization method" + bcolors.ENDC)
    pipeline = create_pipeline(default_quantization_algorithm, engine)


metric_results_FP32 = pipeline.evaluate(model)

compressed_model = pipeline.run(model)

compression.graph.model_utils.save_model(compressed_model, save_path=args.int8_directory, model_name="resunet_ma_int8", for_stat_collection=False)

print(bcolors.BOLD + "\nThe INT8 version of the model has been saved to the directory ".format(args.int8_directory) + \
    bcolors.HEADER + "{}\n".format(args.int8_directory) + bcolors.ENDC)

#save_model(compressed_model, "./int8_openvino_model/")

print(bcolors.BOLD + "\Evaluating INT8 Model..." + bcolors.ENDC)

metric_results_INT8 = pipeline.evaluate(compressed_model)

# print metric value
if metric_results_FP32:
    for name, value in metric_results_FP32.items():
        print(bcolors.OKGREEN + "{: <27s} FP32: {}".format(name, value) + bcolors.ENDC)

if metric_results_INT8:
    for name, value in metric_results_INT8.items():
        print(bcolors.OKBLUE + "{: <27s} INT8: {}".format(name, value) + bcolors.ENDC)


