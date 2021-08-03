
from __future__ import absolute_import
import os
import sys
import torch
import numpy as np
from BrainMaGe.models.networks import fetch_model
from pathlib import Path
import matplotlib.pyplot as plt
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from compare_utils import (
    postprocess_prediction,
    postprocess_save_output,
    postprocess_output,
    dice,
    get_mask_image,
    get_input_image
)
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime
import argparse

from openvino.inference_engine import IECore


def bench_ov_fp32(num_cores, device):
    #### Load OpenVINO model
    ov_model_dir = brainmage_root / 'BrainMaGe/weights/ov/fp32'
    modelname = "resunet_ma"

    model_xml = f'{ov_model_dir}/{modelname}.xml'
    model_bin = f'{ov_model_dir}/{modelname}.bin'

    # Load network to the plugin
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    config = {}
    config['CPU_THREADS_NUM'] = str(num_cores)
    exec_net = ie.load_network(network=net, device_name=device, config=config)
    del net

    input_layer = next(iter(exec_net.input_info))
    output_layer = next(iter(exec_net.outputs))

    # # #### Run OpenVINO Inference
    print (f"Starting OpenVINO FP32 inference with {ov_model_dir} ...")

    ov_stats =[]

    for i, row in tqdm(dataset_df.iterrows()):
        sub_id = row[sub_idx]
        input_path = row[input_path_idx]
        mask_path = row[mask_path_idx]

        try:
            mask_image = get_mask_image(mask_path)
            input_image, patient_nib = get_input_image(input_path)

            i_start = timer()
            ov_output = exec_net.infer(inputs={input_layer: input_image})
            i_end = timer()

            p_start = timer()
            ov_output = ov_output[output_layer][0][0]
            ov_to_save = postprocess_output(ov_output, patient_nib.shape)
            ov_dice_score = dice(ov_to_save, mask_image)
            p_end = timer()

            ov_stat = [i, sub_id, ov_dice_score, i_end-i_start, p_end-p_start]
            ov_stats.append(ov_stat)
        except:
            print (f" Inference Failed: {sub_id} ")

    print (f"Done OpenVINO inference with {ov_model_dir} ...")
    ov_stats_df = pd.DataFrame(ov_stats)

    date_time_str = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    csv_name = f"ov_fp32_stats_nc_{num_cores}_{date_time_str}.csv"
    ov_stats_df.to_csv(csv_name, sep=',', header=False, index=False)
    print (f"Saved {csv_name} ...")

    print_stats("FP32", ov_stats_df)

    return ov_stats_df


def bench_ov_int8(num_cores, device):

    # #### Load INT8 OpenVINO model
    ov_model_dir = brainmage_root / 'openvino/int8_openvino_model'
    modelname = "resunet_ma_int8"

    model_xml = f'{ov_model_dir}/{modelname}.xml'
    model_bin = f'{ov_model_dir}/{modelname}.bin'

    # Load network to the plugin
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    config = {}
    config['CPU_THREADS_NUM'] = str(num_cores)
    exec_net = ie.load_network(network=net, device_name=device, config=config)
    del net

    input_layer = next(iter(exec_net.input_info))
    output_layer = next(iter(exec_net.outputs))

    # #### Run OpenVINO Inference
    print (f"Starting OpenVINO inference with {ov_model_dir} ...")

    ov_int8_stats =[]

    for i, row in tqdm(dataset_df.iterrows()):
        sub_id = row[sub_idx]
        input_path = row[input_path_idx]
        mask_path = row[mask_path_idx]

        try:
            mask_image = get_mask_image(mask_path)
            input_image, patient_nib = get_input_image(input_path)

            i_start = timer()
            ov_output = exec_net.infer(inputs={input_layer: input_image})
            i_end = timer()

            p_start = timer()
            ov_output = ov_output[output_layer][0][0]
            ov_to_save = postprocess_output(ov_output, patient_nib.shape)
            ov_dice_score = dice(ov_to_save, mask_image)
            p_end = timer()

            ov_int8_stat = [i, sub_id, ov_dice_score, i_end-i_start, p_end-p_start]
            ov_int8_stats.append(ov_int8_stat)
        except:
            print (f" Inference Failed: {sub_id} ")

    print (f"Done OpenVINO inference with {ov_model_dir} ...")
    ov_int8_stats_df = pd.DataFrame(ov_int8_stats)

    date_time_str = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    csv_name = f"ov_int8_stats_nc_{num_cores}_{date_time_str}.csv"
    ov_int8_stats_df.to_csv(csv_name, sep=',', header=False, index=False)
    print (f"Saved {csv_name} ...")

    print_stats("INT8", ov_int8_stats_df)

    return ov_int8_stats_df


## Print Summary
def print_stats(data_type, stats_df):
    print (f"OpenVINO {data_type} Dice Mean +/- STD: {stats_df[:][2].mean():.5f} +/- {stats_df[:][2].std():.1f}")
    print (f"OpenVINO {data_type}: Total Inf Time (sec) : {stats_df[:][3].sum():.2f} ")
    print (f"OpenVINO {data_type}: Mean Inf Time (sec) +/- STD: {stats_df[:][3].mean():.2f} +/- {stats_df[:][3].std():.2f} ")
    print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--brainmage_root', dest='brainmage_root', default='../../', help='Path to the BrainMaGe root directory')
    parser.add_argument('--data_path', dest='dataset_csv', default='../nfbs-dataset-test-preprocessed.csv', help='Path to dataset used for testing')
    parser.add_argument('--ov_output_path', dest='ov_output_path', default='ov-outfile', help='Path to log results')
    parser.add_argument('--device', dest='device', default='cpu', help='Device to be used for inference')
    parser.add_argument('--nc', dest='num_cores', default=1, help='Number of cores used for inference')
    parser.add_argument('--data_type', dest='data_type', default='FP32', help='Inference datatype')
    parser.add_argument('--sub_idx', dest='sub_idx', default=0, help='0th col is sub-id')
    parser.add_argument('--input_path_idx', dest='input_path_idx', default=1, help='1st col is input path')
    parser.add_argument('--mask_path_idx', dest='mask_path_idx', default=2, help='2nd col is mask_path')

    args = parser.parse_args()

    brainmage_root = Path(args.brainmage_root)
    dataset_csv = args.dataset_csv

    sub_idx = 0 # 0th col is sub-id, for upenn-baseline-dataset 3st col is input path, 2nd col is mask_path
    input_path_idx = args.input_path_idx
    mask_path_idx = args.mask_path_idx

    ov_output_path = 'ov-outfile' # ONNX output file

    device=args.device

    # ### Load Dataset csv
    dataset_df = pd.read_csv(dataset_csv, header = None)
    print(f"Number of rows: {dataset_df.shape[0]}")
    print(f"Input Image Sample: {dataset_df.iloc[0][input_path_idx]}")
    print(f"Mask Image Sample: {dataset_df.iloc[0][mask_path_idx]}")

    ##
    ## Run Benchmark
    ##
    print (f"Starting OpenVINO inference with datatype: {args.data_type} Num Cores: {args.num_cores} ...")

    if 'FP32' in args.data_type:
        ov_fp32_stats_df = bench_ov_fp32(args.num_cores, args.device)

    elif 'INT8' in args.data_type:
        ov_int8_stats_df = bench_ov_int8(args.num_cores, args.device)

