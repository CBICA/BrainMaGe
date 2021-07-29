
import os
import sys
import torch
import numpy as np
from BrainMaGe.models.networks import fetch_model
from pathlib import Path
import matplotlib.pyplot as plt
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

from openvino.inference_engine import IECore

brainmage_root = Path('../')

# dataset_csv = 'nfbs-dataset-test-1.csv'
# dataset_csv = 'nfbs-dataset-preprocessed.csv'
# dataset_csv = 'nfbs-dataset-test-preprocessed.csv'

# dataset_csv = 'upenn-baseline-dataset.csv'
# dataset_csv = 'upenn-baseline-dataset-test.csv'
# dataset_csv = 'upenn-baseline-dataset-test-10.csv'
dataset_csv = 'upenn-baseline-dataset-test-2.csv'

sub_idx = 0 # 0th col is sub-id, 1st col is input path, 2nd col is mask_path
input_path_idx = 3
mask_path_idx = 2

pt_output_path = 'pt-outfile' # PyTorch output file
ov_output_path = 'ov-outfile' # ONNX output file

pytorch_model_path = brainmage_root / 'BrainMaGe/weights/resunet_ma.pt'
ov_model_dir = brainmage_root / 'BrainMaGe/weights/ov/fp32/'

device="cpu"


# ### Load Dataset csv
dataset_df = pd.read_csv(dataset_csv, header = None)
print(f"Number of rows: {dataset_df.shape[0]}")
print(f"Input Image Sample: {dataset_df.iloc[0][input_path_idx]}")
print(f"Mask Image Sample: {dataset_df.iloc[0][mask_path_idx]}")


def bench_pytorch_fp32():
    ### Load PyTorch model
    pt_model = fetch_model(modelname="resunet", num_channels=1, num_classes=2, num_filters=16)
    checkpoint = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
    pt_model.load_state_dict(checkpoint["model_state_dict"])

    ### Run PyTorch Inference
    print (f"\n Starting PyTorch inference with {pytorch_model_path} ...")

    _ = pt_model.eval()

    pt_stats =[]

    with torch.no_grad():
        for i, row in tqdm(dataset_df.iterrows()):
            sub_id = row[sub_idx]
            input_path = row[input_path_idx]
            mask_path = row[mask_path_idx]

            try:
                mask_image = get_mask_image(mask_path)
                input_image, patient_nib = get_input_image(input_path)

                i_start = timer()
                pt_output = pt_model(input_image)
                i_end = timer()

                p_start = timer()
                pt_output = pt_output.cpu().numpy()[0][0]
                pt_to_save = postprocess_output(pt_output, patient_nib.shape)
                pt_dice_score = dice(pt_to_save, mask_image)
                p_end = timer()

                pt_stat = [i, sub_id, pt_dice_score, i_end-i_start, p_end-p_start]
                pt_stats.append(pt_stat)
            except:
                print (f" Inference Failed: {sub_id} ")

    print (f"Done PyTorch inference with {pytorch_model_path} ...")
    pt_stats_df = pd.DataFrame(pt_stats)

    date_time_str = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    csv_name = f"pt_stats_{date_time_str}.csv"
    pt_stats_df.to_csv(csv_name, sep=',', header=False, index=False)
    print (f"Saved {csv_name} ...")

    print (f"\n PyTorch Dice Mean: {pt_stats_df[:][2].mean():.5f}")
    print (f"PyTorch Total Inf Time: {pt_stats_df[:][3].sum():.2f} sec, Mean: {pt_stats_df[:][3].mean():.2f} sec")

    return pt_stats_df


def bench_ov_fp32():
    #### Load OpenVINO model
    ov_model_dir = brainmage_root / 'BrainMaGe/weights/ov/fp32'
    modelname = "resunet_ma"

    model_xml = f'{ov_model_dir}/{modelname}.xml'
    model_bin = f'{ov_model_dir}/{modelname}.bin'

    # Load network to the plugin
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="CPU")
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
    csv_name = f"ov_fp32_stats_{date_time_str}.csv"
    ov_stats_df.to_csv(csv_name, sep=',', header=False, index=False)
    print (f"Saved {csv_name} ...")

    print (f"\n OpenVINO FP32 Dice Mean: {ov_stats_df[:][2].mean():.5f}")
    print (f"OpenVINO FP32 Total Inf Time: {ov_stats_df[:][3].sum():.2f} sec, Mean: {ov_stats_df[:][3].mean():.2f}")

    return ov_stats_df


def bench_ov_int8():

    # #### Load INT8 OpenVINO model
    ov_model_dir = brainmage_root / 'openvino/int8_openvino_model'
    modelname = "resunet_ma_int8"

    model_xml = f'{ov_model_dir}/{modelname}.xml'
    model_bin = f'{ov_model_dir}/{modelname}.bin'

    # Load network to the plugin
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name="CPU")
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
    csv_name = f"ov_int8_stats_{date_time_str}.csv"
    ov_int8_stats_df.to_csv(csv_name, sep=',', header=False, index=False)
    print (f"Saved {csv_name} ...")

    print (f"\n OpenVINO INT8 Dice Mean: {ov_int8_stats_df[:][2].mean():.5f}")
    print (f"OpenVINO INT8 Total Inf Time: {ov_int8_stats_df[:][3].sum():.2f}  sec, Mean: {ov_int8_stats_df[:][3].mean():.2f} sec")

    return ov_int8_stats_df

##
## Run Benchmark
##

pt_stats_df = bench_pytorch_fp32()
ov_stats_df = bench_ov_fp32()
ov_int8_stats_df = bench_ov_int8()

##
## Print Summary
##

print(f"\n Mean Dice Scores for {dataset_df.shape[0]} images")
print (f"PyTorch Dice Mean +/- STD: {pt_stats_df[:][2].mean():.5f} +/- {pt_stats_df[:][2].std():.1f}")
print (f"OpenVINO FP32 Dice Mean +/- STD: {ov_stats_df[:][2].mean():.5f} +/- {ov_stats_df[:][2].std():.1f}")
print (f"OpenVINO INT8 Dice Mean +/- STD: {ov_int8_stats_df[:][2].mean():.5f} +/- {ov_int8_stats_df[:][2].std():.1f}")

dice_diff_pt_ov = pt_stats_df[:][2] - ov_stats_df[:][2]
dice_diff_pt_ov_int8 = pt_stats_df[:][2] - ov_int8_stats_df[:][2]
print()
print(f"\n Accuracy Dice difference with OV FP32 {dice_diff_pt_ov.sum():.6f}")
#print(dice_diff_pt_ov.value_counts())
print(f" Accuracy Dice difference with OV INT8 {dice_diff_pt_ov_int8.sum():.6f}")
#print(dice_diff_pt_ov_int8.value_counts())

pt_total_inf_time = pt_stats_df[:][3].sum()
ov_total_inf_time = ov_stats_df[:][3].sum()
ov_int8_total_inf_time = ov_int8_stats_df[:][3].sum()
print()
print(f"\n Total Inference Time (sec) for {dataset_df.shape[0]} images")
print (f"PyTorch: {pt_total_inf_time:.2f} , Mean +/- STD: {pt_stats_df[:][3].mean():.2f} +/- {pt_stats_df[:][3].std():.2f} ")
print (f"OpenVINO FP32: {ov_total_inf_time:.2f} , Mean +/- STD: {ov_stats_df[:][3].mean():.2f} +/- {ov_stats_df[:][3].std():.2f} ")
print (f"OpenVINO INT8: {ov_int8_total_inf_time:.2f} , Mean +/- STD: {ov_int8_stats_df[:][3].mean():.2f} +/- {ov_int8_stats_df[:][3].std():.2f} ")

speedup_fp32 = pt_total_inf_time/ov_total_inf_time
speedup_int8 = pt_total_inf_time/ov_int8_total_inf_time
speedup_fp32_int8 = ov_total_inf_time/ov_int8_total_inf_time
print()
print (f"\n Speedup with OpenVINO FP32 for {pt_stats_df.shape[0]} images: {speedup_fp32:.1f}x")
print (f"Speedup with OpenVINO INT8 for {pt_stats_df.shape[0]} images: {speedup_int8:.1f}x")
print (f"Speedup with OpenVINO INT8 over FP32: {speedup_fp32_int8:.1f}x")
