
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
from datetime import timedelta


brainmage_root = Path('../')

nfbs_dataset_csv = 'nfbs-dataset.csv'
nfbs_dataset_csv = 'nfbs-dataset-test.csv'
nfbs_dataset_csv = 'nfbs-dataset-test-10.csv'
nfbs_dataset_csv = 'nfbs-dataset-test-5.csv'
#nfbs_dataset_csv = 'nfbs-dataset-test-1.csv'

pt_output_path = 'pt-outfile' # PyTorch output file
ov_output_path = 'ov-outfile' # ONNX output file

pytorch_model_path = brainmage_root / 'BrainMaGe/weights/resunet_ma.pt'
ov_model_dir = brainmage_root / 'BrainMaGe/weights/ov/fp32/'

device="cpu"

    
# ### Load Dataset csv

nfbs_dataset_df = pd.read_csv(nfbs_dataset_csv, header = None)
print("Number of rows:", nfbs_dataset_df.shape[0])

@profile
def bench_pytorch_fp32():

    ### Load PyTorch model

    pt_model = fetch_model(modelname="resunet", num_channels=1, num_classes=2, num_filters=16)
    checkpoint = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
    pt_model.load_state_dict(checkpoint["model_state_dict"])

    ### Run PyTorch Inference

    print (f"Starting PyTorch inference with {pytorch_model_path} ...")

    _ = pt_model.eval()

    pt_stats =[]

    with torch.no_grad():
        for i, row in tqdm(nfbs_dataset_df.iterrows()):
            sub_id = row[0]
            input_path = row[2]
            mask_path = row[3]

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
    
    pt_stats_df.to_csv('pt_stats.csv', sep=',', header=False, index=False)
    print (f"Saved pt_stats.csv ...")
    
    return pt_stats_df

@profile
def bench_ov_fp32():
    #### Load OpenVINO model

    ov_model_dir = brainmage_root / 'BrainMaGe/weights/ov/fp32'
    modelname = "resunet_ma"

    from openvino.inference_engine import IECore

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

    for i, row in tqdm(nfbs_dataset_df.iterrows()):
        sub_id = row[0]
        input_path = row[2]
        mask_path = row[3]

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
    
    ov_stats_df.to_csv('ov_fp32_stats.csv', sep=',', header=False, index=False)
    print (f"Saved ov_fp32_stats.csv ...")
    
    return ov_stats_df

@profile
def bench_ov_int8():

    # #### Load INT8 OpenVINO model

    ov_model_dir = brainmage_root / 'openvino/int8_openvino_model'
    modelname = "resunet_ma_int8" 


    from openvino.inference_engine import IECore

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

    for i, row in tqdm(nfbs_dataset_df.iterrows()):
        sub_id = row[0]
        input_path = row[2]
        mask_path = row[3]

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
    
    ov_int8_stats_df.to_csv('ov_int8_stats.csv', sep=',', header=False, index=False)
    print (f"Saved ov_int8_stats.csv ...")
    
    return ov_int8_stats_df

##
## Run Benchmark
##

# pt_stats_df = bench_pytorch_fp32()
# ov_stats_df = bench_ov_fp32()
ov_int8_stats_df = bench_ov_int8()


# dice_diff_pt_ov = pt_stats_df[:][2] - ov_stats_df[:][2]
# dice_diff_pt_ov_int8 = pt_stats_df[:][2] - ov_int8_stats_df[:][2]
# print()
# print(f"Accuracy Dice difference with OV FP32 {dice_diff_pt_ov.sum():.6f}")
# #print(dice_diff_pt_ov.value_counts())
# print(f"Accuracy Dice difference with OV INT8 {dice_diff_pt_ov_int8.sum():.6f}")
# #print(dice_diff_pt_ov_int8.value_counts())

# pt_total_inf_time = pt_stats_df[:][3].sum()
# ov_total_inf_time = ov_stats_df[:][3].sum()
# ov_int8_total_inf_time = ov_int8_stats_df[:][3].sum()
# print()
# print(f"Total Inference Time (sec) for {pt_stats_df.shape[0]} images")
# print (f"PyTorch: {pt_total_inf_time:.2f} , Mean: {pt_stats_df[:][3].mean():.2f}")
# print (f"OpenVINO FP32: {ov_total_inf_time:.2f} , Mean: {ov_stats_df[:][3].mean():.2f}")
# print (f"OpenVINO INT8: {ov_int8_total_inf_time:.2f} , Mean: {ov_int8_stats_df[:][3].mean():.2f}")

# speedup_fp32 = pt_total_inf_time/ov_total_inf_time
# speedup_int8 = pt_total_inf_time/ov_int8_total_inf_time
# speedup_fp32_int8 = ov_total_inf_time/ov_int8_total_inf_time
# print()
# print (f"Speedup with OpenVINO FP32 for {pt_stats_df.shape[0]} images: {speedup_fp32:.1f}x")
# print (f"Speedup with OpenVINO INT8 for {pt_stats_df.shape[0]} images: {speedup_int8:.1f}x")
# print (f"Speedup with OpenVINO INT8 over FP32: {speedup_fp32_int8:.1f}x")


# pt_stats_df

# pt_stats_df.shape[0]

# dice_diff = pt_stats_df[:][2] - ov_stats_df[:][2]
# dice_diff.value_counts()

# pt_total_inf_time = pt_stats_df[:][3].sum()
# ov_total_inf_time = ov_stats_df[:][3].sum()

# print(f"Total Inference Time for {pt_stats_df.shape[0]} images")
# print (f"PyTorch: {pt_total_inf_time}")
# print (f"OpenVINO: {ov_total_inf_time}")

# speedup = pt_total_inf_time/ov_total_inf_time
# print (f"Speedup with OpenVINO for {pt_stats_df.shape[0]} images: {speedup:.1f}x")

# pt_mean_inf_time = pt_stats_df[:][3].mean()
# ov_mean_inf_time = ov_stats_df[:][3].mean()

# print(f"Average Inference Time for {pt_stats_df.shape[0]} images")
# print (f"PyTorch: {pt_mean_inf_time}")
# print (f"OpenVINO: {ov_mean_inf_time}")

# speedup = pt_mean_inf_time/ov_mean_inf_time
# print (f"Speedup with OpenVINO for {pt_stats_df.shape[0]} images: {speedup:.1f}x")






