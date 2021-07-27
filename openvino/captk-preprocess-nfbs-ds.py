#!/usr/bin/env python
# coding: utf-8

# ### Generate NFBS Dataset manifest

import os
import pandas as pd
from tqdm import tqdm

nfbs_dataset_csv = 'nfbs-dataset.csv'
atlasImage_path = '/home/sdp/ravi/upenn/data/captk-registration/atlasImage.nii.gz'

captk_bin = '/home/sdp/ravi/upenn/captk/CaPTk/1.8.1/captk'

nfbs_dataset_df = pd.read_csv(nfbs_dataset_csv, header = None)
print("Number of rows:", nfbs_dataset_df.shape[0])

for i, row in tqdm(nfbs_dataset_df.iterrows()):
    sub_id = row[0]
    input_path = row[2]
    mask_path = row[3]
    
    input_out_path = input_path[:-7] + "_reg.nii.gz"
    mask_out_path = mask_path[:-7] + "_reg.nii.gz"
    
    preprocess_input_img_cmd = f"{captk_bin} Preprocessing -i {input_path} -rFI {atlasImage_path} -o {input_out_path} -reg Rigid"
    preprocess_mask_img_cmd = f"{captk_bin} Preprocessing -i {mask_path} -rFI {atlasImage_path} -o {mask_out_path} -reg Rigid -rSg 1"
    
    os.system(preprocess_input_img_cmd)
    os.system(preprocess_mask_img_cmd)
    

