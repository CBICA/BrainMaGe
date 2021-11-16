#!/usr/bin/env python
# coding: utf-8

# ### Generate NFBS Dataset manifest

import os
import sys
import pandas as pd
import argparse


def generate_csv(data_dir):
    sub_dirs = sorted(os.listdir(data_dir))
    rows = []
    # There are inference errors with the following subjects, so remove them from the manifest csv.
    skip_sub_dirs = ["A00040944", "A00043704", "A00053850", "A00054914", "A00058218", "A00058552", "A00060430", "A00062942"]
    sub_dirs = list(set(sub_dirs) - set(skip_sub_dirs))

    for sub_dir in sub_dirs:
        sub_dir_path = data_dir + '/' + sub_dir
        f_names = sorted(os.listdir(sub_dir_path))
        row = [ sub_dir_path + '/' + f for f in f_names ]
        row.insert(0, sub_dir)
        rows.append(row)

    # Sample Row: sub_id, input_img_T1w, input_img_T1w_brain, mask_img
    # A00037112,/home/ubuntu/NFBS_Dataset/A00037112/sub-A00037112_ses-NFB3_T1w.nii.gz,/home/ubuntu/NFBS_Dataset/A00037112/sub-A00037112_ses-NFB3_T1w_brain.nii.gz,/home/sdp/ravi/upenn/data/NFBS_Dataset/A00037112/sub-A00037112_ses-NFB3_T1w_brainmask.nii.gz

    nfbs_ds_csv = 'nfbs-dataset.csv'
    nfbs_ds_train_csv = 'nfbs-dataset-train.csv'
    nfbs_ds_test_csv = 'nfbs-dataset-test.csv'

    nfbs_dataset_df = pd.DataFrame(rows)
    nfbs_dataset_df.to_csv(nfbs_ds_csv, sep=',', header=False, index=False)
    print(f"Number of rows written: {nfbs_dataset_df.shape[0]} in {nfbs_ds_csv}")

    #nfbs_dataset_df = pd.read_csv(nfbs_dataset_csv, header = None)
    train = nfbs_dataset_df.sample(frac=0.8,random_state=200) #random state is a seed value
    test = nfbs_dataset_df.drop(train.index)

    train.to_csv(nfbs_ds_train_csv, sep=',', header=False, index=False)
    test.to_csv(nfbs_ds_test_csv, sep=',', header=False, index=False)

    print(f"Number of rows written: {train.shape[0]} in {nfbs_ds_train_csv}")
    print(f"Number of rows written: {test.shape[0]} in {nfbs_ds_test_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', default=f"{os.getcwd()}/NFBS_Dataset/", help='Path to the Dataset root directory')
    args = parser.parse_args()

    data_dir = args.data_dir

    if not os.path.isdir(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit()
    if not os.listdir(data_dir):
        print(f"ERROR: Empty directory: {data_dir}")
        sys.exit()


    print(f"Generating csv files with {data_dir} ...")
    generate_csv(data_dir)