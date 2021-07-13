#!/usr/bin/env python
# coding: utf-8

# ### Generate NFBS Dataset manifest

import os
import csv

data_dir = '/home/sdp/ravi/upenn/data/NFBS_Dataset'

sub_dirs = sorted(os.listdir(data_dir))
rows = []

for sub_dir in sub_dirs:
    sub_dir_path = data_dir + '/' + sub_dir
    f_names = sorted(os.listdir(sub_dir_path))
    row = [ sub_dir_path + '/' + f for f in f_names ]
    row.insert(0, sub_dir)
    rows.append(row)

# Sample Row:
# A00037112,/home/sdp/ravi/upenn/data/NFBS_Dataset/A00037112/sub-A00037112_ses-NFB3_T1w.nii.gz,/home/sdp/ravi/upenn/data/NFBS_Dataset/A00037112/sub-A00037112_ses-NFB3_T1w_brain.nii.gz,/home/sdp/ravi/upenn/data/NFBS_Dataset/A00037112/sub-A00037112_ses-NFB3_T1w_brainmask.nii.gz

with open('nfbs-dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)





