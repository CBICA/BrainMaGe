import pandas as pd
import os
import csv
import glob

def generate_csv(folder_path, to_save= '.', mode):
	folders = os.listdir(os.path.join(folder_path, '*'))
	if mode == 'train':
		f1 = open(os.path.join(to_save, 'train.csv'))
		for folder in folders:
            gt = glob.glob(os.path.join(folder_path, folder, '*t1_LPS_N4_r_maskFinal.nii.gz'))[0]
			t1 = glob.glob(os.path.join(folder_path, folder, '*t1_LPS_r.nii.gz'))[0]
			t2 = glob.glob(os.path.join(folder_path, folder, '*t2_LPS_r.nii.gz'))[0]
			t1ce = glob.glob(os.path.join(folder_path, folder, '*t1ce_LPS_r.nii.gz'))[0]
			flair = glob.glob(os.path.join(folder_path, folder, '*flair_LPS_r.nii.gz'))[0]
			f1.write(t1 + "," + gt + "\n")
			f1.write(t2 + "," + gt +"\n")
			f1.write(t1ce + "," + gt + "\n")
			f1.write(flair + "," + gt + "\n")
		f1.close()
	elif mode == 'train':
		f1 = open(os.path.join(to_save, 'test.csv'))
		for folder in folders:
            gt = glob.glob(os.path.join(folder_path, folder, '*t1_LPS_N4_r_maskFinal.nii.gz'))[0]
			t1 = glob.glob(os.path.join(folder_path, folder, '*t1_LPS_r.nii.gz'))[0]
			t2 = glob.glob(os.path.join(folder_path, folder, '*t2_LPS_r.nii.gz'))[0]
			t1ce = glob.glob(os.path.join(folder_path, folder, '*t1ce_LPS_r.nii.gz'))[0]
			flair = glob.glob(os.path.join(folder_path, folder, '*flair_LPS_r.nii.gz'))[0]
			f1.write(t1 + "," + gt + "\n")
			f1.write(t2 + "," + gt +"\n")
			f1.write(t1ce + "," + gt + "\n")
			f1.write(flair + "," + gt + "\n")
		f1.close()
	else:
		print("Wrong mode!")   
	return     
