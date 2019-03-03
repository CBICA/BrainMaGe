"""
Created on Fri Feb 22 15:14:26 2019
@author: siddhesh
"""

#### TO DO ####

import tensorflow as tf
import numpy as np
import os
import glob
import signal
import sys
import time
import numpy as np
import getopt
import csv_creator
import pandas as pd
####### SIGNAL HANDLER FUNCTION ########
def signal_handler(signal, frame):
    print('Program interrupted! Aborting')
    sys.exit(0)

    
def main(argv):
    
    if len(argv) < 2:
        help()
        sys.exit(0)
        
    ### REGISTER TIME STAMP ###
    start = time.asctime()
    startstamp = time.time()
    
    ### PRINT TIME STAMP ###
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(start))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()
    
    ### SPECIFYING SIGNAL TRAP ###
    signal.signal(signal.SIGHUP, signal_handler )
    signal.signal(signal.SIGINT, signal_handler )
    signal.signal(signal.SIGTERM, signal_handler )
    
    ### DEFAULT ARGUMENTS ###
    
    model_directory = "/tmp/models"
    temp_directory = "/tmp/stuff"
    
    no_of_modalities = int(3)
    num_classes = int(2)
    
    num_epochs = int(100)
    min_epochs = int(10)
    batch_size = 2
    optimizer = str('sgd')
    learning_rate = np.float32(0.01)
    decay = np.float32(0.5)
    patience = 5
    layers = 3
    
    verbose = int(1)
    EVAL_EVERY_N_STEPS = int(1)
    
    #READING FROM A CFG FILE and check if file exists or not
    if os.path.isfile('./train_params.cfg'):
        df = pd.read_csv('./train_params.cfg', sep = ' = ', names = ['param_name', 'param_value'], comment = '#', skip_blank_lines = True, engine = 'python').fillna(' ')
    else:
        print('Missing train_params.cfg file? Have you placed it in the current folder?')
        sys.exit(0)
        
    params = {}
    for i in range(df.shape[0]):
        params[df.iloc[i, 0]] = df.iloc[i, 1]

    ### LOAD MODULES ###
    print("\n Loading Modules")
    
    import shutil
    import nibabel as nib
    
    ### CREATE TF Record DIRECTORY ###
    if not os.path.isdir(str(params['tfrecord_dir'])):
        os.mkdir(str(params['tfrecord_dir']))
    ### CREATE MODEL DIRECTORY ###
    if not os.path.isdir(str(params['model_dir'])):
        os.mkdir(params['model_dir'])
    ### SETTING VERBOSITY OF TENSORFLOW ###
    if params['verbose'] == 1:
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        tf.logging.set_verbosity(0)
        
    ### PRINT PARSED ARGS ###
    print("Training Folder Dir   :", params['train_dir'])
    print("Validation Dir        :", params['validation_dir'])
    print("TF Record Dir         :", params['tfrecord_dir'])
    print("Model Directory       :", params['model_dir'])
    print("Number of modalities  :", params['num_modalities'])
    print("Number of classes     :", params['num_classes'])
    print("Number of epochs      :", params['max_epochs'])
    print("Batch size            :", params['batch_size'])
    print("Optimizer             :", params['optimizer'])
    print("Learning Rate         :", params['learning_rate'])
    print("Decay Rate            :", params['decay_rate'])
    print("Patience to decay     :", params['patience'])
    print("Depth Layers          :", params['layers'])
    print("Verbosity             :", params['verbose'])
    print("Model used            :", params['model_name'])
    print("Do you want to resume :", params['load'])
    print("Load Weights Dir      :", params['load_weights'])

    
    ### READ SUBJECT LIST FILE ###
    train_sublist = []
    val_sublist = []
    all_sublist = []
    
    print("Generating CSV Files")    
    #Generating training csv files
    csv_creator.generate_csv(params['train_dir'], to_save = params['model_dir'], mode = 'train')
    #Generating validation csv files
    csv_creator.generate_csv(params['validation_dir'], to_save = params['model_dir'], mode = 'validate')
    #Generating Training TF Records
    df_train = pd.read_csv(os.path.join(params['model_dir'], 'train.csv'))
    