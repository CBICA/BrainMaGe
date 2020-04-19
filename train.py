#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:14:34 2020

@author: siddhesh
"""

from __future__ import print_function, division
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.csv_creator_adv import generate_csv
from utils.data import WholeTumorDataset
from utils.optimizers import fetch_optimizer
from utils.cyclicLR import CyclicCosAnnealingLR
from utils.losses import dice_loss
from models.model import unet, resunet


def main():
    # Report the time stamp
    training_start_time = time.asctime()
    startstamp = time.time()
    print("\nHostname   :" + str(os.getenv("HOSTNAME")))
    print("\nStart Time :" + str(training_start_time))
    print("\nStart Stamp:" + str(startstamp))
    sys.stdout.flush()

    # TAKING THE PARAMS FILE AS AN ARGUMENT
    parser = argparse.ArgumentParser(description='Check if files are right \
                                     or not')
    parser.add_argument('params_cfg',
                        help='Name of directory to be scanned')
    args = parser.parse_args()
    print("Starting to check a sanity check on the following directory files:",
          args.params_cfg)
    cfg = args.params_cfg
    cfg = os.path.abspath(cfg)
    print("Checking for this cfg file : ", cfg)

    # READING FROM A CFG FILE and check if file exists or not
    if os.path.isfile(cfg):
        df = pd.read_csv(cfg, sep=' = ', names=['param_name', 'param_value'],
                         comment='#', skip_blank_lines=True,
                         engine='python').fillna(' ')
    else:
        print('Missing train_params.cfg file?')
        sys.exit(0)

    # Reading in all the parameters
    params = {}
    for i in range(df.shape[0]):
        params[df.iloc[i, 0]] = df.iloc[i, 1]

    # Although uneccessary, we still do this
    if not os.path.isdir(str(params['epoch_dir'])):
        os.mkdir(str(params['epoch_dir']))
    if not os.path.isdir(str(params['model_dir'])):
        os.mkdir(params['model_dir'])

    # PRINT PARSED ARGS
    print("\n\n")
    print("Training Folder Dir     :", params['train_dir'])
    print("Validation Dir          :", params['validation_dir'])
    print("Epoch Dir               :", params['epoch_dir'])
    print("Model Directory         :", params['model_dir'])
    print("Mode                    :", params['mode'])
    print("Number of modalities    :", params['num_channels'])
    print("Modalities              :", params['modalities'])
    print("Number of classes       :", params['num_classes'])
    print("Max Number of epochs    :", params['max_epochs'])
    print("Batch size              :", params['batch_size'])
    print("Optimizer               :", params['optimizer'])
    print("Learning Rate           :", params['learning_rate'])
    print("Learning Rate Milestones:", params['lr_milestones'])
    print("Patience to decay       :", params['decay_milestones'])
    print("Early Stopping Patience :", params['early_stop_patience'])
    print("Depth Layers            :", params['layers'])
    print("Model used              :", params['model'])
    print("Do you want to resume   :", params['load'])
    print("Load Weights Dir        :", params['load_weights'])
    sys.stdout.flush()

    # We generate CSV for training if not provided
    print("Generating CSV Files")
    # Generating training csv files
    if not params['csv_provided'] == 'True':
        print('Since CSV were not provided, we are gonna create for you')
        generate_csv(params['train_dir'],
                     to_save=params['model_dir'],
                     mode=params['mode'], ftype='train',
                     modalities=params['modalities'])
        generate_csv(params['validation_dir'],
                     to_save=params['model_dir'],
                     mode=params['mode'], ftype='validation',
                     modalities=params['modalities'])
        train_csv = os.path.join(params['model_dir'], 'train.csv')
        validation_csv = os.path.join(params['model_dir'], 'validation.csv')
    else:
        train_csv = params['train_csv']
        validation_csv = params['validation_csv']

    # Playing with the Dataloader object and setting it up
    dataset_train = WholeTumorDataset(train_csv, params)
    train_loader = DataLoader(dataset_train,
                              batch_size=int(params['batch_size']),
                              shuffle=True, num_workers=1)
    dataset_valid = WholeTumorDataset(validation_csv, params)
    validation_loader = DataLoader(dataset_valid, batch_size=1, shuffle=True,
                                   num_workers=1)
    model = resunet(int(params['num_channels']),
                    int(params['num_classes']),
                    int(params['base_filters']))

    print("Training Data : ", len(train_loader.dataset))
    print("Test Data :", len(validation_loader.dataset))
    sys.stdout.flush()

    print("Current Device : ", torch.cuda.current_device())
    print("Device Count on Machine : ", torch.cuda.device_count())
    # print("Device Name : ", torch.cuda.get_device_name())
    print("Cuda Availibility : ", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1),
              'GB')
        print('Cached: ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
    sys.stdout.flush()

    # If cuda is required, the pushing the model to cuda
    if params['cuda']:
        model.cuda()
    # Setting up the optimizer
    optimizer = fetch_optimizer(params['optimizer'], params['learning_rate'],
                                model)
    # Setting up an optimizer lr reducer
    lr_milestones = [int(i) for i in params['lr_milestones'][1:-1].split(',')]
    decay_milestones = [int(i) for i in params['decay_milestones'][1:-1].split(',')]
    scheduler = CyclicCosAnnealingLR(optimizer,
                                     milestones=lr_milestones,
                                     decay_milestones=decay_milestones,
                                     eta_min=5e-6)
    # Setting up the Evaludation Metric
    def dice(inp, target):
        smooth = 1
        iflat = inp.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return (2*intersection+smooth)/(iflat.sum()+tflat.sum()+smooth)

    def train(epoch, train_loss_list, train_dice_coef_list, params):
        start = time.time()
        temp_1_time = time.time()
        epoch_loss_list = []
        epoch_dice_list = []
        curr_loss = 0
        curr_dice = 0
        total_loss = 0
        total_dice = 0

        # Set model to training mode
        model.train()

        # print out some epoch information
        print('------------------------------------------------------------\n')
        print("\n\tEpoch : {} \t Starting Epoch at : {}".format(epoch,
                                                                time.asctime()))
        for param_group in optimizer.param_groups:
            print("\tEpoch Learning rate: ", param_group['lr'])

        # Start reading through the train loader
        for batch_idx, (subject) in enumerate(train_loader):
            # Load the subject and its grouund truth
            image = subject['image_data']
            mask = subject['gt_data']
            if params['cuda']:
                # Loading images into the GPU and ignoring the affine
                image, mask = image.cuda(), mask.cuda()

            # I Don't know why I do this step and
            # at this point, I am too  afraid to ask
            image, mask = Variable(image), Variable(mask)

            # Making sure that the optimizer has been reset
            optimizer.zero_grad()

            # Forward Propagation to get the output from the models
            output = model(image)

            # Handling the loss
            # Computing the loss function
            loss = dice_loss(output, mask)

            # Back Propagation for model to learn
            loss.backward()
            optimizer.step()

            # Also taking out the current loss for printing purposes
            curr_loss = loss.cpu().data.item()
            # Adding the loss to the global loss list to keep track in things
            # like TensorBoard or Matplotlib etc
            train_loss_list.append(loss.cpu().data.item())
            # Adding the loss to the current epoch loss list to keep track of
            # the things in current epoch for future use
            epoch_loss_list.append(loss.cpu().data.item())
            # Computing the total loss of the epochs current to print stuff
            total_loss += loss.cpu().data.item()
            average_loss = total_loss/(batch_idx+1)

            # Handling the Dice Coefficient
            # Computing the Dice Coefficient for current sample for printing
            dice_coef = dice(output, mask)
            # Also taking out the current loss for printing purposes
            curr_dice = dice_coef.cpu().data.item()
            # Adding the dice to the global dice list to keep track in things
            # like TensorBoard or Matplotlib etc
            train_dice_coef_list.append(dice_coef.cpu().data.item())
            # Adding the dice to the current epoch dice list to keep track of
            # the things in current epoch for future use
            epoch_dice_list.append(dice_coef.cpu().data.item())
            # Computing the total dice of the epochs current to print stuff
            total_dice += dice_coef.cpu().data.item()
            average_dice = total_dice/(batch_idx+1)

            # Start to print some things for maintaining logs
            if (batch_idx+1) % int(params['log_interval']) == 0:
                temp_2_time = time.time()
                print("\n\t\t[{}/{} ({:.0f}%)]\t\tTime Taken per log : {:.6f}s\
                              ".format(batch_idx+1,
                                       len(train_loader),
                                       100*(batch_idx+1) /
                                       len(train_loader),
                                       temp_2_time - temp_1_time))
                print("\t\tAverage Dice loss  : {:.6f}\t\tRecent sample loss : {:.6f}".format(average_loss, curr_loss))
                print("\t\tAverage Dice       : {:.6f}\t\tRecent Dice        : {:.6f}".format(average_dice, curr_dice))
                temp_1_time = time.time()
                sys.stdout.flush()

            # Emptying cache to speedup and save space
            torch.cuda.empty_cache()

        # Always save the most recent model which is current one
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(params['model_dir'],
                                params['model'] + str(epoch)+'.pt'))

        # Printing some information about the complete epoch
        print("\n\t\t[{}/{} ({:.0f}%)]".format(batch_idx+1, len(train_loader),
                                               100*(batch_idx+1) /
                                               len(train_loader)))
        print("\t\tEpoch Average Dice loss  : {:.6f}".format(average_loss))
        print("\t\tEpoch Average Dice       : {:.6f}".format(average_dice))

        # Printing time related information about the epoch
        end = time.time()
        print("\n\tEpoch : {} \t Ending Epoch at : {}".format(epoch,
                                                              time.asctime()))
        print("\tEpoch : {} \t Time Taken : {:.2f} Mins".format(epoch,
                                                                (end - start) /
                                                                60))

        # Returning whatever happened in the epoch
        return (train_loss_list, epoch_loss_list, total_loss,
                train_dice_coef_list, epoch_dice_list, total_dice)

    def test(epoch, test_loss_list, test_dice_coef_list, params):
        start = time.time()
        temp_1_time = time.time()
        test_epoch_loss_list = []
        test_epoch_dice_list = []
        test_curr_loss = 0
        test_curr_dice = 0
        test_total_loss = 0
        test_total_dice = 0

        # set model to evaluation mode
        model.eval()

        # print out some epoch information
        print('------------------------------------------------------------\n')
        print("\n\tEpoch : {} \t Starting Test at : {}".format(epoch,
                                                               time.asctime()))

        # Start reading through the validation loader
        for batch_idx, (subject) in enumerate(validation_loader):
            # Load the subject and its ground truth
            image = subject['image_data']
            mask = subject['gt_data']


            if params['cuda']:
                # Loading images into the GPU and ignore the affine.
                image, mask = image.cuda(), mask.cuda()
            # print("\nTraining: ", image.shape, mask.shape)
            # Set the torch to no grad in order to save space and speed
            with torch.no_grad():

                # Forward Propagation to get the output from the model
                output = model(image)

                # Handling the loss
                # Computing the loss function
                loss = dice_loss(output, mask)
                # Taking the current loss for printing purposes
                test_curr_loss = loss.cpu().data.item()
                # Adding the test loss to the global loss list to keep track in
                # the things like TensorBoard
                test_loss_list.append(loss.cpu().data.item())
                # Adding the dice to the current epoch dice list to keep track
                # of the things in current epoch for future use
                test_epoch_loss_list.append(loss.cpu().data.item())
                # Computing the total loss of the epochs current to print stuff
                test_total_loss += loss.cpu().data.item()
                test_average_loss = test_total_loss/(batch_idx+1)

                # Handling the dice coefficient
                # Taking the current dice for printing purposes
                dice_coef = dice(output, mask)
                # Taking out the current loss for printing purposes
                test_curr_dice = dice_coef.cpu().data.item()
                # Adding the test dice to the global dice list to keep track in
                # the things like TensorBoard
                test_dice_coef_list.append(dice_coef.cpu().data.item())
                # Adding the dice to the current epoch dice list to keep track
                # of the things in current epoch for future use
                test_epoch_dice_list.append(dice_coef.cpu().data.item())
                # Computing the total dice of the epochs current to print stuff
                test_total_dice += dice_coef.cpu().data.item()
                test_average_dice = test_total_dice/(batch_idx+1)
                subject['affine'] = np.eye(4)

            if (batch_idx+1) % int(params['log_interval']) == 0:
                temp_2_time = time.time()
                print("\n\t\t[{}/{} ({:.0f}%)]\t\tTime Taken per log : {:.6f}s\
                              ".format(batch_idx+1,
                                       len(validation_loader),
                                       100*(batch_idx+1) /
                                       len(validation_loader),
                                       temp_2_time - temp_1_time))
                print("\t\tAverage Dice loss  : {:.6f}\t\tRecent sample loss : {:.6f}".format(test_average_loss, test_curr_loss))
                print("\t\tAverage Dice       : {:.6f}\t\tRecent Dice        : {:.6f}".format(test_average_dice, test_curr_dice))

                temp_1_time = time.time()
                sys.stdout.flush()

            # Emptying the cache to speedup and save space
            torch.cuda.empty_cache()

        # Making sure that the learning rate scheduler keeps track
        scheduler.step(test_average_loss)

        # Printing some information about the complete epoch
        print("\n\t\t[{}/{} ({:.0f}%)]".format(batch_idx+1,
                                               len(validation_loader),
                                               100*(batch_idx+1) /
                                               len(validation_loader)))
        print("\t\tAverage Dice loss  : {:.6f}".format(test_average_loss))
        print("\t\tAverage Dice       : {:.6f}".format(test_average_dice))

        # Printing time related information about the epoch
        end = time.time()
        print("\n\tEpoch : {} \t Ending Testing at : {}".format(epoch,
                                                                time.asctime()))
        print("\tEpoch : {} \t Testing Time Taken : {:.2f} Mins".format(epoch,
                                                                        (end -
                                                                         start)
                                                                        / 60))

        return (test_loss_list, test_epoch_loss_list, test_total_loss,
                test_dice_coef_list, test_epoch_dice_list, test_total_dice)

    def create_epoch_dir(folder, epoch_count):
        """
        paramters:
        epoch_count : create folder with epoch name
        folder : where to write that folder
        """
        folder = os.path.abspath(folder)
        path = os.path.join(folder, str(epoch_count))
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            os.mkdir(os.path.join(folder, str(epoch_count+1)))

    def save_array(some_list, file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)
        temp = np.array(some_list, dtype=np.float64)
        np.save(file_name, temp)

    # Initializing blank lists which will be written out later
    train_loss_list = []
    test_loss_list = []
    train_dice_coef_list = []
    test_dice_coef_list = []

    # Important lists to store the best models
    epochs_train_total_loss_list = []
    epochs_test_total_loss_list = []
    epochs_train_total_dice_list = []
    epochs_test_total_dice_list = []

    if not os.path.exists(os.path.join(params['epoch_dir'], 'global')):
        os.mkdir(os.path.join(params['epoch_dir'], 'global'))

    # Start iterating through epochs and begin the training
    for epoch_count in range(int(params['max_epochs'])):

        # Displaying the best info
        print('\n\tEpoch : ', epoch_count)
        print('\tBest    : ', params['save_best'])

        # Showing the recent best epochs
        temp_list = np.array(epochs_test_total_loss_list)
        for i in range(int(params['save_best'])):
            if i >= len(temp_list):
                print("\tEpoch Number : {} \t Test Loss : {} \t\
                      Average Dice : {}".format('inf', 'inf', '0'))
            else:
                found_epoch = temp_list.argsort()[i]
                print("\tEpoch Number : {:.0f} \t Test Loss : {:.6f} \tTest \
                    Average Dice : {:.6f}".format(found_epoch,
                                                  temp_list[found_epoch],
                                                  (epochs_test_total_dice_list[found_epoch]/len(validation_loader))))

        # Create an epoch directory to store random epoch stuff
        create_epoch_dir(params['epoch_dir'], epoch_count)

        # Train Stuff
        # push things to the training function
        (train_loss_list, epoch_loss_list, total_loss, train_dice_coef_list,
         epoch_dice_list, total_dice) = train(epoch_count, train_loss_list,
                                              train_dice_coef_list, params)

        # Append the loss and dice toa  very important list
        epochs_train_total_loss_list.append(total_loss)
        epochs_train_total_dice_list.append(total_dice)

        # Start storing epoch train stuff
        # First store epoch train loss in epoch directory
        save_array(epoch_loss_list, os.path.join(params['epoch_dir'],
                                                 str(epoch_count),
                                                 'epoch_' + str(epoch_count) +
                                                 '_train_loss_list'))
        # Secondly store epoch train dice in epoch directory
        save_array(epoch_dice_list, os.path.join(params['epoch_dir'],
                                                 str(epoch_count),
                                                 'epoch_' + str(epoch_count) +
                                                 '_train_dice_list'))
        # Store the global loss lists
        # First store the global train loss in generic directory
        save_array(train_loss_list, os.path.join(params['epoch_dir'], 'global',
                                                 'global_train_loss_list'))
        # Secondly store the global train dice in generic directory
        save_array(train_dice_coef_list, os.path.join(params['epoch_dir'],
                                                      'global',
                                                      'global_train_dice_list'))

        # Test Stuff
        # push things to the testing function
        (test_loss_list, test_epoch_loss_list, test_total_loss,
         test_dice_coef_list, test_epoch_dice_list, test_total_dice) = test(epoch_count, test_loss_list, test_dice_coef_list, params)

        # Append the loss and dice to a very important list
        epochs_test_total_loss_list.append(test_total_loss)
        epochs_test_total_dice_list.append(test_total_dice)
        # Start storing epoch test stuff
        # First store epoch test loss in epoch directory
        save_array(test_epoch_loss_list, os.path.join(params['epoch_dir'],
                                                      str(epoch_count),
                                                      'epoch_' +
                                                      str(epoch_count) +
                                                      '_test_loss_list'))
        # Secondly store epoch test dice in epoch directory
        save_array(test_epoch_dice_list, os.path.join(params['epoch_dir'],
                                                      str(epoch_count),
                                                      'epoch_' +
                                                      str(epoch_count) +
                                                      '_test_dice_list'))
        # Store the global test lists
        # First store the global test loss in generic directory
        save_array(test_loss_list, os.path.join(params['epoch_dir'], 'global',
                                                'global_test_loss_list'))
        # Secondly store the global test dice in generic directory
        save_array(test_dice_coef_list, os.path.join(params['epoch_dir'],
                                                     'global',
                                                     'global_test_dice_list'))

        # Implement early stopping
        temp_list = np.array(epochs_test_total_loss_list, dtype=np.float64)
        temp_list = temp_list.argsort()[:int(params['save_best'])]
        if epoch_count > (max(temp_list) + int(params['early_stop_patience'])):
            print('Initiating Early stopping since nothing was achieved in \
                  the last {} epochs. Please check the plots later.')
            epoch_count = int(params['max_epochs']) + 1

        to_keep_list = temp_list.argsort()[:int(params['save_best'])]
            
        print("temp list :", temp_list)

if __name__ == "__main__":
    main()

