#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 06:12:55 2020

@author: siddhesh
"""

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as ptl
from Deep_BET.models.networks import fetch_model
from Deep_BET.utils.cyclicLR import CyclicCosAnnealingLR
from Deep_BET.utils.losses import dice_loss, dice
from Deep_BET.utils.data import SkullStripDataset
from Deep_BET.utils.optimizers import fetch_optimizer


class SkullStripper(ptl.LightningModule):
    def __init__(self, params):
        super(SkullStripper, self).__init__()
        self.params = params
        self.model = fetch_model(params['model'],
                                 int(self.params['num_modalities']),
                                 int(self.params['num_classes']),
                                 int(self.params['base_filters']))

    def forward(self, x):
        return self.model(x)

    def my_loss(self, output, mask):
        loss = dice_loss(output, mask)
        return loss

    def training_step(self, batch, batch_nb):
        image, mask = batch['image_data'], batch['ground_truth_data']
        output = self.forward(image)
        loss = self.my_loss(output, mask)
        dice_score = dice(output, mask)
        tensorboard_logs = {'train_loss': loss.cpu().data.item(),
                            'train_dice': dice_score.cpu().data.item()}
        return {'loss': loss,
                'dice': dice_score,
                'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        image, mask = batch['image_data'], batch['ground_truth_data']
        output = self.forward(image)
        loss = self.my_loss(output, mask)
        dice_score = dice(output, mask)
        tensorboard_logs = {'val_loss': loss.cpu().data.item(),
                            'val_dice': dice_score.cpu().data.item()}
        return {'val_loss': loss,
                'val_dice': dice_score,
                'val_log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_dice = torch.stack([x['val_dice'] for x in outputs]).mean()
        logs = {'avg_val_loss': avg_loss, 'avg_val_dice': avg_dice}
        print("Average validation loss :", avg_loss, "Average validation dice", avg_dice)
        return {'val_loss': avg_loss,
                'val_dice': avg_dice,
                'progress_bar': logs,
                'log': logs}

    def configure_optimizers(self):
        # Setting up the optimizer
        optimizer = fetch_optimizer(self.params['optimizer'],
                                    self.params['learning_rate'],
                                    self.model)
        # Setting up an optimizer lr reducer
        lr_milestones = [int(i)
                         for i in self.params['lr_milestones'][1:-1].split(',')]
        decay_milestones = [int(i)
                            for i in
                            self.params['decay_milestones'][1:-1].split(',')]
        scheduler = CyclicCosAnnealingLR(optimizer,
                                         milestones=lr_milestones,
                                         decay_milestones=decay_milestones,
                                         eta_min=5e-6)
        return [optimizer], [scheduler]

    @ptl.data_loader
    def train_dataloader(self):
        dataset_train = SkullStripDataset(self.params['train_csv'], self.params,
                                          test=False)
        return DataLoader(dataset_train,
                          batch_size=int(self.params['batch_size']),
                          shuffle=True, num_workers=4,
                          pin_memory=True)

    @ptl.data_loader
    def val_dataloader(self):
        dataset_valid = SkullStripDataset(self.params['validation_csv'], self.params,
                                          test=False)
        return DataLoader(dataset_valid,
                          batch_size=int(self.params['batch_size']),
                          shuffle=False, num_workers=4,
                          pin_memory=True)
