#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:54:06 2020

@author: siddhesh
"""

import torch.nn.functional as F
import torch.nn as nn
import torch
from seg_modules import in_conv, DownsamplingModule, EncodingModule
from seg_modules import UpsamplingModule, DecodingModule
from seg_modules import out_conv, FCNUpsamplingModule


class unet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters=16):
        super(unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.ins = in_conv(self.n_channels, base_filters)
        self.ds_0 = DownsamplingModule(base_filters, base_filters*2)
        self.en_1 = EncodingModule(base_filters*2, base_filters*2)
        self.ds_1 = DownsamplingModule(base_filters*2, base_filters*4)
        self.en_2 = EncodingModule(base_filters*4, base_filters*4)
        self.ds_2 = DownsamplingModule(base_filters*4, base_filters*8)
        self.en_3 = EncodingModule(base_filters*8, base_filters*8)
        self.ds_3 = DownsamplingModule(base_filters*8, base_filters*16)
        self.en_4 = EncodingModule(base_filters*16, base_filters*16)
        self.us_3 = UpsamplingModule(base_filters*16, base_filters*8)
        self.de_3 = DecodingModule(base_filters*16, base_filters*8)
        self.us_2 = UpsamplingModule(base_filters*8, base_filters*4)
        self.de_2 = DecodingModule(base_filters*8, base_filters*4)
        self.us_1 = UpsamplingModule(base_filters*4, base_filters*2)
        self.de_1 = DecodingModule(base_filters*4, base_filters*2)
        self.us_0 = UpsamplingModule(base_filters*2, 16)
        self.out = out_conv(base_filters*2, self.n_classes-1)

    def forward(self, x):
        x1 = self.ins(x)
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.out(x, x1)
        return x


class resunet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters=16):
        super(resunet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.ins = in_conv(self.n_channels, base_filters, res=True)
        self.ds_0 = DownsamplingModule(base_filters, base_filters*2)
        self.en_1 = EncodingModule(base_filters*2, base_filters*2, res=True)
        self.ds_1 = DownsamplingModule(base_filters*2, base_filters*4)
        self.en_2 = EncodingModule(base_filters*4, base_filters*4, res=True)
        self.ds_2 = DownsamplingModule(base_filters*4, base_filters*8)
        self.en_3 = EncodingModule(base_filters*8, base_filters*8, res=True)
        self.ds_3 = DownsamplingModule(base_filters*8, base_filters*16)
        self.en_4 = EncodingModule(base_filters*16, base_filters*16, res=True)
        self.us_3 = UpsamplingModule(base_filters*16, base_filters*8)
        self.de_3 = DecodingModule(base_filters*16, base_filters*8, res=True)
        self.us_2 = UpsamplingModule(base_filters*8, base_filters*4)
        self.de_2 = DecodingModule(base_filters*8, base_filters*4, res=True)
        self.us_1 = UpsamplingModule(base_filters*4, base_filters*2)
        self.de_1 = DecodingModule(base_filters*4, base_filters*2, res=True)
        self.us_0 = UpsamplingModule(base_filters*2, base_filters)
        self.out = out_conv(base_filters*2, self.n_classes-1, res=True)

    def forward(self, x):
        x1 = self.ins(x)
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        x = self.us_3(x5)
        x = self.de_3(x, x4)
        x = self.us_2(x)
        x = self.de_2(x, x3)
        x = self.us_1(x)
        x = self.de_1(x, x2)
        x = self.us_0(x)
        x = self.out(x, x1)
        return x


class fcn(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters=16):
        super(fcn, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.ins = in_conv(self.n_channels, base_filters)
        self.ds_0 = DownsamplingModule(base_filters, base_filters*2)
        self.en_1 = EncodingModule(base_filters*2, base_filters*2)
        self.ds_1 = DownsamplingModule(base_filters*2, base_filters*4)
        self.en_2 = EncodingModule(base_filters*4, base_filters*4)
        self.ds_2 = DownsamplingModule(base_filters*4, base_filters*8)
        self.en_3 = EncodingModule(base_filters*8, base_filters*8)
        self.ds_3 = DownsamplingModule(base_filters*8, base_filters*16)
        self.en_4 = EncodingModule(base_filters*16, base_filters*16)
        self.us_4 = FCNUpsamplingModule(base_filters*16, 1, scale_factor=5)
        self.us_3 = FCNUpsamplingModule(base_filters*8, 1, scale_factor=4)
        self.us_2 = FCNUpsamplingModule(base_filters*4, 1, scale_factor=3)
        self.us_1 = FCNUpsamplingModule(base_filters*2, 1, scale_factor=2)
        self.us_0 = FCNUpsamplingModule(base_filters, 1, scale_factor=1)
        self.conv_0 = nn.Conv3d(in_channels=5, out_channels=self.n_classes-1,
                                kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.ins(x)
        x2 = self.ds_0(x1)
        x2 = self.en_1(x2)
        x3 = self.ds_1(x2)
        x3 = self.en_2(x3)
        x4 = self.ds_2(x3)
        x4 = self.en_3(x4)
        x5 = self.ds_3(x4)
        x5 = self.en_4(x5)

        u5 = self.us_4(x5)
        u4 = self.us_3(x4)
        u3 = self.us_2(x3)
        u2 = self.us_1(x2)
        u1 = self.us_0(x1)
        x = torch.cat([u5, u4, u3, u2, u1], dim=1)
        x = self.conv_0(x)
        return F.sigmoid(x)