#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:18:22 2020

@author: siddhesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class in_conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3,
                 dropout_p=0.3, leakiness=1e-2, conv_bias=True,
                 inst_norm_affine=True, res=False, lrelu_inplace=True):
        """[The initial convolution to enter the network, kind of like encode]

        [This function will create the input convolution]

        Arguments:
            input_channels {[int]} -- [the input number of channels, in our
                                       case the number of modalities]
            output_channels {[int]} -- [the output number of channels, will
                                        determine the upcoming channels]

        Keyword Arguments:
            kernel_size {number} -- [size of filter] (default: {3})
            dropout_p {number} -- [dropout probablity] (default: {0.3})
            leakiness {number} -- [the negative leakiness] (default: {1e-2})
            conv_bias {bool} -- [to use the bias in filters] (default: {True})
            inst_norm_affine {bool} -- [affine use in norm] (default: {True})
            res {bool} -- [to use residual connections] (default: {False})
            lrelu_inplace {bool} -- [To update conv outputs with lrelu outputs]
                                    (default: {True})
        """
        nn.Module.__init__(self)
        self.residual = res
        self.dropout_p = dropout_p
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.inst_norm_affine = inst_norm_affine
        self.lrelu_inplace = lrelu_inplace
        self.dropout = nn.Dropout3d(dropout_p)
        self.in_0 = nn.InstanceNorm3d(output_channels,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.in_1 = nn.InstanceNorm3d(output_channels,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.conv0 = nn.Conv3d(input_channels, output_channels, kernel_size=3,
                               stride=1, padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)
        self.conv1 = nn.Conv3d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)

    def forward(self, x):
        """The forward function for initial convolution

        input --> conv0 --> | --> in --> lrelu --> conv1 --> dropout --> in -|
                            |                                                |
                 output <-- + <-------------------------- conv2 <-- lrelu <--|

        Arguments:
            x {[Tensor]} -- [Takes in a type of torch Tensor]

        Returns:
            [Tensor] -- [Returns a torch Tensor]
        """
        x = self.conv0(x)
        if self.residual:
            skip = x
        x = F.leaky_relu(self.in_0(x), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = self.conv1(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(self.in_1(x), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = self.conv2(x)
        if self.residual:
            x = x + skip
        return x

class DownsamplingModule(nn.Module):
    def __init__(self, input_channels, output_channels, leakiness=1e-2,
                 dropout_p=0.3, kernel_size=3, conv_bias=True,
                 inst_norm_affine=True, lrelu_inplace=True):
        """[To Downsample a given input with convolution operation]

        [This one will be used to downsample a given comvolution while doubling
        the number filters]

        Arguments:
            input_channels {[int]} -- [The input number of channels are taken
                                       and then are downsampled to double
                                       usually]
            output_channels {[int]} -- [the output number of channels are
                                        usually the double of what of input]

        Keyword Arguments:
            leakiness {float} -- [the negative leakiness] (default: {1e-2})
            conv_bias {bool} -- [to use the bias in filters] (default: {True})
            inst_norm_affine {bool} -- [affine use in norm] (default: {True})
            lrelu_inplace {bool} -- [To update conv outputs with lrelu outputs]
                                    (default: {True})
        """
        #nn.Module.__init__(self)
        super(DownsamplingModule, self).__init__()
        self.dropout_p = dropout_p
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.inst_norm_affine = inst_norm_affine
        self.lrelu_inplace = lrelu_inplace
        self.in_0 = nn.InstanceNorm3d(output_channels,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.conv0 = nn.Conv3d(input_channels, output_channels, kernel_size=3,
                               stride=2, padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)

    def forward(self, x):
        """[This is a forward function for ]

        [input -- > in --> lrelu --> ConvDS --> output]

        Arguments:
            x {[Tensor]} -- [Takes in a type of torch Tensor]

        Returns:
            [Tensor] -- [Returns a torch Tensor]
        """
        x = F.leaky_relu(self.in_0(self.conv0(x)),
                         negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        return x

class EncodingModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3,
                 dropout_p=0.3, leakiness=1e-2, conv_bias=True,
                 inst_norm_affine=True, res=False, lrelu_inplace=True):
        """[The Encoding convolution module to learn the information and use]

            [This function will create the Learning convolutions]

            Arguments:
                input_channels {[int]} -- [the input number of channels, in our
                                           case the number of channels from
                                           downsample]
                output_channels {[int]} -- [the output number of channels, will
                                            determine the upcoming channels]

            Keyword Arguments:
                kernel_size {number} -- [size of filter] (default: {3})
                dropout_p {number} -- [dropout probablity] (default: {0.3})
                leakiness {number} -- [the negative leakiness]
                                      (default: {1e-2})
                conv_bias {bool} -- [to use the bias in filters]
                                      (default: {True})
                inst_norm_affine {bool} -- [affine use in norm]
                                      (default: {True})
                res {bool} -- [to use residual connections] (default: {False})
                lrelu_inplace {bool} -- [To update conv outputs with lrelu
                                         outputs] (default: {True})
        """
        nn.Module.__init__(self)
        self.res = res
        self.dropout_p = dropout_p
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.inst_norm_affine = inst_norm_affine
        self.lrelu_inplace = lrelu_inplace
        self.dropout = nn.Dropout3d(dropout_p)
        self.in_0 = nn.InstanceNorm3d(output_channels,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.in_1 = nn.InstanceNorm3d(output_channels,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.conv0 = nn.Conv3d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)
        self.conv1 = nn.Conv3d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)

    def forward(self, x):
        """The forward function for initial convolution

        [input --> | --> in --> lrelu --> conv0 --> dropout --> in -|
                   |                                                |
        output <-- + <-------------------------- conv1 <-- lrelu <--|]

        Arguments:
            x {[Tensor]} -- [Takes in a type of torch Tensor]

        Returns:
            [Tensor] -- [Returns a torch Tensor]
        """
        if self.res:
            skip = x
        x = F.leaky_relu(self.in_0(x), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = self.conv0(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(self.in_1(x), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = self.conv1(x)
        if self.res:
            x = x + skip
        return x

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=True):
        super(Interpolate, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size,
                                         scale_factor=self.scale_factor,
                                         mode=self.mode,
                                         align_corners=self.align_corners)

class UpsamplingModule(nn.Module):
    def __init__(self, input_channels, output_channels, leakiness=1e-2,
                 lrelu_inplace=True, kernel_size=3, scale_factor=2,
                 conv_bias=True, inst_norm_affine=True):
        """[summary]

        [description]

        Arguments:
            input__channels {[type]} -- [description]
            output_channels {[type]} -- [description]

        Keyword Arguments:
            leakiness {number} -- [description] (default: {1e-2})
            lrelu_inplace {bool} -- [description] (default: {True})
            kernel_size {number} -- [description] (default: {3})
            scale_factor {number} -- [description] (default: {2})
            conv_bias {bool} -- [description] (default: {True})
            inst_norm_affine {bool} -- [description] (default: {True})
        """
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.scale_factor = scale_factor
        self.interpolate = Interpolate(scale_factor=self.scale_factor,
                                       mode='trilinear', align_corners=True)
        self.conv0 = nn.Conv3d(input_channels, output_channels, kernel_size=1,
                               stride=1, padding=0, bias=self.conv_bias)

    def forward(self, x):
        """[summary]

        [description]

        Extends:
        """
        x = self.conv0(self.interpolate(x))
        return x

class FCNUpsamplingModule(nn.Module):
    def __init__(self, input_channels, output_channels, leakiness=1e-2,
                 lrelu_inplace=True, kernel_size=3, scale_factor=2,
                 conv_bias=True, inst_norm_affine=True):
        """[summary]

        [description]

        Arguments:
            input__channels {[type]} -- [description]
            output_channels {[type]} -- [description]

        Keyword Arguments:
            leakiness {number} -- [description] (default: {1e-2})
            lrelu_inplace {bool} -- [description] (default: {True})
            kernel_size {number} -- [description] (default: {3})
            scale_factor {number} -- [description] (default: {2})
            conv_bias {bool} -- [description] (default: {True})
            inst_norm_affine {bool} -- [description] (default: {True})
        """
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.scale_factor = scale_factor
        self.interpolate = Interpolate(scale_factor=2**(self.scale_factor-1),
                                       mode='trilinear', align_corners=True)
        self.conv0 = nn.Conv3d(input_channels, output_channels, kernel_size=1,
                               stride=1, padding=0, bias=self.conv_bias)

    def forward(self, x):
        """[summary]

        [description]

        Extends:
        """
        x = self.interpolate(self.conv0(x))
        return x

class DecodingModule(nn.Module):
    def __init__(self, input_channels, output_channels, leakiness=1e-2,
                 conv_bias=True, kernel_size=3, inst_norm_affine=True,
                 res=True, lrelu_inplace=True):
        """[The Decoding convolution module to learn the information and use
            later]

        [This function will create the Learning convolutions]

        Arguments:
            input_channels {[int]} -- [the input number of channels, in our
                                       case the number of channels from
                                       downsample]
            output_channels {[int]} -- [the output number of channels, will
                                        determine the upcoming channels]

        Keyword Arguments:
            kernel_size {number} -- [size of filter] (default: {3})
            leakiness {number} -- [the negative leakiness] (default: {1e-2})
            conv_bias {bool} -- [to use the bias in filters] (default: {True})
            inst_norm_affine {bool} -- [affine use in norm] (default: {True})
            res {bool} -- [to use residual connections] (default: {False})
            lrelu_inplace {bool} -- [To update conv outputs with lrelu outputs]
                                    (default: {True})
        """
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.res = res
        self.in_0 = nn.InstanceNorm3d(input_channels,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.in_1 = nn.InstanceNorm3d(output_channels,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.in_2 = nn.InstanceNorm3d(output_channels,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.conv0 = nn.Conv3d(input_channels, output_channels, kernel_size=3,
                               stride=1, padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)
        self.conv1 = nn.Conv3d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=3,
                               stride=1, padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.leaky_relu(self.in_0(x))
        x = self.conv0(x)
        if self.res:
            skip = x
        x = F.leaky_relu(self.in_1(x))
        x = F.leaky_relu(self.in_2(self.conv1(x)))
        x = self.conv2(x)
        if self.res:
            x = x + skip
        return x

class out_conv(nn.Module):
    def __init__(self, input_channels, output_channels, leakiness=1e-2,
                 kernel_size=3, conv_bias=True, inst_norm_affine=True,
                 res=True, lrelu_inplace=True):
        """[The Out convolution module to learn the information and use later]

        [This function will create the Learning convolutions]

        Arguments:
            input_channels {[int]} -- [the input number of channels, in our
                                       case the number of channels from
                                       downsample]
            output_channels {[int]} -- [the output number of channels, will
                                        determine the upcoming channels]

        Keyword Arguments:
            kernel_size {number} -- [size of filter] (default: {3})
            leakiness {number} -- [the negative leakiness] (default: {1e-2})
            conv_bias {bool} -- [to use the bias in filters] (default: {True})
            inst_norm_affine {bool} -- [affine use in norm] (default: {True})
            res {bool} -- [to use residual connections] (default: {False})
            lrelu_inplace {bool} -- [To update conv outputs with lrelu outputs]
                                    (default: {True})
        """
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.res = res
        self.in_0 = nn.InstanceNorm3d(input_channels,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.in_1 = nn.InstanceNorm3d(input_channels//2,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.in_2 = nn.InstanceNorm3d(input_channels//2,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.in_3 = nn.InstanceNorm3d(input_channels//2,
                                      affine=self.inst_norm_affine,
                                      track_running_stats=True)
        self.conv0 = nn.Conv3d(input_channels, input_channels//2,
                               kernel_size=3, stride=1,
                               padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)
        self.conv1 = nn.Conv3d(input_channels//2, input_channels//2,
                               kernel_size=3, stride=1,
                               padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)
        self.conv2 = nn.Conv3d(input_channels//2, input_channels//2,
                               kernel_size=3, stride=1,
                               padding=(kernel_size - 1) // 2,
                               bias=self.conv_bias)
        self.conv3 = nn.Conv3d(input_channels//2, output_channels,
                               kernel_size=1, stride=1, padding=0,
                               bias=self.conv_bias)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.leaky_relu(self.in_0(x))
        x = self.conv0(x)
        if self.res:
            skip = x
        x = F.leaky_relu(self.in_1(x))
        x = F.leaky_relu(self.in_2(self.conv1(x)))
        x = self.conv2(x)
        if self.res:
            x = x + skip
        x = F.leaky_relu(self.in_3(x))
        x = F.sigmoid(self.conv3(x))
        return x
