#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:25:52 2019

@author: siddhesh
"""
import tensorflow as tf

###################################################### B A S I C   L A Y E R S ##############################################################
def batch_norm(inputs, training):
	return tf.layers.batch_normalization(inputs = inputs, training = training, \
                        center = True, scale = True, momentum = 0.9, fused=True)

def instance_norm(inputs):
    return tf.contrib.layers.instance_norm(inputs)

def maxpool_layer(inputs, pool_size, strides):
    return tf.layers.max_pooling3d(inputs = inputs, pool_size = pool_size, 
                                 strides = strides, padding='same',
                                 data_format='channels_last')
    
def conv3d_transpose(inputs, n_filter = 4, filter_size = 1, stride = 1):
    return tf.layers.conv3d_transpose(inputs = inputs, filters = n_filter, \
                    kernel_size = [filter_size, filter_size, filter_size], \
                    strides = [stride, stride], padding = "same", \
                    use_bias = False, activation = None, \
                    kernel_initializer = tf.initializers.variance_scaling(distribution = 'uniform'))

def conv3d(inputs, n_filter = 4, filter_size = 1, stride = 1):
    return tf.layers.conv3d(inputs = inputs, filters = n_filter, \
            kernel_size = [filter_size, filter_size, filter_size], \
            strides = [stride, stride], padding = "same", use_bias = False, \
            activation = None, kernel_initializer = tf.initializers.variance_scaling(distribution = 'uniform'))

###################################################### A D V A N C E D   L A Y E R S ##############################################################

def conv_layer_bn_relu(inputs, n_filters, filter_size, stride, training,scope='conv_bn_relu'):
    with tf.variable_scope(scope):
        conv = conv3d(inputs, n_filters, filter_size, stride)
        bn = batch_norm(conv, training = training)
        relu = tf.nn.relu(bn)
        return relu
    

def ConvUnit(inputs, n_filter, ksize, training, scope, instance_norm = False):
    with tf.variable_scope(scope):
        conv1 = conv3d(inputs, n_filter, filter_size = 3)
        if not instance_norm:
            bn1 = batch_norm( conv1,training )
            relu1 = tf.nn.relu( bn1 )
        else:
            in1 = instance_norm( conv1 )
            relu1 = tf.nn.relu( in1 )
        conv2 = conv3d( relu1, n_filter = n_filter/2, filter_size = 3)
        if not instance_norm:
            bn2 = batch_norm( conv2,training )
            relu2 = tf.nn.relu( bn2 )
        else:
            in2 = instance_norm( conv2 )
            relu2 = tf.nn.relu( in2 )
    return relu2

def ConvUnitUp(inputs, n_filter, ksize, training, scope, instance_norm):
    with tf.variable_scope(scope):
        conv1 = conv3d_transpose(inputs, n_filter, filter_size = 3)
        if not instance_norm:
            bn1 = batch_norm( conv1,training )
            relu1 = tf.nn.relu( bn1 )
        else:
            in1 = instance_norm( conv1 )
            relu1 = tf.nn.relu( in1 )
        conv2 = conv3d( relu1, n_filter = n_filter/2, filter_size = 3)
        if not instance_norm:
            bn2 = batch_norm( conv2,training )
            relu2 = tf.nn.relu( bn2 )
        else:
            in2 = instance_norm( conv2 )
            relu2 = tf.nn.relu( in2 )
        conv3 = conv3d( relu2, n_filter = n_filter/2, filter_size = 3)
        if not instance_norm:
            bn3 = batch_norm( conv3,training )
            relu3 = tf.nn.relu( bn3 )
        else:
            in3 = instance_norm( conv3 )
            relu3 = tf.nn.relu( in3 )
    return relu3

def ResNetUnit(inputs, n_filter = 64, ksize = 3, training = False,scope= 'resnetlayer', instance_norm = False):
    with tf.variable_scope(scope):
        conv1 = conv3d(inputs, n_filter = n_filter/2, filter_size = 3)
        if not instance_norm:
            bn1 = batch_norm( conv1,training )
            relu1 = tf.nn.relu( bn1 )
        else:
            in1 = instance_norm( conv1 )
            relu1 = tf.nn.relu( in1 )
        conv2 = conv3d( relu1, n_filter = n_filter/2, filter_size = 3)
        if not instance_norm:
            bn2 = batch_norm( conv2,training )
            relu2 = tf.nn.relu( bn2 )
        else:
            in2 = instance_norm( conv2 )
            relu2 = tf.nn.relu( in2 )

        res = relu2 + inputs
        return res

def ResNetUnitUp(inputs, n_filter = 64, ksize = 3, training = False,scope= 'resnetlayerup'):
    with tf.variable_scope(scope):
        conv1 = conv3d_transpose(inputs, n_filter = n_filter/2, filter_size = 3)
        if not instance_norm:
            bn1 = batch_norm( conv1,training )
            relu1 = tf.nn.relu( bn1 )
        else:
            in1 = instance_norm( conv1 )
            relu1 = tf.nn.relu( in1 )
        
        conv2 = conv3d( relu1, n_filter = n_filter/2, filter_size = 3)
        if not instance_norm:
            bn2 = batch_norm( conv2,training )
            relu2 = tf.nn.relu( bn2 )
        else:
            in2 = instance_norm( conv2 )
            relu2 = tf.nn.relu( in2 )

        conv3 = conv3d( relu2, n_filter = n_filter/2, filter_size = 3)
        if not instance_norm:
            bn3 = batch_norm( conv3,training )
            relu3 = tf.nn.relu( bn3 )
        else:
            in3 = instance_norm( conv2 )
            relu3 = tf.nn.relu( in3 )
        res = relu3 + relu1
        return res
    


