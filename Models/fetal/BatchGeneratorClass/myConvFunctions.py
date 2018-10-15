# -*- coding: utf-8 -*-
"""
Created on Wed May 30 16:27:04 2018

@author: jcerrola
"""

import random
import os
import numpy as np
import nibabel as nib
import tensorflow as tf


def tf_get_batch_size(input_var):
    return tf.shape(input_var)[0]

def tf_get_batch_vector_size(input_var):
    return tf.shape(input_var)[1]

def upsampling_filter(filter_shape):
    """Bilinear upsampling filter."""
    size = filter_shape[0:3]
    factor = (np.array(size) + 1)

    center = np.zeros_like(factor, np.float)

    for i in range(len(factor)):
        if size[i] % 2 == 1:
            center[i] = factor[i] - 1
        else:
            center[i] = factor[i] - 0.5

    og = np.ogrid[:size[0], :size[1], :size[2]]

    x_filt = (1 - abs(og[0] - center[0]) / np.float(factor[0]))
    y_filt = (1 - abs(og[1] - center[1]) / np.float(factor[1]))
    z_filt = (1 - abs(og[2] - center[2]) / np.float(factor[2]))

    filt = x_filt * y_filt * z_filt

    weights = np.zeros(filter_shape)
    for i in range(np.min(filter_shape[3:5])):
        weights[:, :, :, i, i] = filt

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    return tf.get_variable(name="upsampling_filter", initializer=init,
                            shape=weights.shape, trainable=True)
    
    

# 3D convolutional layer: 3Dconv + add_bias + batch_norm + non_linearity
def tf_conv3d(input_, output_dim,
              k_d = 5, k_h = 5, k_w = 5, # kernel size (k_d x k_h x k_w)
              d_d = 1, d_h = 1, d_w = 1, # strides (d_d, d_h, d_w)
              stddev = 0.02, name = 'conv3D', trainf = True,
              reuse = False, activation = None, padding='SAME', 
              bn=True): #batch normalization flag
    
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        #batch norm and non-linearity
        if bn:
            conv = tf.layers.batch_normalization(conv, training=trainf)
        if activation is not None:
            conv = activation(conv)
        return conv


# 3D Transposed Convolution Layer: 3Dcon3d_trans + add_bias + batch_norm + non-linearity
def tf_deconv3d(input_, output_shape,
             k_d=5, k_h=5, k_w=5, # kernel size (k_d x k_h x k_w)
             d_d=1, d_h=1, d_w=1, # strides (d_d, d_h, d_w)
             stddev=0.02, 
             bn=False, #batch normalization flag
             trainf = True,
             name="deconv3D", padding='SAME', reuse=False, activation=None):
    
    with tf.variable_scope(name, reuse=reuse):
        
        batch_size = tf_get_batch_size(input_)
        up_filt_shape = [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]]
        up_kernel = upsampling_filter(up_filt_shape)
        deconv = tf.nn.conv3d_transpose(input_, up_kernel, [batch_size, ] + output_shape,
                                        strides=[1, d_d, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv = tf.reshape(deconv, [batch_size, ] + output_shape)

        #batch norm and non-linearity
        if bn:
            deconv = tf.layers.batch_normalization(deconv, training=trainf)
        if activation is not None:
            deconv = activation(deconv)
        return deconv



def full_conn_layer(input_, name = 'fullyConn', 
                    out_dim = 10, # output size of the fully connected layers
                    stddev = 0.02,
                    bn=False, #batch normalization flag
                    trainf = True,
                    reuse=False, activation=None):

    with tf.variable_scope(name, reuse=reuse):
        
        fc_size = input_.get_shape()[-1]
        
        w = tf.get_variable('w', [fc_size, out_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        
        fc = tf.matmul(input_, w)
        fc = tf.nn.bias_add(fc,biases)
        
        #batch norm and non-linearity
        if bn:
            fc = tf.layers.batch_normalization(fc, training=trainf)
            
        if activation is not None:
            fc = activation(fc)
        return fc
       



# 2D convolutional layer: 3Dconv + add_bias + batch_norm + non_linearity
def tf_conv2d(input_, output_dim,
              k_h = 5, k_w = 5, # kernel size (k_h x k_w)
              d_h = 1, d_w = 1, # strides (d_h, d_w)
              stddev = 0.02, name = 'conv3D', trainf = True,
              reuse = False, activation = None, padding='SAME', 
              bn=True): #batch normalization flag
    
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        #batch norm and non-linearity
        if bn:
            conv = tf.layers.batch_normalization(conv, training=trainf)
        if activation is not None:
            conv = activation(conv)
        return conv
