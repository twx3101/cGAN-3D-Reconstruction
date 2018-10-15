from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, Conv3DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.backend import int_shape
from keras.utils import plot_model
from keras import backend as K
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import segment2d
import math


import sys
sys.path.append("/homes/wt814/IndividualProject/code/BatchGeneratorClass")
import logging

import imp
import BatchGenerator_v2
imp.reload(BatchGenerator_v2)
from BatchGenerator_v2 import *
from GenerateNiftiFilesFromData import *

def binarisation(data, threshold=0.9):
    binary_data = (data > threshold).astype(np.float64)
    return binary_data


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    y_binary = binarisation(y_pred, 0.5)
    return 1-dice_coef(y_true, y_pred)

def soft_dice_numpy(y_pred, y_true, eps=1e-7):
    '''
    c is number of classes
    :param y_pred: b x c x X x Y( x Z...) network output, must sum to 1 over c channel (such as after softmax)
    :param y_true: b x c x X x Y( x Z...) one hot encoding of ground truth
    :param eps:
    :return:
    '''

    axes = tuple(range(2, len(y_pred.shape)))
    intersect = np.sum(y_pred * y_true, axes)
    denom = np.sum(y_pred + y_true, axes)
    return - (2. *intersect / (denom + eps)).mean()

def soft_dice_loss(input, target):
    smooth = 1.

    iflat = input.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

class CGAN():
    def __init__(self):
        # Input shape
        self.img_height = 80
        self.img_width = 80
        self.img_depth = 80
        self.channels = 1
        self.img_shape = (self.img_height, self.img_width, self.img_depth, self.channels)

        #Condtions

        self.latent_dim =(80,80,1)

        optimizerAdam = Adam(0.0001, 0.9, 0.995)
        optimizerSGD = SGD(0.001)
        #optimizer = RMSprop(0.0001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizerSGD,
            metrics=['accuracy'])
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(80,80,1), name='Noise')
        x1 = Input(shape=(80,80,1), name='X1')
        x2 = Input(shape=(80,80,1), name= 'X2')
        x3 = Input(shape=(80,80,1), name='X3')
        img = self.generator([noise, x1, x2, x3])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, x1, x2, x3])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, x1, x2, x3], valid)
        plot_model(self.combined,
                       to_file='cGAN_cardiac_combined.png',
                       show_shapes=True)
        plot_model(self.discriminator,
                    to_file='cGAN_cardiac_discriminator.png',
                    show_shapes=True)
        plot_model(self.generator,
                        to_file='cGAN_cardiac_generator.png',
                        show_shapes=True)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizerSGD)

    def build_generator(self):

        #model = Sequential()

        #in 96x96x1 noise  + 96x96x1 + 96x96x1 + 96x96x1  labels
        #out 96*96*96
        noise = Input(shape=(80,80,1))
        x1 = Input(shape=(80,80,1))
        x2 = Input(shape=(80,80,1))
        x3 = Input(shape=(80,80,1))


        tower_1 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        tower_1 = Flatten()(tower_1)
        tower_1 = Dense(16)(tower_1)

        tower_2 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        tower_2 = Flatten()(tower_2)
        tower_2 = Dense(16)(tower_2)

        tower_3 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_3)
        tower_3 = Flatten()(tower_3)
        tower_3 = Dense(16)(tower_3)

        n_tower = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(noise)
        n_tower = (BatchNormalization(momentum=0.9))(n_tower)
        n_tower = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(n_tower)
        n_tower = (BatchNormalization(momentum=0.9))(n_tower)
        n_tower = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(n_tower)
        n_tower = (BatchNormalization(momentum=0.9))(n_tower)
        n_tower = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(n_tower)
        n_tower = (BatchNormalization(momentum=0.9))(n_tower)
        n_tower = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(n_tower)
        n_tower = (BatchNormalization(momentum=0.9))(n_tower)
        n_tower = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(n_tower)
        n_tower = (BatchNormalization(momentum=0.9))(n_tower)
        n_tower = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(n_tower)
        n_tower = Flatten()(n_tower)
        n_tower = Dense(16)(n_tower)

        model_input = concatenate([n_tower, tower_1, tower_2, tower_3], axis=-1)

        # model_input = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(merge)
        # model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # #model_input = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(model_input)
        # #model_input = BatchNormalization()(model_input)
        # #model_input = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(model_input)
        # #model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(1, (2,2), strides=3, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # model_input = Flatten()(model_input)
        # model_input = Dense(64, activation='relu')(model_input)


        x = Reshape((4,4,4,1))(model_input)

        x = (Conv3DTranspose(64, (7,7,7), strides=5, padding='same'))(x)
        x = (BatchNormalization(momentum=0.9))(x)
        x = (Activation('relu'))(x)
        #x = (Dropout(0.4))(x)

        x = (Conv3DTranspose(64, (3,3,3), strides=1, padding='same'))(x)
        x = (BatchNormalization(momentum=0.9))(x)
        x = (Activation('relu'))(x)
        #x = (Dropout(0.4))(x)

        x = (Conv3DTranspose(32, (4,4,4), strides=2, padding='same'))(x)
        x = (BatchNormalization(momentum=0.9))(x)
        x = (Activation('relu'))(x)

        x = (Conv3DTranspose(32, (3,3,3), strides=1, padding='same'))(x)
        x = (BatchNormalization(momentum=0.9))(x)
        x = (Activation('relu'))(x)

        x = (Conv3DTranspose(16, (4,4,4), strides=2, padding='same'))(x)
        x = (BatchNormalization(momentum=0.9))(x)
        x = (Activation('relu'))(x)

        x = (Conv3DTranspose(16, (3,3,3), strides=1, padding='same'))(x)
        x = (BatchNormalization(momentum=0.9))(x)
        x = (Activation('relu'))(x)

        # x = (Conv3DTranspose(16, (4,4,4), strides=2, padding='same'))(x)
        # x = (BatchNormalization(momentum=0.9))(x)
        # x = (Activation('relu'))(x)

        x = (Conv3DTranspose(1, (1,1,1), strides=1, padding='same'))(x)
        x = (Activation('sigmoid'))(x)



        #label_embedding = Flatten()(Embedding(96*96*3, self.latent_dim)(label))

        # hello = Model([noise, label], x)
        # plot_model(hello,
        #            to_file='cGAN_cardiac_generator.png',
        #            show_shapes=True)
        return Model([noise, x1, x2, x3], x)

    def build_discriminator(self):

        input_shape = (self.img_height, self.img_width, self.img_depth, self.channels)
        dropout = 0.4

        #input 96*96*96
        img = Input(shape=self.img_shape, name='Image')

        #conditioning vaiable 96*96*3
        # label = Input(shape=(96,96,3,), name='Condition')
        # model_input = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(label)
        # model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # model_input = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(model_input)
        # model_input = BatchNormalization()(model_input)
        # model_input = Flatten()(model_input)
        # model_input = Dense(48, activation='relu')(model_input)
        x1 = Input(shape=(80,80,1,),)
        x2 = Input(shape=(80,80,1,),)
        x3 = Input(shape=(80,80,1,),)
        tower_1 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        tower_1 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        tower_1 = Flatten()(tower_1)
        tower_1 = Dense(16)(tower_1)

        tower_2 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        tower_2 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        tower_2 = Flatten()(tower_2)
        tower_2 = Dense(16)(tower_2)

        tower_3 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_3)
        tower_3 = (BatchNormalization(momentum=0.9))(tower_3)
        tower_3 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_3)
        tower_3 = Flatten()(tower_3)
        tower_3 = Dense(16)(tower_3)


        model_input = concatenate([tower_1, tower_2, tower_3], axis=-1)
        model_input = Dense(48)(model_input)

        x = (Conv3D(16, (3,3,3), strides=2, padding='same',input_shape=(input_shape)))(img)
        x = (LeakyReLU(alpha=0.2))(x)
        #x = Dropout(dropout)(x)

        x = (Conv3D(16, (3,3,3), strides=1, padding='same'))(x)
        x = LeakyReLU(alpha=0.2)(x)
        #x = (Dropout(dropout))(x)

        x = (Conv3D(32, (3,3,3), strides=2, padding='same'))(x)
        x = LeakyReLU(alpha=0.2)(x)
        #x = (Dropout(dropout))(x)

        x = (Conv3D(32, (3,3,3), strides=1, padding='same'))(x)
        x = (LeakyReLU(alpha=0.2))(x)
        #x = (Dropout(dropout))(x)

        x = (Conv3D(64, (3,3,3), strides=2, padding='same'))(x)
        x = (LeakyReLU(alpha=0.2))(x)
        #x = (Dropout(dropout))(x)

        x = (Conv3D(64, (3,3,3), strides=1, padding='same'))(x)
        x = (LeakyReLU(alpha=0.2))(x)
        #x = (Dropout(dropout))(x)


        x = (Conv3D(1, (3,3,3), strides=3, padding='same'))(x)
        x = (Flatten())(x)

        x = Dense(16, activation='relu')(x)
        #x = Dense(16, activation='sigmoid')(x)
        merged = Concatenate(axis=-1)([x, model_input])

        x = Dense(64, activation='relu')(merged)
        x = (Dense(1))(x)
        x = (Activation('sigmoid'))(x)



        # flat_img = Flatten()(img)
        # flat_label = Flatten()(label)
        # merged = Concatenate(axis=-1)([flat_img, flat_label])
        # size = int_shape(merged)[1]
        #
        # model_input = Dense(64, activation='relu')(merged)
        # hello = Model([img, label], x)
        # plot_model(hello,
        #            to_file='cGAN_discriminator.png',
        #            show_shapes=True)
        return Model([img, x1, x2, x3], x)

    def shuffle(self, gt, x1, x2, x3, seed=0):
        np.random.seed(seed)
        indices = np.random.permutation(gt.shape[0])

        return gt[indices], x1[indices], x2[indices], x3[indices]


    def train(self, fold_no, epochs, batch_size=1, sample_interval=50, iter=False, folds=1):

        # Load the dataset

        train_gt, train_x1, train_x2, train_x3 = segment2d.load_train()
        test_gt, test_x1, test_x2, test_x3 = segment2d.load_test()
        # # Configure input
        # x_train = np.reshape(x_train, (len(x_train), 96, 96, 96, 1))  # adapt this if using `channels_first` image data format
        # x_test = np.reshape(x_test, (len(x_test), 96, 96, 96, 1))
        #
        # #conditioning
        # x_condition = np.dstack((x1_train, x2_train))
        # x_condition = np.dstack((x_condition,x3_train))

        # Adversarial ground truths


        writer = tf.summary.FileWriter('./logs/3iter/%d' %(fold_no+1))

        for epoch in range(epochs):
            #shuffle data before starting
            gt_t, x1_t, x2_t, x3_t = self.shuffle(train_gt, train_x1, train_x2, train_x3, epoch)
            avg_d_cost = 0
            avg_g_cost = 0
            avg_acc = 0
            total_batches = int(math.ceil(len(gt_t) / batch_size))
            gt = np.array_split(gt_t, total_batches)
            x1 = np.array_split(x1_t, total_batches)
            x2 = np.array_split(x2_t, total_batches)
            x3 = np.array_split(x3_t, total_batches)

            for batch_no in range(total_batches):
            #batch generator

                #condition
                #x_condition = np.concatenate((x1_train, x2_train), axis=-1)
                #x_condition = np.concatenate((x_condition,x3_train), axis=-1)
                valid = np.ones((len(gt[batch_no]), 1))
                fake = np.zeros((len(gt[batch_no]), 1))

                imgs = gt[batch_no]

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half batch of images
                # idx = np.random.randint(0, x_train.shape[0], batch_size)
                # imgs, labels = x_train[idx], x_condition[idx]

                # Sample noise as generator input
                #noise = np.random.normal(0, 1, (batch_size, 64))
                noise = np.random.normal(0, 1, (len(gt[batch_no]), 80,80))
                noise = np.reshape(noise,(len(gt[batch_no]),80,80,1))

                #noise = np.reshape(noise,(batch_size,96,96,))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict([noise, x1[batch_no],x2[batch_no],x3[batch_no]])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([imgs,  x1[batch_no],  x2[batch_no],  x3[batch_no]], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs,  x1[batch_no],  x2[batch_no],  x3[batch_no]], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                #summary = tf.Summary(value=[tf.Summary.Value(tag="d_loss_real",simple_value=d_loss_real[0]),])
                #summary1 = tf.Summary(value=[tf.Summary.Value(tag="d_loss_fake",simple_value=d_loss_fake[0]),])
                #writer.add_summary(summary, global_step=epoch)
                #writer.add_summary(summary1, global_step=epoch)
                #summary2 = tf.Summary(value=[tf.Summary.Value(tag="acc",simple_value=d_loss[1]),])
                #writer.add_summary(summary2, global_step=epoch)
                avg_d_cost += d_loss[0]/total_batches
                avg_acc += d_loss[1]/total_batches

                #Train discriminator twice
                if iter == True:
                    d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                    d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    summary = tf.Summary(value=[tf.Summary.Value(tag="d_loss_real",simple_value=d_loss_real[0]),])
                    summary1 = tf.Summary(value=[tf.Summary.Value(tag="d_loss_fake",simple_value=d_loss_fake[0]),])
                    writer.add_summary(summary, global_step=epoch)
                    writer.add_summary(summary1, global_step=epoch)
                    summary2 = tf.Summary(value=[tf.Summary.Value(tag="acc",simple_value=d_loss[1]),])
                    writer.add_summary(summary2, global_step=epoch)
                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator
                g_loss = self.combined.train_on_batch([noise,  x1[batch_no],  x2[batch_no],  x3[batch_no]], valid)
                iter_gen = 2
                for i in range(iter_gen):
                    g_loss = self.combined.train_on_batch([noise, x1[batch_no],  x2[batch_no],  x3[batch_no]], valid)
                    avg_g_cost += g_loss/(total_batches*(iter_gen + 1))
                else:
                    avg_g_cost += g_loss/total_batches


            # Calculate validation loss after 1 epoch
            val_loss = 0

            for i in range(len(test_gt)):
                #noise = np.random.normal(0, 1, (1, 16))
                noise = np.random.normal(0, 1, (1, 80,80))
                noise = np.reshape(noise,(1,80,80,1))
                #x_condition = np.concatenate((x1_test_real[i], x2_test_real[i]), axis=-1)
                #x_condition = np.concatenate((x_condition,x3_test_real[i]), axis=-1)
                #x_condition = np.reshape(x_condition, (1,96,96,3))
                #sampled_labels = x_condition
                # x_val = np.rollaxis(x_val, 1, 5)
                # x1 = np.rollaxis(x1_val, 1, 4)
                # x2 = np.rollaxis(x2_val, 1, 4)
                # x3 = np.rollaxis(x3_val, 1, 4)
                x1 = np.reshape(test_x1[i], (1,80,80,1))
                x2 = np.reshape(test_x2[i], (1,80,80,1))
                x3 = np.reshape(test_x3[i], (1,80,80,1))
                gen_imgs = self.generator.predict([noise, x1, x2, x3])
                val_loss += soft_dice_loss(gen_imgs, test_gt[i])/(len(test_gt))


            print("%d: [D loss %f, acc.: %.2f%%] [G loss: %f]" % (epoch, avg_d_cost, 100*avg_acc, avg_g_cost))
            print("Epoch: %d, val loss: %f" % (epoch, val_loss))


            summary = tf.Summary(value=[tf.Summary.Value(tag="d_loss",simple_value=avg_d_cost),])
            #summary1 = tf.Summary(value=[tf.Summary.Value(tag="d_loss_fake",simple_value=avg_d_fake_cost),])
            writer.add_summary(summary, global_step=epoch)
            #writer.add_summary(summary1, global_step=epoch)
            summary2 = tf.Summary(value=[tf.Summary.Value(tag="acc",simple_value=avg_acc),])
            writer.add_summary(summary2, global_step=epoch)
            summary3 = tf.Summary(value=[tf.Summary.Value(tag="g_loss", simple_value=avg_g_cost)])
            summary4 = tf.Summary(value=[tf.Summary.Value(tag="val_loss",simple_value=val_loss),])
            writer.add_summary(summary3, global_step=epoch)
            writer.add_summary(summary4, global_step=epoch)
                    # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch,train_gt[0], train_x1[0], train_x2[0], train_x3[0])

            if epochs - epoch == 1:
                self.sample_all()
                #os._exit(1)



    def sample_images(self, epoch, x, x1, x2, x3):
        #noise = np.random.normal(0, 1, (1, 16))
        noise = np.random.normal(0, 1, (1, 80,80))
        noise = np.reshape(noise,(1,80,80,1))
        #x_condition = np.concatenate((x1, x2), axis=-1)
        #x_condition = np.concatenate((x_condition,x3), axis=-1)
        #x_condition = np.reshape(x_condition, (1,96,96,3))
        #sampled_labels = x_conditio

        x1 = np.reshape(x1, (1,80,80,1))
        x2 = np.reshape(x2, (1,80,80,1))
        x3 = np.reshape(x3, (1,80,80,1))

        gen_imgs = self.generator.predict([noise, x1,x2,x3])

        # Rescale images 0 - 1
        #gen_imgs = 0.5 * gen_imgs + 0.5

        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
        #         axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig("images/%d.png" % epoch)
        # plt.close()

        np.save('cardiacImage/3iter/sample/%d.npy' %epoch, gen_imgs)

    def sample_all(self):
        x, x1_c, x2_c, x3_c = segment2d.load_evaluate()
            #noise = np.random.normal(0, 1, (1, 16))
        noise = np.random.normal(0, 1, (len(x), 80,80))
        noise = np.reshape(noise,(len(x),80,80,1))
        #x_condition = np.concatenate((x1[i], x2[i]), axis=-1)
        #x_condition = np.concatenate((x_condition,x3[i]), axis=-1)
        #x_condition = np.reshape(x_condition, (1,96,96,3))
        #sampled_labels = x_condition
        gen_imgs = self.generator.predict([noise, x1_c, x2_c, x3_c])

        np.save('cardiacImage/3iter/evaluate.npy', gen_imgs)

        self.generator.save('models/3iter/cGAN_generator.h5')
        self.combined.save('models/3iter/cGAN_full.h5')
        self.discriminator.save('models/3iter/cGAN_discriminator.h5')



if __name__ == '__main__':

    cgan = CGAN()
    cgan.train(fold_no=0, epochs=800, batch_size=16, sample_interval=100)
