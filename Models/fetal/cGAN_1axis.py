from __future__ import print_function, division


########################### Import Keras #######################################
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, Conv3DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.backend import int_shape
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt


######################## Import datasets ########################################
import numpy as np
import tensorflow as tf

#load affine and load test dataset
import load_3d

########################## Import Batch Generator #############################
import sys
sys.path.append("/homes/wt814/IndividualProject/code/BatchGeneratorClass")
import logging

import imp
import BatchGenerator_v2
imp.reload(BatchGenerator_v2)
from BatchGenerator_v2 import *
from GenerateNiftiFilesFromData import *


########################### Helper Functions ##################################
def soft_dice_loss(input, target):
    smooth = 1.

    iflat = input.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))




######################### CGAN Architecture ####################################

class CGAN():
    def __init__(self):
        # Input shape
        self.img_height = 96
        self.img_width = 96
        self.img_depth = 96
        self.channels = 1
        self.img_shape = (self.img_height, self.img_width, self.img_depth, self.channels)

        #Condtions (Noise)
        self.latent_dim =(96,96,1)

        #Optimisers
        optimizerAdam = Adam(0.0001, 0.9, 0.995)
        optimizerSGD = SGD(0.001)
        #optimizer = RMSprop(0.0001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizerSGD,
            metrics=['accuracy'])

        #self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()
        #self.generator.summary()


        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(96,96,1), name='Noise')
        #x1 = Input(shape=(96,96,1), name='X1')
        #x2 = Input(shape=(96,96,1), name= 'X2')
        x3 = Input(shape=(96,96,1), name='X3')
        img = self.generator([noise, x3])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, x3])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, x3], valid)
        # plot_model(self.combined,
        #                to_file='cGAN_combined.png',
        #                show_shapes=True)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizerSGD)

    def build_generator(self):

        #model = Sequential()

        #in 96*96*1 (noise)  + (96*96*1) + (96*96*1) + (96*96*1)  [labels]
        #out 96*96*96
        noise = Input(shape=(96,96,1))
        #x1 = Input(shape=(96,96,1))
        #x2 = Input(shape=(96,96,1))
        x3 = Input(shape=(96,96,1))

        # ############### Condition X1 ######################################
        # tower_1 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        # tower_1 = Flatten()(tower_1)
        # tower_1 = Dense(16)(tower_1)


        ############ Condition X2 ############################################
        # tower_2 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        # tower_2 = Flatten()(tower_2)
        # tower_2 = Dense(24)(tower_2)

        ########### Condition X3 ##############################################
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
        tower_3 = Dense(32)(tower_3)

        ########### Noise #####################################################
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
        n_tower = Dense(32)(n_tower)

        ################### Latent Space (vector of size 64) ###################
        model_input = concatenate([n_tower, tower_3], axis=-1)


        ################## Reconstruct 3D image #################################
        x = Reshape((4,4,4,1))(model_input)

        x = (Conv3DTranspose(64, (7,7,7), strides=3, padding='same'))(x)
        x = (BatchNormalization(momentum=0.9))(x)
        x = (Activation('relu'))(x)

        x = (Conv3DTranspose(64, (3,3,3), strides=1, padding='same'))(x)
        x = (BatchNormalization(momentum=0.9))(x)
        x = (Activation('relu'))(x)

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

        x = (Conv3DTranspose(16, (4,4,4), strides=2, padding='same'))(x)
        x = (BatchNormalization(momentum=0.9))(x)
        x = (Activation('relu'))(x)

        x = (Conv3DTranspose(1, (1,1,1), strides=1, padding='same'))(x)
        x = (Activation('sigmoid'))(x)

        return Model([noise, x3], x)

    def build_discriminator(self):

        input_shape = (self.img_height, self.img_width, self.img_depth, self.channels)

        #input 96*96*96
        img = Input(shape=self.img_shape, name='Image')

        ######################## Conditioning inputs ###########################
        #x1 = Input(shape=(96,96,1,),)
        #x2 = Input(shape=(96,96,1,),)
        x3 = Input(shape=(96,96,1,),)

        ####################### Condition X1 ###################################
        # tower_1 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_1)
        # tower_1 = (BatchNormalization(momentum=0.9))(tower_1)
        # tower_1 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_1)
        # tower_1 = Flatten()(tower_1)
        # tower_1 = Dense(16)(tower_1)

        ###################### Condition X2 ####################################
        # tower_2 = Conv2D(16, (2,2), strides=2, padding='same', activation='relu')(x2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(16, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(32, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(32, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(64, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(64, (2,2), strides=1, padding='same', activation='relu')(tower_2)
        # tower_2 = (BatchNormalization(momentum=0.9))(tower_2)
        # tower_2 = Conv2D(1, (2,2), strides=2, padding='same', activation='relu')(tower_2)
        # tower_2 = Flatten()(tower_2)
        # tower_2 = Dense(24)(tower_2)

        ################## Condition X3 ########################################
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
        model_input = Dense(32)(tower_3)

        ################### Vectorised conditions ##############################
        # model_input = concatenate([tower_3], axis=-1)
        # model_input = Dense(48)(model_input)


        ################### Vectorising 3D images ##################################
        x = (Conv3D(16, (3,3,3), strides=2, padding='same',input_shape=(input_shape)))(img)
        x = (LeakyReLU(alpha=0.2))(x)

        x = (Conv3D(16, (3,3,3), strides=1, padding='same'))(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = (Conv3D(32, (3,3,3), strides=2, padding='same'))(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = (Conv3D(32, (3,3,3), strides=1, padding='same'))(x)
        x = (LeakyReLU(alpha=0.2))(x)

        x = (Conv3D(64, (3,3,3), strides=2, padding='same'))(x)
        x = (LeakyReLU(alpha=0.2))(x)

        x = (Conv3D(64, (3,3,3), strides=1, padding='same'))(x)
        x = (LeakyReLU(alpha=0.2))(x)


        x = (Conv3D(1, (3,3,3), strides=3, padding='same'))(x)
        x = (Flatten())(x)

        x = Dense(32, activation='relu')(x)

        ####################### Vectorised conditions + vectorised 3D images #############

        merged = Concatenate(axis=-1)([x, model_input])

        ############################# Discriminator ##########################
        x = Dense(64, activation='relu')(merged)
        x = (Dense(1))(x)
        x = (Activation('sigmoid'))(x)

        return Model([img, x3], x)

##################### Training Procedure #######################################

    def train(self, files, epochs, batch_size=1, sample_interval=100, d_iter=0, g_iter=0):


        _, x_test_real = load_3d.load_US_3d_fold()
        _, x1_test_real, _, x2_test_real, _, x3_test_real = load_3d.load_2d_fold()
        affine = load_3d.get_affine_3d()

        writer = tf.summary.FileWriter(files['logs'])

        #start Batch Generator
        confTrain = {}
        print("start\n")
        if sys.version_info[0] < 3:
            execfile(files['config_path'], confTrain)
        else:
            exec(open(files['config_path']).read(),confTrain)

        assert epochs == confTrain['numEpochs'], "Number of epochs should be \
        similar as in the config file (%d)" %(confTrain['numEpochs'])

        assert batch_size == confTrain['batchSizeTraining'],  "Batch size \
        should be similar as in the config file (%d)" %(confTrain['batchSizeTraining'])


        train_length = 0
        val_length = 0

        with open(confTrain['channelsTraining'][0]) as f:
            for line in f:
                train_length += 1

        with open(confTrain['channelsValidation_ID']) as f:

            for line in f:
                val_length += 1

        batchGen = BatchGeneratorVolAnd2DplanesMultiThread(confTrain, mode='training', infiniteLoop=False, maxQueueSize = 5)
        batchGen.generateBatches()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))



        for epoch in range(epochs):
            avg_d_cost = 0
            avg_g_cost = 0
            avg_acc = 0
            total_batches = int(train_length / batch_size)

            for batch_no in range(total_batches):
            #batch generator

                x_train, _ , x1_train, x2_train, x3_train = batchGen.getBatchAndScalingFactor()

                # Move channels to last axis
                x_train = np.rollaxis(x_train, 1, 5)
                x1_train = np.rollaxis(x1_train, 1, 4)
                x2_train = np.rollaxis(x2_train, 1, 4)
                x3_train = np.rollaxis(x3_train, 1, 4)

                imgs = x_train
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Sample noise as generator input

                noise = np.random.normal(0, 1, (batch_size, 96,96))
                noise = np.reshape(noise,(batch_size,96,96,1))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict([noise, x3_train])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([imgs, x3_train], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, x3_train], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                avg_d_cost += d_loss[0]/(total_batches*(d_iter+1))
                avg_acc += d_loss[1]/(total_batches*(d_iter+1))

                #Train discriminator twice
                for iterations in range(d_iter):
                    d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                    d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    avg_d_cost += d_loss[0]/(total_batches*(d_iter+1))
                    avg_acc += d_loss[1]/(total_batches*(d_iter+1))

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, x3_train], valid)
                for i in range(g_iter):
                    g_loss = self.combined.train_on_batch([noise, x3_train], valid)
                    avg_g_cost += g_loss/(total_batches*(g_iter + 1))


            print("%d: [D loss %f, acc.: %.2f%%] [G loss: %f]" % (epoch, avg_d_cost, 100*avg_acc, avg_g_cost))
            #print("Epoch: %d, val loss: %f" % (epoch, val_loss))


            summary = tf.Summary(value=[tf.Summary.Value(tag="d_loss",simple_value=avg_d_cost),])
            writer.add_summary(summary, global_step=epoch)
            summary2 = tf.Summary(value=[tf.Summary.Value(tag="acc",simple_value=avg_acc),])
            writer.add_summary(summary2, global_step=epoch)
            summary3 = tf.Summary(value=[tf.Summary.Value(tag="g_loss", simple_value=avg_g_cost)])
            writer.add_summary(summary3, global_step=epoch)

            if epochs - epoch == 1:
                self.sample_all(x_test_real, x1_test_real, x2_test_real, x3_test_real, affine,
                                save_path=files['predict_path'], model_path=files['models'])
                #os._exit(1)

        batchGen.finish()

    def train_k_folds(self, files, fold_no, epochs, batch_size=1, sample_interval=100, d_iter=0, g_iter=0, save=True):

        #assert folds > 0, "Need at least 1 fold to run"

        #for fold_no in range(folds):
        print("Fold: %d\n" %(fold_no+1))

        #load size of train and test set ################################
        x_train_real, x_test_real = load_3d.load_US_3d_fold(fold_no)
        x1_train_real, x1_test_real, x2_train_real, x2_test_real, x3_train_real, x3_test_real = load_3d.load_2d_fold(fold_no)
        affine = load_3d.get_affine_3d()


        ############ location of tensorboard log ########################################
        log_path = files['logs'] + "%d" %(fold_no+1)
        writer = tf.summary.FileWriter(log_path)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        #start Batch Generator
        confTrain = {}
        config_path = files['k_config_path'] + "%d.cfg" %(fold_no+1)
        print("start\n")
        if sys.version_info[0] < 3:
            execfile(config_path, confTrain)
        else:
            exec(open(config_path).read(),confTrain)


        assert epochs == confTrain['numEpochs'], "Number of epochs should be \
        similar as in the config file (%d)" %(confTrain['numEpochs'])

        assert batch_size == confTrain['batchSizeTraining'],  "Batch size \
        should be similar as in the config file (%d)" %(confTrain['batchSizeTraining'])


        train_length = 0
        val_length = 0

        with open(confTrain['channelsTraining'][0]) as f:
            for line in f:
                train_length += 1

        with open(confTrain['channelsValidation_ID']) as f:

            for line in f:
                val_length += 1

        batchGen = BatchGeneratorVolAnd2DplanesMultiThread(confTrain, mode='training', infiniteLoop=False, maxQueueSize = 5)
        #Validation Batch
        batchGenV = BatchGeneratorVolAnd2DplanesMultiThread(confTrain, mode='validation', infiniteLoop=False, maxQueueSize = 4)
        batchGen.generateBatches()
        batchGenV.generateBatches()

        for epoch in range(epochs):
            avg_d_cost = 0
            avg_g_cost = 0
            avg_acc = 0
            total_batches = int(train_length / batch_size)

            for batch_no in range(total_batches):

                x_train, _ , x1_train, x2_train, x3_train = batchGen.getBatchAndScalingFactor()
                x_train = np.rollaxis(x_train, 1, 5)
                x1_train = np.rollaxis(x1_train, 1, 4)
                x2_train = np.rollaxis(x2_train, 1, 4)
                x3_train = np.rollaxis(x3_train, 1, 4)



                imgs = x_train
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Sample noise as generator input
                #noise = np.random.normal(0, 1, (batch_size, 64))
                noise = np.random.normal(0, 1, (batch_size, 96,96))
                noise = np.reshape(noise,(batch_size,96,96,1))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict([noise, x3_train])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([imgs, x3_train], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, x3_train], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                avg_d_cost += d_loss[0]/(total_batches*(d_iter+1))
                avg_acc += d_loss[1]/(total_batches*(d_iter+1))

                #Train discriminator an extra d_iter time
                for iteration in range(d_iter):
                    d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                    d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    avg_d_cost += d_loss[0]/(total_batches*(d_iter+1))
                    avg_acc += d_loss[1]/(total_batches*(d_iter+1))
                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, x3_train], valid)
                avg_g_cost += g_loss/(total_batches*(g_iter + 1))

                ######### Generator iterations #################################
                for i in range(g_iter):
                    g_loss = self.combined.train_on_batch([noise, x3_train], valid)
                    avg_g_cost += g_loss/(total_batches*(g_iter + 1))

            # Calculate validation loss after 1 epoch
            val_loss = 0

            for i in range(val_length):
                #get noise
                noise = np.random.normal(0, 1, (1, 96,96))
                noise = np.reshape(noise,(1,96,96,1))

                #Get condition
                _, x_val, x1_val, x2_val, x3_val = batchGenV.getBatchAndScalingFactor()
                x_val = np.rollaxis(x_val, 1, 5)
                #x1 = np.rollaxis(x1_val, 1, 4)
                x2 = np.rollaxis(x2_val, 1, 4)
                x3 = np.rollaxis(x3_val, 1, 4)

                gen_imgs = self.generator.predict([noise, x3])
                gen_imgs = np.reshape(gen_imgs, (96,96,96))
                val_loss += soft_dice_loss(gen_imgs, x_val)/(val_length)


            print("%d: [D loss %f, acc.: %.2f%%] [G loss: %f]" % (epoch, avg_d_cost, 100*avg_acc, avg_g_cost))
            print("Epoch: %d, val loss: %f" % (epoch, val_loss))

            ######## add to logs ############################################
            summary = tf.Summary(value=[tf.Summary.Value(tag="d_loss",simple_value=avg_d_cost),])
            writer.add_summary(summary, global_step=epoch)
            summary2 = tf.Summary(value=[tf.Summary.Value(tag="acc",simple_value=avg_acc),])
            writer.add_summary(summary2, global_step=epoch)
            summary3 = tf.Summary(value=[tf.Summary.Value(tag="g_loss", simple_value=avg_g_cost)])
            summary4 = tf.Summary(value=[tf.Summary.Value(tag="val_loss",simple_value=val_loss),])
            writer.add_summary(summary3, global_step=epoch)
            writer.add_summary(summary4, global_step=epoch)
            ##############################################################

            # If at save interval => save generated image samples
            if sample_interval != 0 and epoch % sample_interval == 0:
                self.sample_images(epoch,x_train_real[0], x1_train_real[0], x2_train_real[0], x3_train_real[0], affine,
                 fold_no, save_path=files['sample_path'])

            if save == True and epochs - epoch == 1:
                self.sample_all(x_test_real, x1_test_real, x2_test_real, x3_test_real, affine, fold_no,
                save_path=files['predict_path'], model_path=files['models'])

        batchGen.finish()
        batchGenV.finish()


    def sample_images(self, epoch, x, x1, x2, x3, affine, fold, save_path):
        noise = np.random.normal(0, 1, (1, 96,96))
        noise = np.reshape(noise,(1,96,96,1))

        #x1_c =  np.reshape(x1, (1,96,96,1))
        x2_c =  np.reshape(x2, (1,96,96,1))
        x3_c =  np.reshape(x3, (1,96,96,1))
        gen_imgs = self.generator.predict([noise,x3_c])
        gen_imgs = np.reshape(gen_imgs, (96,96,96))

        save_img = nib.Nifti1Image(gen_imgs, affine)

        path = save_path + "%d/" %(fold+1)

        if not os.path.isdir(path):
            os.makedirs(path)

        nib.save(save_img, path + "%d.nii.gz" %(epoch))

    def sample_all(self, x, x1, x2, x3, affine, fold=0, save_path=None, model_path=None):
        for i in range(len(x)):
            if save_path is None:
                print("No save path is defined")
                break

            noise = np.random.normal(0, 1, (1, 96,96))
            noise = np.reshape(noise,(1,96,96,1))

            #x1_c =  np.reshape(x1[i], (1,96,96,1))
            x2_c =  np.reshape(x2[i], (1,96,96,1))
            x3_c =  np.reshape(x3[i], (1,96,96,1))
            gen_imgs = self.generator.predict([noise, x3_c])
            gen_imgs = np.reshape(gen_imgs, (96,96,96))

            save_img = nib.Nifti1Image(gen_imgs, affine)
            path = save_path + "%d/" % (fold+1)

            if not os.path.isdir(path):
                os.makedirs(path)
            nib.save(save_img, path + "%d.nii.gz" %i)


        if model_path is None:
            gen_path = './cGAN_generator.h5'
            full_path = './cGAN_full.h5'
            discri_path = './cGAN_discriminator.h5'
        else:
            if not os.path.isdir(model_path + "%d" %(fold+1)):
                os.makedirs(model_path + "%d" %(fold+1))
            gen_path = model_path + "%d/" % (fold+1) + 'cGAN_generator.h5'
            full_path = model_path + "%d/" % (fold+1) +'cGAN_full.h5'
            discri_path = model_path + "%d/" % (fold+1) +'cGAN_discriminator.h5'
        self.generator.save(gen_path)
        self.combined.save(full_path)
        self.discriminator.save(discri_path)

if __name__ == '__main__':

    fileLocations = {
                "config_path" : '/homes/wt814/IndividualProject/code/BatchGeneratorClass/BatchGenerator_v2_config_Full.cfg',
                "k_config_path" : '/homes/wt814/IndividualProject/code/BatchGeneratorClass/BatchGenerator_v2_config_fold_',
                "sample_path" : './niftiImage/1axis/Test/',
                "predict_path" : './niftiImage/1axis/Sample/',
                "models" : './cGAN_model/1axis/',
                "logs" : './logs/1axis/',
            }

    ### Training without coronal view ###
    for fold in range(3):
        cgan = CGAN()
    #cgan.train(files=fileLocations, epochs=1500, batch_size=2, sample_interval=100, d_iter=0, g_iter=1)
        cgan.train_k_folds(files=fileLocations, fold_no=fold, epochs=1500, batch_size=2, sample_interval=100, d_iter=0, g_iter=1)
