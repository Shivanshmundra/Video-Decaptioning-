#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Reshape, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import tensorflow as tf
import os, sys
import data_manager#, scoring

fsize = 128 # 256
# network architecture
def build_model():
    input_shape = (fsize, fsize, 3)
    input_img = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(1,1), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3, 3), strides=(1,1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), strides=(2,2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # here is the bottleneck

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    #x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, decoded)
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['mse'])
    print(model.summary())

    return model

# training loop
def train(model, nb_epochs, batch_size):

    nb_train_clips = 20000
    nb_val_clips = 2000

    training_generator = data_manager.generate_data(nb_train_clips, batch_size, 'train')
    validation_generator = data_manager.generate_data(nb_val_clips, batch_size, 'dev')
    
    checkpointer = ModelCheckpoint(monitor='val_loss', filepath="models/model_{epoch:02d}-{val_loss:.2f}.h5", verbose=1, save_best_only=False)
    
    model.fit_generator(generator = training_generator, steps_per_epoch = int(nb_train_clips//batch_size),
                    validation_data = validation_generator, validation_steps = int(nb_val_clips//batch_size),
                    callbacks=[checkpointer], shuffle=False, epochs=nb_epochs)

    model.save('models/model_baseline1.h5')

# predict function
# files_list: string. path of a file, list of clipnames: mp4 to process, ie: '../data/dataset-mp4/_files_lists_/dev.X.list'
# write: boolean. if True, generated clip is written using ffmpeg in outputs folder
# score: boolean. if True, compute MSE according to groundtruth. Groundtruth is assumed to be a mp4 clip whose name is
# equal to clipname where 'X' is replaced by 'Y'
def predict(files_list, write, score): 
    model = load_model('models/model_baseline1.h5')
   
    with open(files_list) as f:
        clips = f.readlines()
        clips = [x.strip() for x in clips]

    scores = []
    for clipname in clips:
        X = data_manager.getAllFrames(clipname)
        Y = model.predict(X)
        length = X.shape[0]
        gtclipname = clipname.replace('X', 'Y')

        if score:
            Ygt = data_manager.getAllFrames(gtclipname)

            if Ygt.shape[0] != length: continue # might happen

            #mse = scoring.MSE(Ygt, Y) 
            mse = np.linalg.norm(Ygt - Y)/np.prod(Y.shape)
            scores.append(mse)
            print(clipname+' MSE = ', mse)
        
        if write:
            data_manager.createVideoClip(Y, 'outputs/B1', os.path.basename(gtclipname))
    
    if score:
        print('Average MSE = ', np.mean(scores))


if __name__ == "__main__":

    batch_size = 8
    epochs = 15
    
    if len(sys.argv) < 2:
        print('usage: python '+sys.argv[0]+' {train|predict} [file_list]')
    
    else:
        model = build_model()
        
        if sys.argv[1] == 'train': 
            train(model, epochs, batch_size)
    
        elif sys.argv[1] == 'predict':
            file_list = '../dataset-mp4/_files_lists_/dev.X.list'
            if len(sys.argv) > 2: file_list = sys.argv[2]
            predict(file_list, True, False)

