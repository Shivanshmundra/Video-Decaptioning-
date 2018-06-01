#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Reshape, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import tensorflow as tf
import os, sys
import data_manager #, scoring

fsize = 128 # 256

# network architecture
def build_model():
    # encoder 
    input_shape = (32,32,3)
    input_img = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (3, 3), strides=(2,2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(1,1), activation='relu', padding='same')(x)
    encoded = BatchNormalization()(x)
    # here representation is (8, 8, 128) 

    # decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # classifier
    x = MaxPooling2D()(encoded)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    classified = Dense(1, activation='sigmoid')(x)

    # compile
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['mse'])
    print(autoencoder.summary())

    classifier = Model(input_img, classified)
    classifier.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    print(classifier.summary())

    return autoencoder, classifier


# training loop
def train(models, nb_epochs):
    
    # load minibatches from pickle
    train_batches = data_manager.load_batches(0, 3500)
    nb_batches = len(train_batches) 
    
    autoencoder = models[0]
    classifier = models[1]

    # continue training from current state
    #autoencoder = load_model('models/model_baseline2_ae.h5')
    #classifier = load_model('models/model_baseline2_clf.h5')
    
    min_loss = 10000000000.
    for epoch in range(nb_epochs):
        aelosses = []
        clflosses = []
        for batch in range(nb_batches):
            # get minibatch and shuffle it
            X, Y, T = train_batches[batch] 
            rand_idx = list(range(len(T)))
            np.random.shuffle(rand_idx)
            X = X[rand_idx]
            Y = Y[rand_idx]
            T = T[rand_idx]
    
            # train encoder-decoder using patches that include text 
            text_idxs = np.where(T > 0)[0]
            loss = autoencoder.train_on_batch(X[text_idxs], Y[text_idxs])
            aelosses.append(loss)
            
            # train patch classifier text/not-text
            loss = classifier.train_on_batch(X, T)
            clflosses.append(loss)
        
        # print losses
        aelosses = np.mean(np.array(aelosses), axis=0)
        loss_ae, mse_ae = aelosses.tolist()
        print('decoder loss: ', loss_ae, '\t decoder mse: ', mse_ae)
        clflosses = np.mean(np.array(clflosses), axis=0)
        loss_clf, acc_clf = clflosses.tolist()
        print('classifier loss: ', loss_clf, '\t classifier accuracy: ', acc_clf)
        
        # save models
        autoencoder.save('models/model_baseline2_ae.h5')
        classifier.save('models/model_baseline2_clf.h5')
        if loss_clf + loss_ae < min_loss:
            min_loss = loss_clf + loss_ae
            print('save best model at epoch ', epoch)
            autoencoder.save('models/model_baseline2_ae_best.h5')
            classifier.save('models/model_baseline2_clf_best.h5')
    
    return autoencoder, classifier


# predict function
# files_list: string. path of a file, list of clipnames: mp4 to process, ie:
# '../data/dataset-mp4/_files_lists_/dev.X.list'
# write: boolean. if True, generated clip is written using ffmpeg in outputs folder
# score: boolean. if True, compute MSE according to groundtruth. Groundtruth is assumed to be a mp4 clip whose name is
# equal to clipname where 'X' is replaced by 'Y'
def predict(files_list, write, score):
    autoencoder = load_model('models/model_baseline2_ae.h5')
    classifier = load_model('models/model_baseline2_clf.h5')
   
    # load list of mp4 files to process
    with open(files_list) as f:
        clips = f.readlines()
    clips = [x.strip() for x in clips] 

    scores = []
    for clipname in clips:

        X = data_manager.getAllFrames(clipname)
        length = X.shape[0]

        # get non-overlap patches and normalize (rescaling to 0-1)
        X = X.reshape(-1, fsize//32,32,fsize//32,32, 3).swapaxes(2,3).reshape(-1,32,32,3) 
        # make predictions from patches in X
        Y = autoencoder.predict(X)
        T = classifier.predict(X)

        # get patch idx with no text (classifier's output below 0.2)
        idxs = np.where(T < 0.2)[0]
        Xidxs = X[idxs]
        
        # reshape predictions
        T = np.reshape(T, (-1, (fsize//32)*(fsize//32), 1))
        Y = np.reshape(Y, (-1, (fsize//32)*(fsize//32), 32, 32, 3))
        X = np.reshape(X, (-1, (fsize//32)*(fsize//32), 32, 32, 3))
        Xidxs = np.reshape(Xidxs, (-1, 32, 32, 3))

        # post-processing: normalize pixels values of generated parts
        # according to non-text pixels ranges

        # 1: get channels normalization ranges on non-text parts
        maxr = np.max(Xidxs[:,:,:,0]) 
        maxg = np.max(Xidxs[:,:,:,1])
        maxb = np.max(Xidxs[:,:,:,2])
        minr = np.min(Xidxs[:,:,:,0])
        ming = np.min(Xidxs[:,:,:,1])
        minb = np.min(Xidxs[:,:,:,2])

        # 2: normalize predicted parts ..
        Yr = np.clip(Y[:,:,:,:,0], minr,maxr) 
        Yg = np.clip(Y[:,:,:,:,1], ming,maxg)
        Yb = np.clip(Y[:,:,:,:,2], minb,maxb)
        Y = np.stack((Yr,Yg,Yb), axis=-1)

        # build inpainted video clip
        clip = np.ndarray((length, fsize, fsize, 3), dtype='float32')
        
        for i in range(length): # for each frames
            for j in range((fsize//32)*(fsize//32)):
                x = int(j // int(fsize/32))
                y = int(j % int(fsize/32))
                
                # copy either predicted patch, or input patch, according to classifier output
                if T[i,j] > 0.5:
                    clip[i, x*32:(x+1)*32, y*32:(y+1)*32, :] = Y[i,j]
                else:
                    clip[i, x*32:(x+1)*32, y*32:(y+1)*32, :] = X[i,j]
        
        gtclipname = clipname.replace('X', 'Y')
        if score:
            Ygt = data_manager.getAllFrames(gtclipname)
            
            if Ygt.shape[0] != length: continue # might happen

            #mse = scoring.MSE(Ygt, clip)  
            mse = np.linalg.norm(Ygt - clip)/np.prod(clip.shape)
            scores.append(mse)
            print(clipname+' MSE = ', mse)

        if write:
            data_manager.createVideoClip(clip, 'outputs/B2', os.path.basename(gtclipname))

    # scoring
    if score:
        print('Average MSE = ', np.mean(scores))


if __name__ == "__main__":
    
    nb_epochs = 150

    if len(sys.argv) < 2:
        print('usage: python '+sys.argv[0]+' {train|predict} [file_list]')
    
    else: 
        models = build_model()
        
        if sys.argv[1] == 'train': 
            models = train(models, nb_epochs)

        elif sys.argv[1] == 'predict':
            file_list = '../dataset-mp4/_files_lists_/dev.X.list'
            if len(sys.argv) > 2: file_list = sys.argv[2]
            predict(file_list, True, False)

