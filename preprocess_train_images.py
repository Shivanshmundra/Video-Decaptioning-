# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
#import dlib
import glob
#import h5py
#from skimage import io
import time
import numpy as np
import collections
from imutils import face_utils
#import cv2
#from scipy.misc import imsave, imresize
from moviepy.editor import *

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(im_i, im_o, name):
    """Converts a dataset to tfrecords."""
    rows = im_i.shape[1]
    cols = im_i.shape[2]
    depth = im_i.shape[3]

    filename = os.path.join(save_path, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(im_i.shape[0]):
        image_raw_i = im_i[index].tostring()
        image_raw_o = im_o[index].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'image_raw_i': _bytes_feature(image_raw_i),
            'image_raw_o': _bytes_feature(image_raw_o)}))

        writer.write(example.SerializeToString())

    writer.close()

if __name__ =='__main__':
    save_path = 'train_records/'
    
    imp_path = '../Inpaintin/train/X/'
    out_path = '../Inpaintin/train/Y/'

    face_image_list = os.listdir(imp_path)  # dirs of listed videos
    counter = 0
    ilist = os.listdir(imp_path)
    olist = [x.replace('X', 'Y') for x in ilist]
    tfrecord_ind = 0
    il = []
    ol = []
    # print ('1')

    for im in range(len(ilist)):
        # print ('2')
        counter += 1
        video_i = VideoFileClip(imp_path + ilist[im])
        video_o = VideoFileClip(out_path + olist[im])
        

        # print ('2')
        f_ri =  [x for x in video_i.iter_frames()]
        f_ro =  [x for x in video_o.iter_frames()]


        # print ('3')
        for x in range(0, len(f_ri), 15):
            # print (f_ri[x].shape)
            if f_ri[x].shape == (128, 128,3):
                # print('here')
                il.append(f_ri[x])
                ol.append(f_ro[x])

        if len(il) > 10000:
            convert_to(np.asarray(il), np.asarray(ol), 'I_O'  + str(tfrecord_ind)) 
            il, ol = [], []
            tfrecord_ind += 1
      
    convert_to(np.asarray(il), np.asarray(ol), 'I_O'  + str(tfrecord_ind))  
    # convert_to(np.asarray(image_list), 'celebA_' + str(tfrecord_ind))
