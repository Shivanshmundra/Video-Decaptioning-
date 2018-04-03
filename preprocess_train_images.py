from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import dlib
import glob
import h5py
from skimage import io
import time
import numpy as np
import collections
from imutils import face_utils
import cv2
from scipy.misc import imsave, imresize

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(images,  name):
    """Converts a dataset to tfrecords."""
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(save_path, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(images.shape[0]):
        image_raw = images[index].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'image_raw': _bytes_feature(image_raw)}))
        
        writer.write(example.SerializeToString())

    writer.close()

if __name__ =='__main__':
    if len(sys.argv) != 2:
        print(
            
            " python prepare_bb_land.py  shape_predictor_68_face_landmarks.dat "
            "You can download a trained facial shape predictor from:\n"
            "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        exit()

    predictor_path = sys.argv[1]
    
    save_path = 'train_records/'
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    images_dir_path = '../../PreGan/data/data/celebA/img_align_celeba/'

    face_image_list = os.listdir(images_dir_path)  # dir of extracted faces
    counter = 0

    image_list = []
    tfrecord_ind = 0

    for imgs in face_image_list:
        counter += 1

        filename = os.path.join(images_dir_path, imgs) 
          
        img = io.imread(filename)
        arr = np.array(img) 
        H, W, C = arr.shape   # we assume that we are getting face cropped images

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        #print("Number of faces detected: {}".format(len(dets)))
        
        for k, d in enumerate(dets):
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            shape = face_utils.shape_to_np(shape)
            try:
                face_part = img[d.top():d.bottom(), d.left():d.right()]
                # print (face_part.shape)
                face_part = imresize(face_part, (128,128, 3))
                image_list.append(face_part)
            except:
                continue
            if len(image_list) == 10000:
                convert_to(np.asarray(image_list), 'celebA_' + str(tfrecord_ind))
                image_list = []
                tfrecord_ind += 1
        
    convert_to(np.asarray(image_list), 'celebA_' + str(tfrecord_ind))
