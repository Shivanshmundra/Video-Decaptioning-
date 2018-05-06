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
import cv2
#from scipy.misc import imsave, imresize
from moviepy.editor import *

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



if __name__ =='__main__':
    
    
    vid_path = '../Inpaintin/dev/X/'


    counter = 0
    ilist = os.listdir(vid_path)
    # olist = [x.replace('X', 'Y') for x in ilist]
    # tfrecord_ind = 0
    il = []
    # ol = []
    # print ('1')

    # for im in range(len(ilist)):
    for im in range(1):
        # print ('2')
        counter += 1
        video_i = VideoFileClip(vid_path + ilist[im])
        # video_o = VideoFileClip(out_path + olist[im])
        

        # print ('2')
        f_ri =  [x for x in video_i.iter_frames()]
        # f_ro =  [x for x in video_o.iter_frames()]


        # print ('3')
        for x in range(0, len(f_ri)):
            # print (f_ri[x].shape)
            if f_ri[x].shape == (128, 128,3):
                # print('here')
                cv2.imwrite('./out_vid_im/'+str(ilist[im])+ "_" + str(x)+ ".png", f_ri[x]) 
                # ol.append(f_ro[x])
            else:
                print("Image is not of correct size")
    #     if len(il) > 10000:
    #         convert_to(np.asarray(il), np.asarray(ol), 'I_O'  + str(tfrecord_ind)) 
    #         il, ol = [], []
    #         tfrecord_ind += 1
      
    # convert_to(np.asarray(il), np.asarray(ol), 'I_O'  + str(tfrecord_ind))  
    # convert_to(np.asarray(image_list), 'celebA_' + str(tfrecord_ind))
