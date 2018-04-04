from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import scipy.misc

import tensorflow as tf

def read_and_decode(filename_queue, batch_size):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'image_raw_i': tf.FixedLenFeature([], tf.string),
        'image_raw_o': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image_i = tf.decode_raw(features['image_raw_i'], tf.uint8)
    image_o = tf.decode_raw(features['image_raw_o'], tf.uint8)
   
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    image_shape = tf.stack([height, width, depth])
    
    image_i = tf.reshape(tf.cast(image_i, tf.float32), image_shape)
    image_o = tf.reshape(tf.cast(image_o, tf.float32), image_shape)
    
    image_i.set_shape((128, 128, 3))
    image_o.set_shape((128, 128, 3))
    
    images_i, images_o = tf.train.shuffle_batch([image_i, image_o],
                                     batch_size=batch_size,
                                     num_threads=16,
                                     capacity=10000,
                                     min_after_dequeue=1000)
    
    return images_i, images_o