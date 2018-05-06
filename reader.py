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
                                     num_threads=6,
                                     capacity=1000,
                                     min_after_dequeue=270)
    
    return images_i, images_o

if __name__ =='__main__':
    # tfrecords_filename = ['./train_records/' + x for x in os.listdir('train_records/')]
    tfrecords_filename = ['./train_records/I_O4.tfrecords' ]
    # print (tfrecords_filename)#['data_records/celebA_0.tfrecords','data_records/celebA_1.tfrecords', 'data_records/celebA_2.tfrecords']
    filename_queue = tf.train.string_input_producer(
                            tfrecords_filename, num_epochs=1)

    images, outs = read_and_decode(filename_queue, 64)


    # init_op = tf.group(tf.global_variables_initializer(),
    #                tf.local_variables_initializer())
    # sess.run(init_op)
    with tf.Session() as sess:
      # Start populating the filename queue.
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        sess.run(init_op)
        # tf.initialize_local_variables()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        cnt = 0
        # time.sleep(30)
        print ('here')
        try:
            while not coord.should_stop():
            
            # for i in range():
                # Retrieve a single instance:
                print ('vfj')
                x1, x2 = sess.run([images, outs])
                print (cnt, x1.shape, x2.shape)
                cnt += 1 
                # scipy.misc.imsave('samples_complete/img' + str(i) + '.png', example[0])
                # scipy.misc.imsave('samples_complete/ky' + str(i) + '.png', label[0])
                # print ('Done')

        except tf.errors.OutOfRangeError:
            print('Done training for')
        finally:
    #   # When done, ask the threads to stop.
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
