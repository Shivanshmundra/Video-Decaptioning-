from __future__ import division
import os
import sys
import time
import reader
import random
import vggface
from ops import *
import scipy.misc
import numpy as np
import tensorflow as tf
from six.moves import xrange
from skimage.measure import compare_psnr
from skimage.measure import compare_mse
from tensorflow.python.platform import gfile

F = tf.app.flags.FLAGS

class DCGAN(object):
    def __init__(self, sess):
        self.sess = sess
        self.ngf = 128
        self.ndf = 64
        self.nt = 128
        self.k_dim = 16
        self.image_shape = [F.output_size, F.output_size, 3]
        self.build_model()
        if F.output_size == 64:
            self.is_crop = True
        else:
            self.is_crop = False

    def build_model(self):
        # main method for training the conditional GAN
        if F.use_tfrecords == True:
            # load images from tfrecords + queue thread runner for better GPU utilization
            tfrecords_filename = ['train_records/' + x for x in os.listdir('train_records/')]
            filename_queue = tf.train.string_input_producer(
                                tfrecords_filename, num_epochs=100)

            self.images_i, self.images_o = reader.read_and_decode(filename_queue, F.batch_size)

            if F.output_size == 64:
                self.images_i = tf.image.resize_images(self.images_i, (64, 64))
                self.images_o = tf.image.resize_images(self.images_o, (64, 64))

            self.images_i = (self.images_o / 127.5) - 1

        else:    
            self.images_i = tf.placeholder(tf.float32,
                                       [F.batch_size, F.output_size, F.output_size,
                                        F.c_dim],
                                       name='real_images_i')
            self.images_o = tf.placeholder(tf.float32,
                                        [F.batch_size, F.output_size, F.output_size,
                                            F.c_dim],
                                       name='real_images_o')
        
        # self.mask = tf.placeholder(tf.float32, [F.batch_size, F.output_size, F.output_size, 3], name='mask')
        self.is_training = tf.placeholder(tf.bool, name='is_training')        
        # self.get_z_init = tf.placeholder(tf.bool, name='get_z_init')

        self.output = self.unet(self.images_i)
        # self.images_ = tf.multiply(self.mask, self.images)
        # self.z_gen = tf.cond(self.get_z_init, lambda: self.generate_z(self.images_), lambda: tf.placeholder(tf.float32, [F.batch_size, 100], name='z_gen'))

        # self.G = self.generator(self.z_gen)
        self.loss = 0

        if F.vgg_loss == True:
            ## Change VGGFace to VGG net 
            vgg_net_inp = tf.concat([self.output, self.images], 0)
            print (vgg_net_inp.get_shape())

            #Reduce mean values from pixel values
            rgb_mean = tf.constant([129.18628, 104.76238,  93.59396], dtype=tf.float32)
            rgb_mean = tf.reshape(rgb_mean, [1, 1, 1, 3])

            vgg_net_inp = vgg_net_inp - rgb_mean

            vgg_net = vggface.vgg_face('vgg-face.mat', vgg_net_inp)

            self.loss += tf.reduce_mean(tf.square(vgg_net['relu3_3'][:F.batch_size] - vgg_net['relu3_3'][F.batch_size:]))# + \
                        # tf.reduce_mean(tf.square(vgg_net[:F.batch_size] - vgg_net[F.batch_size:]))
        else:
            self.loss += tf.reduce_sum(tf.square(self.output - self.images_o))

        tf.summary.scalar('loss', self.loss)
            
        # create summaries  for Tensorboard visualization
        # self.g_loss = tf.constant(0) 

        t_vars = tf.trainable_variables()
        self.z_vars = [var for var in t_vars if 'U/z_' in var.name]
        # self.g_vars = [var for var in t_vars if 'G/g_' in var.name]
        # self.d_vars = [var for var in t_vars if 'G/d_' in var.name]

        # self.saver_gen = tf.train.Saver(self.g_vars) # + self.d_vars)
        self.saver = tf.train.Saver(self.z_vars)

    def train_unet(self):    
        # main method for training conditonal GAN
        global_step = tf.placeholder(tf.int32, [], name="global_step_iterations")

        learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        
        self.summary_op = tf.summary.merge_all()

        z_optim = tf.train.AdamOptimizer(0.001, beta1=F.beta1D)\
          .minimize(self.loss, var_list=self.z_vars)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        start_time = time.time()

        # self.load_G("checkpoint/celebA/model_weights_" + str(F.output_size))

        if F.load_chkpt:
            try:
                self.load(F.checkpoint_dir)
                print(" [*] Checkpoint Load Success !!!")
            except:
                print(" [!] Checkpoint Load failed !!!!")
        else:
            print(" [*] Not Loaded")

        self.ra, self.rb = -1, 1
        counter = 1
        step = 1
        idx = 1

        writer = tf.summary.FileWriter(F.log_dir, graph=tf.get_default_graph())

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            while not coord.should_stop():
                start_time = time.time()
                step += 1

                masks = self.next_mask()
                train_summary, _, zloss = self.sess.run(
                        [self.summary_op, z_optim,  self.loss],
                        feed_dict={global_step: counter, self.mask: masks, self.is_training: True})
                writer.add_summary(train_summary, counter)

                print(("Iteration: [%6d] mse loss:%.2e")
                      % (idx, zloss))
                 
                # periodically save checkpoints for future loading
                if np.mod(counter, F.saveInterval) == 1:
                    # self.save(F.checkpoint_dir, counter)
                    print("Checkpoint saved successfully !!!")
                    save_imgs, save_opts = self.sess.run([self.images_, self.G], feed_dict={global_step: counter, self.mask: masks, self.is_training: False, self.get_z_init: True})

                    #### Write code to save samples to visualise
                    
                counter += 1
                idx += 1
                
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (F.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
        return out.astype(np.float32)

    def train(self):    
        # main method for training conditonal GAN
        global_step = tf.placeholder(tf.int32, [], name="global_step_iterations")

        learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        
        self.summary_op = tf.summary.merge_all()

        z_optim = tf.train.AdamOptimizer(0.001, beta1=F.beta1D)\
          .minimize(self.loss, var_list=self.z_vars)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        start_time = time.time()

        self.load_G("checkpoint/celebA/model_weights_" + str(F.output_size))

        if F.load_chkpt:
            try:
                self.load(F.checkpoint_dir)
                print(" [*] Checkpoint Load Success !!!")
            except:
                print(" [!] Checkpoint Load failed !!!!")
        else:
            print(" [*] Not Loaded")

        self.ra, self.rb = -1, 1
        counter = 1
        step = 1
        idx = 1

        writer = tf.summary.FileWriter(F.log_dir, graph=tf.get_default_graph())

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            while not coord.should_stop():
                start_time = time.time()
                step += 1

                masks = self.next_mask()
                train_summary, _, zloss = self.sess.run(
                        [self.summary_op, z_optim,  self.loss],
                        feed_dict={global_step: counter, self.mask: masks, self.is_training: True, self.get_z_init: True})
                writer.add_summary(train_summary, counter)

                print(("Iteration: [%6d] mse loss:%.2e")
                      % (idx, zloss))
                 
                # periodically save checkpoints for future loading
                if np.mod(counter, F.saveInterval) == 1:
                    # self.save(F.checkpoint_dir, counter)
                    print("Checkpoint saved successfully !!!")
                    save_imgs, save_opts = self.sess.run([self.images_, self.G], feed_dict={global_step: counter, self.mask: masks, self.is_training: False, self.get_z_init: True})

                    save_images(save_imgs, [8, 8], "z_gens/samples_imgs" + str(counter) + ".png")
                    save_images(save_opts, [8, 8], "z_gens/samples_opts" + str(counter) + ".png")

                counter += 1
                idx += 1
                
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (F.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
        return out.astype(np.float32)

    def unet(self, images_i):
        dim = 64
        k = 5
        with tf.variable_scope("U"):
              s2, s4, s8, s16 = int(F.output_size / 2), int(F.output_size / 4), int(F.output_size / 8), int(F.output_size / 16)

              h0 = linear(z, s16 * s16 * dim * 16, 'g_lin')
              h0 = tf.reshape(h0, [F.batch_size, s16, s16, dim * 16])

              h1 = deconv2d(h0, [F.batch_size, s8, s8, dim * 8], k, k, 2, 2, name = 'g_deconv1')
              h1 = tf.nn.relu(batch_norm(name = 'g_bn1')(h1, self.is_training))
                      
              h2 = deconv2d(h1, [F.batch_size, s4, s4, dim * 4], k, k, 2, 2, name = 'g_deconv2')
              h2 = tf.nn.relu(batch_norm(name = 'g_bn2')(h2, self.is_training))

              h3 = deconv2d(h2, [F.batch_size, s2, s2, dim * 2], k, k, 2, 2, name = 'g_deconv4')
              h3 = tf.nn.relu(batch_norm(name = 'g_bn3')(h3, self.is_training))

              h4 = deconv2d(h3, [F.batch_size, F.output_size, F.output_size, 3], k, k, 2, 2, name ='g_hdeconv5')
              h4 = tf.nn.tanh(h4, name = 'g_tanh')
              return h4

    def generator(self, z):
        dim = 64
        k = 5
        with tf.variable_scope("G"):
              s2, s4, s8, s16 = int(F.output_size / 2), int(F.output_size / 4), int(F.output_size / 8), int(F.output_size / 16)

              h0 = linear(z, s16 * s16 * dim * 16, 'g_lin')
              h0 = tf.reshape(h0, [F.batch_size, s16, s16, dim * 16])

              h1 = deconv2d(h0, [F.batch_size, s8, s8, dim * 8], k, k, 2, 2, name = 'g_deconv1')
              h1 = tf.nn.relu(batch_norm(name = 'g_bn1')(h1, self.is_training))
                      
              h2 = deconv2d(h1, [F.batch_size, s4, s4, dim * 4], k, k, 2, 2, name = 'g_deconv2')
              h2 = tf.nn.relu(batch_norm(name = 'g_bn2')(h2, self.is_training))

              h3 = deconv2d(h2, [F.batch_size, s2, s2, dim * 2], k, k, 2, 2, name = 'g_deconv4')
              h3 = tf.nn.relu(batch_norm(name = 'g_bn3')(h3, self.is_training))

              h4 = deconv2d(h3, [F.batch_size, F.output_size, F.output_size, 3], k, k, 2, 2, name ='g_hdeconv5')
              h4 = tf.nn.tanh(h4, name = 'g_tanh')
              return h4

    def discriminator(self, image, reuse=False):
        with tf.variable_scope('D'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            dim = 64
            if F.output_size == 128:
                  h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
                  h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv'), self.is_training))
                  h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv'), self.is_training))
                  h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv'), self.is_training))
                  h4 = lrelu(batch_norm(name='d_bn4')(conv2d(h3, dim * 16, name='d_h4_conv'), self.is_training))
                  h4 = tf.reshape(h4, [F.batch_size, -1])
                  h5 = linear(h4, 1, 'd_h5_lin')
                  return tf.nn.sigmoid(h5), h5

            else:
                  h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
                  h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv'), self.is_training))
                  h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv'), self.is_training))
                  h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv'), self.is_training))
                  h4 = tf.reshape(h3, [F.batch_size, -1])
                  h5 = linear(h4, 1, 'd_h5_lin')
                  return tf.nn.sigmoid(h5), h5

    def generate_z(self, image):
        with tf.variable_scope('Z'):
            dim = 64
            if F.output_size == 128:
                  h0 = lrelu(conv2d(image, dim, name='z_h0_conv'))
                  h1 = lrelu(batch_norm(name='z_bn1')(conv2d(h0, dim * 2, name='z_h1_conv'), self.is_training))
                  h2 = lrelu(batch_norm(name='z_bn2')(conv2d(h1, dim * 4, name='z_h2_conv'), self.is_training))
                  h3 = lrelu(batch_norm(name='z_bn3')(conv2d(h2, dim * 8, name='z_h3_conv'), self.is_training))
                  h4 = lrelu(batch_norm(name='z_bn4')(conv2d(h3, dim * 16, name='z_h4_conv'), self.is_training))
                  h4 = tf.reshape(h4, [F.batch_size, -1])
                  h5 = linear(h4, 100, 'z_h5_lin')
                  return tf.nn.tanh(h5)

            else:
                  h0 = lrelu(conv2d(image, dim, name='z_h0_conv'))
                  h1 = lrelu(batch_norm(name='z_bn1')(conv2d(h0, dim * 2, name='z_h1_conv'), self.is_training))
                  h2 = lrelu(batch_norm(name='z_bn2')(conv2d(h1, dim * 4, name='z_h2_conv'), self.is_training))
                  h3 = lrelu(batch_norm(name='z_bn3')(conv2d(h2, dim * 8, name='z_h3_conv'), self.is_training))
                  h4 = tf.reshape(h3, [F.batch_size, -1])
                  h5 = linear(h4, 100, 'z_h5_lin')
                  return tf.nn.tanh(h5)

    def save(self, checkpoint_dir, step=0):
        model_name = "model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load_G(self, checkpoint_dir):
        print(" [*] Reading checkpoints of G...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_gen.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
