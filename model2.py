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
from keras.models import load_model
import skvideo.io
import matplotlib.pyplot as plt

from VGG_loss import *
import pims
from utils import *



F = tf.app.flags.FLAGS

class DCGAN(object):
    def __init__(self, sess):
        self.sess = sess
        self.image_shape = [F.output_size, F.output_size, 3]
        self.build_model()
        self.is_crop = False

    def build_model(self):
        if F.use_tfrecords == True:
            # load images from tfrecords + queue thread runner for better GPU utilization
            tfrecords_filename = ['train_records/' + x for x in os.listdir('train_records/')]
            filename_queue = tf.train.string_input_producer(
                                tfrecords_filename, num_epochs=100)

            self.images_i, self.images_o = reader.read_and_decode(filename_queue, F.batch_size)

            print('image_loaded')
            if F.output_size == 64:
                self.images_i = tf.image.resize_images(self.images_i, (64, 64))
                self.images_o = tf.image.resize_images(self.images_o, (64, 64))
            print(self.images_i.shape)
            print('*********')
            # augmentation
            #self.images_i = tf.image.random_brightness(self.images_i, max_delta=0.5)
            #self.images_i = tf.image.random_contrast(self.images_i, 0.2, 0.5)
            #self.images_i = tf.image.random_hue(self.images_i, 0.2)
           
            # self.images_i = gaussian_noise_layer(self.images_i, 0.2)

            self.images_i = (self.images_i / 127.5) - 1
            self.images_o = (self.images_o / 127.5) - 1

        else:    
            self.images_o = tf.placeholder(tf.float32,
                                       [F.batch_size, F.output_size, F.output_size,
                                        F.c_dim],
                                       name='real_images')
        
        # self.z_gen = tf.placeholder(tf.float32, [None, F.z_dim], name='z')       

        self.is_training = tf.placeholder(tf.bool, name='is_training')        


        self.G_mean = self.generator(self.images_i)
        self.D, self.D_logits = self.discriminator(self.images_o, self.images_i, reuse=False)
        self.D_, self.D_logits_, = self.discriminator(self.G_mean, self.images_i, reuse=True)

        #calculations for getting hard predictions
        # +1 means fooled the D network while -1 mean D has won
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits, labels = tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_, labels = tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.u_loss = tf.reduce_mean(tf.square(self.G_mean - self.images_o))

        self.g_loss_actual = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_, labels = tf.ones_like(self.D_)))

        tf.summary.scalar('loss', self.g_loss_actual)

        # self.g_loss = tf.constant(0)        
        # print('loss calculated')
        # if F.error_conceal == True:
        #     self.mask = tf.placeholder(tf.float32, [F.batch_size] + self.image_shape, name='mask')
        #     self.contextual_loss = tf.reduce_sum(
        #       tf.contrib.layers.flatten(
        #       tf.abs(tf.multiply(self.mask, self.G_mean) - tf.multiply(self.mask, self.images_o))), 1)
        #     self.perceptual_loss = self.g_loss_actual
        #     self.complete_loss = self.contextual_loss + F.lam * self.perceptual_loss
        #     self.grad_complete_loss = tf.gradients(self.complete_loss, self.z_gen)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        """Train DCGAN"""
        global_step = tf.placeholder(tf.int32, [], name="global_step_epochs")

        

        learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        learning_rate_G = tf.train.exponential_decay(F.learning_rate_G, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        learning_rate_U = tf.train.exponential_decay(F.learning_rate_U, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        self.summary_op = tf.summary.merge_all()

        d_optim = tf.train.AdamOptimizer(learning_rate_D, beta1=F.beta1D)\
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate_G, beta1=F.beta1G)\
            .minimize(self.g_loss_actual, var_list=self.g_vars)
        u_optim = tf.train.AdamOptimizer(learning_rate_U, beta1=F.beta1D)\
            .minimize(self.u_loss, var_list=self.g_vars)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        # print('training_started')
        counter = 0
        start_time = time.time()

        if F.load_chkpt:
            try:
                self.load(F.checkpoint_dir)
                print(" [*] Load SUCCESS")
            except:
                print(" [!] Load failed...")
        else:
            print(" [*] Not Loaded")

        self.ra, self.rb = -1, 1

        writer = tf.summary.FileWriter(F.log_dir, graph=tf.get_default_graph())


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)


        epoch = 0
        try:
            while not coord.should_stop():                

                epoch += 1
                idx = 0
                # sample_z_gen = np.random.uniform(
                #   self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
                # # print ('sample generated')
                # errG_actual = self.g_loss_actual.eval()
                # # print('checkpoint 1')
                # E_Fake = self.d_loss_fake.eval({self.z_gen: sample_z_gen})
                # # print('checkpoint 2')
                # E_Real = self.d_loss_real.eval()
                # # print ('loss evaluated')
        
                # Update D network
                iters = 1

                if False: 
                #print('Train D net')
                # print('training disc')
                  train_summary, _,  ulossf = self.sess.run(
                      [self.summary_op, u_optim,  self.u_loss],
                      feed_dict={global_step: epoch, self.is_training: True})
                  writer.add_summary(train_summary, counter)



                if True:
                # if epoch > 10000: 
                  #print('Train D net')
                  # print('training disc')
                  train_summary, _,  dlossf = self.sess.run(
                      [self.summary_op, d_optim,  self.d_loss],
                      feed_dict={global_step: epoch, self.is_training: True})
                writer.add_summary(train_summary, counter)
                 
                 

                # Update G network
                iters = 1
                if True :
                   # print('training generator')
                   # sample_z_gen = np.random.uniform(self.ra, self.rb,
                   #      [F.batch_size, F.z_dim]).astype(np.float32)
                   #print('Train G Net')
                   train_summary, _,   gloss = self.sess.run(
                        [self.summary_op, g_optim, self.g_loss_actual],
                        feed_dict={global_step: epoch, self.is_training: True})
                                    
                writer.add_summary(train_summary, counter)
                  

                errD_fake = self.d_loss_fake.eval(feed_dict={self.is_training: False})
                errD_real = self.d_loss_real.eval(feed_dict={self.is_training: False})
                errG_actual = self.g_loss_actual.eval(feed_dict={self.is_training: False})
                lrateD = learning_rate_D.eval({global_step: epoch})
                lrateG = learning_rate_G.eval({global_step: epoch})
                

                counter += 1
                idx += 1
                if True:                
                  print(("Epoch:[%2d]  d_loss_f:%.8f d_loss_r:%.8f " +
                        "g_loss_act:%.2f ")
                        % (epoch,  errD_fake,
                           errD_real, errG_actual))


                  if np.mod(counter, 500) == 1:
                      # sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
                      samples, d_loss, g_loss_actual = self.sess.run(
                          [self.G_mean, self.d_loss, self.g_loss_actual],
                          feed_dict={self.is_training: False}
                      )
                      save_images(samples, [8, 8],
                                  F.sample_dir + "/sample.png")
                      print("samples saved")
                else:
                  print(("Epoch:[%2d]  " +
                        "g_loss_act:%.2f ")
                        % (epoch, errG_actual))


                if np.mod(counter, 500) == 1:
                    self.save(F.checkpoint_dir)
                    print("Checkpoint saved")
                
                # sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
                if True:
                  samples, d_loss, g_loss_actual = self.sess.run(
                      [self.G_mean, self.d_loss, self.g_loss_actual],
                      feed_dict={self.is_training: False}                     
                  )
                  save_images(samples, [8, 8],
                              F.sample_dir + "/train_{:03d}.png".format(epoch))
                else:
                  samples, g_loss_actual = self.sess.run(
                    [self.G_mean, self.g_loss_actual], feed_dict={self.is_training: False})
                  save_images(samples, [8, 8],
                              F.sample_dir + "/train_{:03d}.png".format(epoch))

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (F.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
        return out.astype(np.float32)


    def discriminator(self, image_model, image_real, reuse=False):
        # print('entered discriminator')
        # print(image_model.shape, image_real.shape)
        image = tf.concat([image_model, image_real], 3)
        with tf.variable_scope('d'):
            dim = 64
            if reuse:
                tf.get_variable_scope().reuse_variables()

            h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
            h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv')))
            h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv')))
            h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv')))
            h4 = lrelu(batch_norm(name='d_bn4')(conv2d(h3, dim * 16, name='d_h4_conv')))
            h4 = tf.reshape(h4, [F.batch_size, -1])
            h5 = linear(h4, 1, 'd_h5_lin')
            return tf.nn.sigmoid(h5), h5

    def generator(self, images_i, reuse=False):
        s = F.output_size
        dim = 64
        k = 5
        #s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
        with tf.variable_scope("g"):
          if reuse:
              tf.get_variable_scope().reuse_variables()
          s2, s4, s8, s16 = int(F.output_size / 2), int(F.output_size / 4), int(F.output_size / 8), int(F.output_size / 16)

          # # z = self.generate_z(images_i)

          h0 = lrelu(conv2d(images_i, dim, 3, 3, 1, 1, name='u_h0_conv'))
          # print(h0.shape)
          h0 = lrelu(conv2d(h0, dim, 3, 3, 1, 1, name='u_h0_conv2'))
          # print(h0.shape)
          # h0 = tf.nn.max_pool(h0, ksize=[1, k, k, 1], strides=[1, k, k, 1],
          #      padding='SAME', name='u_pool_1')
          # print(h0.shape)
          
          h1 = lrelu(batch_norm(name='u_bn1')(conv2d(h0, dim * 2, 3, 3, name='u_h1_conv'), self.is_training))
          print(h1.shape)
          h1 = lrelu(batch_norm(name='u_bn1_2')(conv2d(h1, dim * 2, 3, 3, d_h=1, d_w=1, name='u_h1_conv2'), self.is_training))
          # print(h1.shape)

          h2 = lrelu(batch_norm(name='u_bn2')(conv2d(h1, dim * 4, 3, 3,name='u_h2_conv'), self.is_training))
          h2 = lrelu(batch_norm(name='u_bn2_2')(conv2d(h2, dim * 4, 3, 3, d_h=1, d_w=1, name='u_h2_conv_2'), self.is_training))
          # print(h2.shape)

          h3 = lrelu(batch_norm(name='u_bn3')(conv2d(h2, dim * 8, 3, 3,name='u_h3_conv'), self.is_training))
          h3 = lrelu(batch_norm(name='u_bn3_2')(conv2d(h3, dim * 8, 3, 3,d_h=1, d_w=1, name='u_h3_conv_2'), self.is_training))
          # print(h3.shape)



          h4 = lrelu(batch_norm(name='u_bn4')(conv2d(h3, dim * 16, 3, 3,name='u_h4_conv'), self.is_training))
          h4 = lrelu(batch_norm(name='u_bn4_2')(conv2d(h4, dim * 16, 3, 3,d_h=1, d_w=1, name='u_h4_conv_2'), self.is_training))
          # print(h4.shape)
          
           # resnet structure adopted from srgan
          n = h4
          for i in range(4):
              nn = lrelu(batch_norm(name='u_resb_1_' + str(i))(conv2d(n, dim * 16, 3, 3,d_h=1, d_w=1, name='u_res_conv_1' + str(i)), self.is_training))
              nn = lrelu(batch_norm(name='u_resb_2_' + str(i))(conv2d(nn, dim * 16, 3, 3,d_h=1, d_w=1, name='u_res_conv_2'+ str(i)), self.is_training))
              nn = tf.add(n, nn, name='res_add' + str(i))
              n = nn
          h5 = nn


          g1 = deconv2d(h5, [F.batch_size, s8, s8, dim * 16], k, k, 2, 2, name = 'g_deconv1')
          g1 = tf.nn.relu(batch_norm(name = 'g_bn1')(g1, self.is_training))
          # print('********')
          # print(g1.shape)
          up1 = tf.concat([g1, h3], 3)
          g_conv_1 = lrelu(batch_norm(name='g_bn1_1')(conv2d(up1, dim * 8, 3, 3,d_h=1, d_w=1,name='g_conv_1'), self.is_training))
          g_conv_1 = lrelu(batch_norm(name='g_bn1_2')(conv2d(g_conv_1, dim * 8, 3, 3,d_h=1, d_w=1,name='g_conv_1_2'), self.is_training))


          g2 = deconv2d(g_conv_1, [F.batch_size, s4, s4, dim * 8], k, k, 2, 2, name = 'g_deconv2')
          g2 = tf.nn.relu(batch_norm(name = 'g_bn2')(g2, self.is_training))
          # print(g2.shape)
          up2 = tf.concat([g2, h2], 3)
          g_conv_2 = lrelu(batch_norm(name='g_bn2_1')(conv2d(up2, dim * 4, 3, 3,d_h=1, d_w=1,name='g_conv_2'), self.is_training))
          g_conv_2 = lrelu(batch_norm(name='g_bn2_2')(conv2d(g_conv_2, dim * 4, 3, 3,d_h=1, d_w=1,name='g_conv_2_2'), self.is_training))




          g3 = deconv2d(g_conv_2, [F.batch_size, s2, s2, dim * 4], k, k, 2, 2, name = 'g_deconv3')
          g3 = tf.nn.relu(batch_norm(name = 'g_bn3')(g3, self.is_training))
          # print(g3.shape)
          up3 = tf.concat([g3, h1], 3)
          g_conv_3 = lrelu(batch_norm(name='g_bn3_1')(conv2d(up3, dim * 2, 3, 3,d_h=1, d_w=1,name='g_conv_3'), self.is_training))
          g_conv_3 = lrelu(batch_norm(name='g_bn3_2')(conv2d(g_conv_3, dim * 2, 3, 3,d_h=1, d_w=1,name='g_conv_3_2'), self.is_training))


          g4 = deconv2d(g_conv_3, [F.batch_size, F.output_size, F.output_size, dim * 2], k, k, 2, 2, name = 'g_deconv4')
          g4 = tf.nn.relu(batch_norm(name = 'g_bn4')(g4, self.is_training))
          # print(g4.shape)
          up4 = tf.concat([g4, h0], 3)
          g_conv_4 = lrelu(batch_norm(name='g_bn4_1')(conv2d(up4, dim , 3, 3,d_h=1, d_w=1,name='g_conv_4'), self.is_training))
          g_conv_4 = lrelu(batch_norm(name='g_bn4_2')(conv2d(g_conv_4, dim , 3, 3,d_h=1, d_w=1,name='g_conv_4_2'), self.is_training))


          g5 = deconv2d(g_conv_4, [F.batch_size, F.output_size, F.output_size, 3], k, k, 1,1, name ='g_deconv5')
          g5 = tf.nn.tanh(g5, name = 'g_tanh')
          return g5


    # def predict(self):

    #   init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #   self.sess.run(init_op)
    #   classifier = load_model('Track2/starting_kit_v2/models/model_baseline2_clf.h5')

    #   counter = 0
    #   if F.load_chkpt:
    #       try:
    #           self.load(F.checkpoint_dir)
    #           print(" [*] Checkpoint Load Success !!!")
    #       except:
    #           print(" [!] Checkpoint Load failed !!!!")
    #   else:
    #       print(" [*] Not Loaded")

    #   print(self.images_i.get_shape)
    #   print('*******************')
    #   vid_path = '../Inpaintin/dev/X/'
    #   ilist = os.listdir(vid_path)
    #   counter_video = 0
    #   for vid in range(len(ilist)):
    #     counter += 1
    #     print("Running " + str(counter) + "video ")
    #     f_ri_i = skvideo.io.vread(vid_path + ilist[vid])

    #     batch_size = F.batch_size
    #     o_list = []
    #     print('***************')
    #     f_ri = np.zeros((len(f_ri_i),F.output_size,F.output_size,F.c_dim)).tolist()
    #     for x in range(len(f_ri_i)):
    #       f_ri[x] = transform(f_ri_i[x], is_crop=True)
    #     # f_ri = f_ri.tolist()

    #     epoch = 0

    #     batch_temp = 0 # used to store the size of incompelete batch
    #     id = 0
    #     fsize = F.output_size
    #     for idx in range(0,len(f_ri),batch_size):
    #       batch = []
    #       if idx+batch_size<len(f_ri):
    #         # print(idx)
    #         batch = f_ri[idx:idx+batch_size]
    #         # F.batch_size = len(batch)
    #         # print(len(batch))
    #       else:
    #         batch = f_ri[idx:]
    #         # F.batch_size = len(batch)
    #         batch_temp = len(batch)
    #         temp = np.zeros(( (batch_size-len(batch)) , F.output_size, F.output_size, F.c_dim))
    #         batch = np.append(batch, temp, axis=0)

    #       epoch +=1
    

    #       t_inp, t_out = self.sess.run([self.images_i, self.output],feed_dict={self.images_i: batch, self.is_training: False})
    #       # save_images(t_inp, [8, 8], './in_images/'+ str(epoch) + '.png')
    #       t_inp = np.array(((t_inp +1.)*(127.5)), dtype=np.uint8)
    #       t_out = np.array(((t_out +1.)*(127.5)), dtype=np.uint8)
          
    #       t_inp_b = np.array(t_inp/ 255. )
    #       t_out_b = np.array(t_out/ 255. )

    #       length = t_inp_b.shape[0]
    #       X = t_inp_b.reshape(-1, fsize//32,32,fsize//32,32, 3).swapaxes(2,3).reshape(-1,32,32,3) 
    #       T = classifier.predict(X)
    #       idxs = np.where(T < 0.2)[0]
    #       Xidxs = X[idxs]

      
    #       T = np.reshape(T, (-1, (fsize//32)*(fsize//32), 1))
    #       Y = np.zeros((length, (fsize//32)*(fsize//32), 32, 32, 3))
          
    #       k = 0
    #       for i in range((fsize//32)):
    #         for j in range((fsize//32)):
    #           Y[:,k,:,:,:] = t_out_b[:, i*32:(i+1)*32, j*32:(j+1)*32, :]
    #           k += 1
         
       
    #       X = np.reshape(X, (-1, (fsize//32)*(fsize//32), 32, 32, 3))
    #       Xidxs = np.reshape(Xidxs, (-1, 32, 32, 3))
          
    #       # 1: get channels normalization ranges on non-text parts
    #     #   maxr = np.max(Xidxs[:,:,:,0]) 
    #     #   maxg = np.max(Xidxs[:,:,:,1])
    #     #   maxb = np.max(Xidxs[:,:,:,2])
    #     #   minr = np.min(Xidxs[:,:,:,0])
    #     #   ming = np.min(Xidxs[:,:,:,1])
    #     #   minb = np.min(Xidxs[:,:,:,2])

    #     # # 2: normalize predicted parts ..
    #     #   Yr = np.clip(Y[:,:,:,:,0], minr,maxr) 
    #     #   Yg = np.clip(Y[:,:,:,:,1], ming,maxg)
    #     #   Yb = np.clip(Y[:,:,:,:,2], minb,maxb)
    #     #   Y = np.stack((Yr,Yg,Yb), axis=-1)

    #     # build inpainted video clip
    #       clip = np.ndarray((length, fsize, fsize, 3), dtype='float32')          
    #       for i in range(length): # for each frames
    #         for j in range((fsize//32)*(fsize//32)):
    #             x = int(j // int(fsize/32))
    #             y = int(j % int(fsize/32))
                
    #             # copy either predicted patch, or input patch, according to classifier output
    #             if T[i,j] > 0.5:
    #                 clip[i, x*32:(x+1)*32, y*32:(y+1)*32, :] = Y[i,j]
    #             else:
    #                 clip[i, x*32:(x+1)*32, y*32:(y+1)*32, :] = X[i,j]
    #                 # name = raw_input("name 2")
    #       t_out = clip*255          

    #       if(batch_temp != 0):
    #         t_inp = t_inp[:batch_temp]
    #         t_out = t_out[:batch_temp]
    #       # print(t_out[:2])
          
    #       for im_i, im_o in zip(t_inp, t_out):

    #         id += 1


    #         scipy.misc.imsave('./in_images/'+ str(id) + '.png', im_i)
    #         scipy.misc.imsave('./out_images/' + str(id) + '.png', im_o)
    #         o_list.append(im_o)


    #       # save_images(t_out, [8, 8], './out_images/' + str(epoch) + '.png')
          

    #     # def make_frame(t):
    #         # return o_list[t]

    #     time_i = time.time()
    #     skvideo.io.vwrite("./out_video/" + str(ilist[vid]).replace('X', 'Y'), o_list) 
    #     time_f = time.time()
    #     print("Elapsed time: " + str(time_f - time_i))



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



