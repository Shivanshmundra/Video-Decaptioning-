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
        self.ngf = 128
        self.ndf = 64
        self.nt = 128
        self.k_dim = 16
        self.image_shape = [F.output_size, F.output_size, F.c_dim]
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
            self.images_i = tf.placeholder(tf.float32,
                                       [F.batch_size, F.output_size, F.output_size,
                                        F.c_dim],
                                       name='real_images_i')
            self.images_o = tf.placeholder(tf.float32,
                                        [F.batch_size, F.output_size, F.output_size,
                                            F.c_dim_o],
                                       name='real_images_o')
        
        # self.mask = tf.placeholder(tf.float32, [F.batch_size, F.output_size, F.output_size, 3], name='mask')
        #here the function of mask is to be added so that mask is extracted from image
        # this mask needs to be the subtitles in images.
        # self.mask = self.images_i
        # just for running code I am using mask same as input image.
        # for img in range(F.batch_size):
        #     mask = get_mask(self.images_i[img])
        #     self.images_i[img] = tf.multiply(mask, self.images_i[img])

        self.is_training = tf.placeholder(tf.bool, name='is_training')        
        # self.get_z_init = tf.placeholder(tf.bool, name='get_z_init')

        
        # self.images_m = tf.multiply(self.mask, self.images_i)
        self.output = self.unet(self.images_i)
        # self.z_gen = tf.cond(self.get_z_init, lambda: self.generate_z(self.images_), lambda: tf.placeholder(tf.float32, [F.batch_size, 100], name='z_gen'))

        # self.G = self.generator(self.z_gen)
        self.loss = 0

        if F.vgg_loss == True:
            data_dict = loadWeightsData('./vgg16.npy')
            lambda_f = 1
            # content target feature 
            vgg_c = custom_Vgg16(self.output, data_dict=data_dict)
            # feature_i = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]
            feature_i = [ vgg_c.conv3_3, vgg_c.conv4_3]
            # feature after transformation 
            vgg = custom_Vgg16(self.images_o, data_dict=data_dict)
            # feature_o = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]
            feature_o = [vgg.conv3_3, vgg.conv4_3]
            # compute feature loss
            # self.loss = tf.zeros(F.batch_size, tf.float32)
            print(tf.shape(self.loss))
            for f, f_ in zip(feature_i, feature_o):
                # self.loss += lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])
                self.loss += lambda_f * tf.reduce_mean(tf.subtract(f_, f) ** 2)
            # self.loss += tf.reduce_mean(tf.square(vgg_net['relu3_3'][:F.batch_size] - vgg_net['relu3_3'][F.batch_size:]))# + \
                        # tf.reduce_mean(tf.square(vgg_net[:F.batch_size] - vgg_net[F.batch_size:]))
        else:
            self.loss += (tf.reduce_sum(tf.square(self.output - self.images_o)) + (0.05*tf.reduce_sum(tf.image.total_variation(self.output))))
            #self.loss = tf.reduce_sum(tf.image.total_variation(self.output))
            # self.loss += tf.reduce_sum(tf.square(self.output - self.images_o))

        tf.summary.scalar('loss', self.loss)
            
        # create summaries  for Tensorboard visualization
        # self.g_loss = tf.constant(0) 

        t_vars = tf.trainable_variables()
        # print t_vars
        self.z_vars = [var for var in t_vars if 'U/' in var.name]

        #print self.z_vars
        # self.g_vars = [var for var in t_vars if 'G/g_' in var.name]
        # self.d_vars = [var for var in t_vars if 'G/d_' in var.name]

        # self.saver_gen = tf.train.Saver(self.g_vars) # + self.d_vars)
        # try:
        self.saver = tf.train.Saver()
        # saved_path = self.saver.save(sess, "/tmp/model.ckpt")
        # print(" [*] trained weight saved!!")
        # print("Model saved in %s", saved_path)

        # except:
          # print(" [*] trained weight saving failed")

    def train_unet(self):    
        # main method for training conditonal GAN
        global_step = tf.placeholder(tf.int32, [], name="global_step_iterations")

        learning_rate_U = tf.train.exponential_decay(F.learning_rate_U, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        
        self.summary_op = tf.summary.merge_all()

       

        z_optim = tf.train.AdamOptimizer(learning_rate_U, beta1=F.beta1D)\
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

                #masks = self.next_mask()
                train_summary, _, zloss = self.sess.run(
                        [self.summary_op, z_optim,  self.loss],
                        feed_dict={global_step: counter, self.is_training: True})
                writer.add_summary(train_summary, counter)
               
                print(("Iteration: [%6d] mse loss:%.2e")
                      % (idx, zloss))
                 
                # periodically save checkpoints for future loading
                if np.mod(counter, F.saveInterval) == 1:
                    self.save(F.checkpoint_dir, counter)
                    print("Checkpoint saved successfully !!!")
                    # inp, out = self.sess.run([self.images_i, self.images_o ], feed_dict={global_step: counter,  self.is_training: False})
                    # save_images(inp, [8, 8], 'inp.png')
                    # save_images(out, [8, 8], 'out.png')
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

    def unet(self, images_i):
        dim = 32
        k = 5
        print(images_i.shape)
      

        with tf.variable_scope("U"):
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

    # def mod_dil(self, images_i):
    #   dim = 32
    #   k = 5
    #   print(images_i.shape)
    #   with tf.variable_scope("D"):
    #     h0 = lrelu(batch_norm(name='d_bn_0')(conv2d(images_i, 64, 5, 5, 1, 1, name='d_h0_conv'), self.is_training))

    #     h1 = lrelu(batch_norm(name='d_bn_1')(conv2d(h0, 128, 3, 3, 2, 2, name='d_h1_conv'), self.is_training))

    #     h2 = lrelu(batch_norm(name='d_bn_2')(conv2d(h1, 128, 3, 3, 1, 1, name='d_h2_conv'), self.is_training))

    #     h3 = lrelu(batch_norm(name='d_bn_3')(conv2d(h2, 256, 3, 3, 2, 2, name='d_h3_conv'), self.is_training))

    #     h4 = lrelu(batch_norm(name='d_bn_4')(conv2d(h3, 256, 3, 3, 1, 1, name='d_h4_conv'), self.is_training))

    #     h5 = lrelu(batch_norm(name='d_bn_5')(conv2d(h4, 256, 3, 3, 1, 1, name='d_h5_conv'), self.is_training))

    #     h6 = lrelu(batch_norm(name='d_bn_6')(conv_dil_2d(h5, 256, 3, 3, 2, name='d_h6_dil_conv'), self.is_training))

    #     h7 = lrelu(batch_norm(name='d_bn_7')(conv_dil_2d(h6, 256, 3, 3, 4, name='d_h7_dil_conv'), self.is_training))

    #     h8 = lrelu(batch_norm(name='d_bn_8')(conv_dil_2d(h7, 256, 3, 3, 8, name='d_h8_dil_conv'), self.is_training))

    #     h9 = lrelu(batch_norm(name='d_bn_9')(conv_dil_2d(h8, 256, 3, 3, 16, name='d_h9_dil_conv'), self.is_training))

    #     h10 = lrelu(batch_norm(name='d_bn_10')(conv2d(h9, 256, 3, 3, 1, 1, name='d_h10_conv'), self.is_training))

    #     h11 = lrelu(batch_norm(name='d_bn_11')(conv2d(h10, 256, 3, 3, 1, 1, name='d_h11_conv'), self.is_training))

    #     h12 = tf.nn.relu(batch_norm(name='d_bnd_12')(deconv2d(h11, [F.batch_size, 64, 64, 128], 4, 4, 2, 2, name='d_h12_deconv'), self.is_training))

    #     h13 = lrelu(batch_norm(name='d_bn_13')(conv2d(h12, 128, 3, 3, 1, 1, name='d_h13_conv'), self.is_training))

    #     h14 = tf.nn.relu(batch_norm(name='d_bnd_14')(deconv2d(h13, [F.batch_size, 128, 128, 64], 4, 4, 2, 2, name='d_h14_deconv'), self.is_training))

    #     h15 = lrelu(batch_norm(name='d_bn_15')(conv2d(h14, 32, 3, 3, 1, 1, name='d_h15_conv'), self.is_training))

    #     h16 = tf.nn.tanh(conv2d(h15, 3, 3, 3, 1, 1, name='d_h16_conv'))

    #     return h16



    def predict(self):

      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      self.sess.run(init_op)
      classifier = load_model('Track2/starting_kit_v2/models/model_baseline2_clf.h5')

      counter = 0
      if F.load_chkpt:
          try:
              self.load(F.checkpoint_dir)
              print(" [*] Checkpoint Load Success !!!")
          except:
              print(" [!] Checkpoint Load failed !!!!")
      else:
          print(" [*] Not Loaded")

      print(self.images_i.get_shape)
      print('*******************')
      vid_path = '../Inpaintin/dev/X/'
      ilist = os.listdir(vid_path)
      counter_video = 0
      for vid in range(len(ilist)):
        counter += 1
        print("Running " + str(counter) + "video ")
        f_ri_i = skvideo.io.vread(vid_path + ilist[vid])

        batch_size = F.batch_size
        o_list = []
        print('***************')
        f_ri = np.zeros((len(f_ri_i),F.output_size,F.output_size,F.c_dim)).tolist()
        for x in range(len(f_ri_i)):
          f_ri[x] = transform(f_ri_i[x], is_crop=True)
        # f_ri = f_ri.tolist()

        epoch = 0

        batch_temp = 0 # used to store the size of incompelete batch
        id = 0
        fsize = F.output_size
        for idx in range(0,len(f_ri),batch_size):
          batch = []
          if idx+batch_size<len(f_ri):
            # print(idx)
            batch = f_ri[idx:idx+batch_size]
            # F.batch_size = len(batch)
            # print(len(batch))
          else:
            batch = f_ri[idx:]
            # F.batch_size = len(batch)
            batch_temp = len(batch)
            temp = np.zeros(( (batch_size-len(batch)) , F.output_size, F.output_size, F.c_dim))
            batch = np.append(batch, temp, axis=0)

          epoch +=1
    

          t_inp, t_out = self.sess.run([self.images_i, self.output],feed_dict={self.images_i: batch, self.is_training: False})
          # save_images(t_inp, [8, 8], './in_images/'+ str(epoch) + '.png')
          t_inp = np.array(((t_inp +1.)*(127.5)), dtype=np.uint8)
          t_out = np.array(((t_out +1.)*(127.5)), dtype=np.uint8)
          
          t_inp_b = np.array(t_inp/ 255. )
          t_out_b = np.array(t_out/ 255. )

          length = t_inp_b.shape[0]
          X = t_inp_b.reshape(-1, fsize//32,32,fsize//32,32, 3).swapaxes(2,3).reshape(-1,32,32,3) 
          T = classifier.predict(X)
          idxs = np.where(T < 0.2)[0]
          Xidxs = X[idxs]

      
          T = np.reshape(T, (-1, (fsize//32)*(fsize//32), 1))
          Y = np.zeros((length, (fsize//32)*(fsize//32), 32, 32, 3))
          
          k = 0
          for i in range((fsize//32)):
            for j in range((fsize//32)):
              Y[:,k,:,:,:] = t_out_b[:, i*32:(i+1)*32, j*32:(j+1)*32, :]
              k += 1
         
       
          X = np.reshape(X, (-1, (fsize//32)*(fsize//32), 32, 32, 3))
          Xidxs = np.reshape(Xidxs, (-1, 32, 32, 3))
          
          # 1: get channels normalization ranges on non-text parts
        #   maxr = np.max(Xidxs[:,:,:,0]) 
        #   maxg = np.max(Xidxs[:,:,:,1])
        #   maxb = np.max(Xidxs[:,:,:,2])
        #   minr = np.min(Xidxs[:,:,:,0])
        #   ming = np.min(Xidxs[:,:,:,1])
        #   minb = np.min(Xidxs[:,:,:,2])

        # # 2: normalize predicted parts ..
        #   Yr = np.clip(Y[:,:,:,:,0], minr,maxr) 
        #   Yg = np.clip(Y[:,:,:,:,1], ming,maxg)
        #   Yb = np.clip(Y[:,:,:,:,2], minb,maxb)
        #   Y = np.stack((Yr,Yg,Yb), axis=-1)

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
                    # name = raw_input("name 2")
          t_out = clip*255          

          if(batch_temp != 0):
            t_inp = t_inp[:batch_temp]
            t_out = t_out[:batch_temp]
          # print(t_out[:2])
          
          for im_i, im_o in zip(t_inp, t_out):

            id += 1


            scipy.misc.imsave('./in_images/'+ str(id) + '.png', im_i)
            scipy.misc.imsave('./out_images/' + str(id) + '.png', im_o)
            o_list.append(im_o)


          # save_images(t_out, [8, 8], './out_images/' + str(epoch) + '.png')
          

        # def make_frame(t):
            # return o_list[t]

        time_i = time.time()
        skvideo.io.vwrite("./out_video/" + str(ilist[vid]).replace('X', 'Y'), o_list) 
        time_f = time.time()
        print("Elapsed time: " + str(time_f - time_i))



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



