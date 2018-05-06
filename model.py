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
from moviepy.editor import *
import skvideo.io
import matplotlib.pyplot as plt

from VGG_loss import *
# from Track2/starting_kit_v2/data_manager import *
# import subprocess as sp
import pims


data_dict = loadWeightsData('./vgg16.npy')


F = tf.app.flags.FLAGS

class DCGAN(object):
    def __init__(self, sess):
        self.sess = sess
        self.ngf = 128
        self.ndf = 64
        self.nt = 128
        self.k_dim = 16
        self.image_shape = [F.output_size, F.output_size, 3]
        
        self.h = F.output_size
        self.w = F.output_size
        self.code_len = 3  # don't know the meaning so just giving an arbit value 
        # can be changed/tuned

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

            self.images_i = (self.images_i / 127.5) - 1
            self.images_o = (self.images_o / 127.5) - 1

        else:    
            self.images_i = tf.placeholder(tf.float32,
                                       [F.batch_size, F.output_size, F.output_size,
                                        F.c_dim],
                                       name='real_images_i')
            self.images_o = tf.placeholder(tf.float32,
                                        [F.batch_size, F.output_size, F.output_size,
                                            F.c_dim],
                                       name='real_images_o')
        
        #self.mask = tf.placeholder(tf.float32, [F.batch_size, F.output_size, F.output_size, 3], name='mask')
        #here the function of mask is to be added so that mask is extracted from image
        # this mask needs to be the subtitles in images.
        # self.mask = self.images_i
        # just for running code I am using mask same as input image.
        self.is_training = tf.placeholder(tf.bool, name='is_training')        
        # self.get_z_init = tf.placeholder(tf.bool, name='get_z_init')

        
        # self.images_m = tf.multiply(self.mask, self.images_i)
        self.output = self.unet(self.images_i)
        # self.z_gen = tf.cond(self.get_z_init, lambda: self.generate_z(self.images_), lambda: tf.placeholder(tf.float32, [F.batch_size, 100], name='z_gen'))

        # self.G = self.generator(self.z_gen)
        self.loss = 0

        if F.vgg_loss == True:
            lambda_f = 1
            # content target feature 
            vgg_c = custom_Vgg16(self.images_i, data_dict=data_dict)
            # feature_i = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]
            feature_i = [ vgg_c.conv3_3, vgg_c.conv4_3]
            # feature after transformation 
            vgg = custom_Vgg16(self.output, data_dict=data_dict)
            # feature_o = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]
            feature_o = [vgg.conv3_3, vgg.conv4_3]
            # compute feature loss
            # self.loss = tf.zeros(F.batch_size, tf.float32)
            print(tf.shape(self.loss))
            for f, f_ in zip(feature_i, feature_o):
                # self.loss += lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])
                self.loss += lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2)
            # self.loss += tf.reduce_mean(tf.square(vgg_net['relu3_3'][:F.batch_size] - vgg_net['relu3_3'][F.batch_size:]))# + \
                        # tf.reduce_mean(tf.square(vgg_net[:F.batch_size] - vgg_net[F.batch_size:]))
        else:
            self.loss += tf.reduce_sum(tf.square(self.output - self.images_o))

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
                    inp, out = self.sess.run([self.images_i, self.images_o ], feed_dict={global_step: counter,  self.is_training: False})
                    save_images(inp, [8, 8], 'inp.png')
                    save_images(out, [8, 8], 'out.png')
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
              h1 = lrelu(batch_norm(name='u_bn1')(conv2d(h0, dim * 2, name='u_h1_conv'), self.is_training))
              h2 = lrelu(batch_norm(name='u_bn2')(conv2d(h1, dim * 4, name='u_h2_conv'), self.is_training))
              h3 = lrelu(batch_norm(name='u_bn3')(conv2d(h2, dim * 8, name='u_h3_conv'), self.is_training))
             


              h4 = lrelu(batch_norm(name='u_bn4')(conv2d(h3, dim * 16, name='u_h4_conv'), self.is_training))
              # h5 = tf.reshape(h4, [F.batch_size, -1])
              # h6 = linear(h5, 100, 'u_h5_lin')
              # g0 = linear(h6, s16 * s16 * dim * 16, 'g_lin')
              # g0 = tf.reshape(g0, [F.batch_size, s16, s16, dim * 16])
              # up1 = tf.concat([g0, h4], 3)


              g1 = deconv2d(h4, [F.batch_size, s8, s8, dim * 8], k, k, 2, 2, name = 'g_deconv1')
              g1 = tf.nn.relu(batch_norm(name = 'g_bn1')(g1, self.is_training))
              up1 = tf.concat([g1, h3], 3)

              g2 = deconv2d(up1, [F.batch_size, s4, s4, dim * 4], k, k, 2, 2, name = 'g_deconv2')
              g2 = tf.nn.relu(batch_norm(name = 'g_bn2')(g2, self.is_training))
              up2 = tf.concat([g2, h2], 3)


              g3 = deconv2d(up2, [F.batch_size, s2, s2, dim * 2], k, k, 2, 2, name = 'g_deconv4')
              g3 = tf.nn.relu(batch_norm(name = 'g_bn3')(g3, self.is_training))
              up3 = tf.concat([g3, h1], 3)

              g4 = deconv2d(up3, [F.batch_size, F.output_size, F.output_size, 3], k, k, 2, 2, name ='g_hdeconv5')
              g4 = tf.nn.tanh(g4, name = 'g_tanh')
              return g4

    def predict(self):

      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      self.sess.run(init_op)

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
        f_ri = skvideo.io.vread(vid_path + ilist[vid])
        # video_i = VideoFileClip(vid_path + ilist[vid])
        # f_ri = [x for x in video_i.iter_frames()]
        # f_ri = self.getAllFrames(vid_path + ilist[vid])
        # f_ri = f_ri.tolist() # optimisation has to be done list to numpy
        o_list = []


        epoch = 0
        for img in f_ri:
         
          # print("Running " + str(epoch) + 'epoch ')
          epoch +=1
          batch = []
        
          img = [transform(img, is_crop=True)]
          # batch.append(img)


          # if examples >= F.batch_size:
          #     batch = np.asarray(batch)[:, :, :, [2, 1, 0]]
          #     yield batch
          #     batch = []
          #     examples = 0

          t_inp, t_out = self.sess.run([self.images_i, self.output],feed_dict={self.images_i: img, self.is_training: False})
          # save_images(t_inp, [8, 8], './in_images/'+ str(epoch) + '.png')
          t_inp = np.array(((t_inp +1)*127.5), dtype=np.uint8)
          t_out = np.array(((t_out +1)*127.5), dtype=np.uint8)
          scipy.misc.imsave('./in_images/'+ str(epoch) + '.png', t_inp[0])
          scipy.misc.imsave('./out_images/' + str(epoch) + '.png', t_out[0])

          # save_images(t_out, [8, 8], './out_images/' + str(epoch) + '.png')
          o_list.append(t_out[0])

        # def make_frame(t):
            # return o_list[t]

        time_i = time.time()
        skvideo.io.vwrite("./out_video/" + str(ilist[vid]).replace('X', 'Y'), o_list) 
        time_f = time.time()
        print("Elapsed time: " + str(time_f - time_i))
          #a method to create video ..too slow
        # self.createVideoClip(o_list, "./out_video", str(ilist[vid]).replace('X', 'Y'))
        # print(video_i.fps)
        # print(video_i.duration)
        # clips = [ImageClip(m).set_duration(video_i.duration/len(o_list)) for m in o_list]
        # concat_clips = concatenate_videoclips(clips, method="compose")
        # concat_clips.write_videofile("./out_video/" + str(ilist[vid])+ '.mp4', fps=video_i.fps)
          # cv2.imwrite('./out_images/'+str(epoch)+'.png', t_out)
          # cv2.imwrite('./in_images/'+str(epoch)+'.png', t_inp)
    def createVideoClip(self, clip, folder, name):
      clip = np.array(clip, dtype='float32')
      print (clip.shape)
      # clip = (clip + 1.) * 127.5.
      # clip = clip.astype('uint8')

      # write video stream #
      command = [ 'ffmpeg',
      '-y',  # overwrite output file if it exists
      '-f', 'rawvideo',
      '-s', '128x128', #'256x256', # size of one frame
      '-pix_fmt', 'rgb24',
      '-r', '25', # frames per second
      '-an',  # Tells FFMPEG not to expect any audio
      '-i', '-',  # The input comes from a pipe
      '-vcodec', 'libx264',
      '-b:v', '100k',
      '-vframes', '125', # 5*25
      '-s', '128x128', #'256x256', # size of one frame
      folder+'/'+name+'.mp4' ]

      pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
      # out, err = pipe.communicate(clip.tostring())
      # try:
      outs, errs = pipe.communicate(clip.tostring())
      # except sp.TimeoutExpired:
      #   pipe.kill()
      # outs, errs = pipe.communicate()
      print ('bjsd')
      pipe.wait()
      print ('bkagf2')
      pipe.terminate()
      print ('3')
        #print(err)


    def getAllFrames(self, clipname):
      print(clipname)

      # open one video clip sample
      try:
          data = pims.Video(root_dataset+'/'+clipname)
      except:
          data = pims.Video(clipname)

      data = np.array(data, dtype='float32')
      length = data.shape[0]

      return data[:125]
          
    


    # def predict_video(self):

    #   image_dir = './out_vid_im/'
    #   ilist = os.listdir(image_dir)
    #   sort_l = []
    #   for infile in sorted(ilist):
    #     print "Current File Being Processed is: " + infile
    #     sort_l.append(infile)
    #   print(len(sort_l))
    #   print(len(ilist))
    #   clips = [ImageClip(image_dir + m).set_duration(2)
    #           for m in sort_l]

    #   concat_clip = concatenate_videoclips(clips, method="compose")
    #   concat_clip.write_videofile("test.mp4", fps=25)


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


 #    def yield_batch(self):

 #      im_dir = './out_vid_im/'      
      
 #      self.data = os.listdir(im_dir)
 #      # print(self.data)
 #      self.num_batches =  len(self.data)// F.batch_size
 #      # print(len(self.data))

 #      # img2 = cv2.imread('./test_images/index.jpeg', 1)
 #      # print(img2)
 #      # img1 = cv2.imdecode(img1, 1)
 #      # print(img1)
 #      for img in self.data:
 #          # cursor = txn.cursor()

 #          examples = 0
 #          batch = []
 #          # print(img.shape)
 #          # img = np.fromstring(img, dtype=np.uint8)
 #          # print(img.shape)
 #          # img = cv2.imdecode(img, 1)
 #          # print(img.shape)
 #          img_path = str(im_dir) + str(img)
 #          img = cv2.imread(img_path,1)
 #          img = transform(img, is_crop=True)
 #          batch.append(img)
 #          examples += 1

 #          if examples >= F.batch_size:
 #              batch = np.asarray(batch)[:, :, :, [2, 1, 0]]
 #              yield batch
 #              batch = []
 #              examples = 0
 # 