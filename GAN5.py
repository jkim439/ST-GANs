from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import random
from ops import *
from utils import *
from transformer import spatial_transformer_network as stn


class WGAN_GP(object):
    model_name = "WGAN_GP"  # name for checkpoint

    def __init__(self, sess, iterations, batch_size, checkpoint_dir, result_dir, log_dir, group, resume, train, gpus):
        self.sess = sess
        # Training Instance
        self.checkpoint_dir = checkpoint_dir  # Location of Checkpoint
        self.result_dir = result_dir          # Location to write images [ n x n ]
        self.log_dir = log_dir                # Location to write log for tensorboard
        self.group = group                    # Current group of training / Unique name to differentiate the training
        self.gpus = gpus                      # the number of gpus chosen eg. 1 or 4
        self.test_bs = int(batch_size / self.gpus) # batch size during testing [testing always uses 1 gpu]
        self.iter_ctr = 1                     # variable to keep track of number of iterations on a saved model
        self.resume = resume                  # 1 for trying to reload a trained model to retrain, 0 otherwise

        # Data params
        self.height = 256                     # height of image
        self.width = 256                      # width of image
        self.c_dim = 1                        # number of channels in image (Note: Further coding changes may be needed)

        # WGAN params
        self.lambd = 10                       # Gradient Penalty Multiplier
        self.disc_iters = 5                  # The number of critic iterations for one-step of generator (default 5)

        # Hyper params
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = 0.00008
        self.beta1 = 0.1
        self.beta2 = 0.99
        self.epsilon = .000001

        # Custom Loss params
        self.theta_penalty = .6              # Penalty for large changes in theta
        self.area_penalty = .01

        # Display params
        self.sample_num = 64  # number of generated images to be saved [ 8 x 8 ] for 64.
        self.merge_coeff = .5  # merge_coeff
        path = "/home/jkim/NAS/members/mkarki/STGAN/brain"
        if train:
            # load data
            self.X_test_bg = np.load("{0}/normal_image_test_256.npy".format(path)) / 255.0
            # self.X_test_bg = np.expand_dims(self.X_test_bg, axis=3)

            random.shuffle(self.X_test_bg)

            self.X_train_bg = np.load("{0}/normal_image_train_256.npy".format(path)) / 255.0
            # self.X_train_bg = np.expand_dims(self.X_train_bg, axis=3)
            self.X_train_real = np.load("{0}/hemo_1_train_256_all.npy".format(path))
            self.X_train_real = np.concatenate((self.X_train_real, np.load("{0}/hemo_2_train_256_all.npy".format(path))),
                                               axis=0)
            self.X_train_real = np.concatenate((self.X_train_real, np.load("{0}/hemo_3_train_256_all.npy".format(path))),
                                               axis=0)
            self.X_train_real = np.concatenate((self.X_train_real, np.load("{0}/hemo_4_train_256_all.npy".format(path))),
                                               axis=0)
            self.X_train_real = np.concatenate((self.X_train_real, np.load("{0}/hemo_5_train_256_all.npy".format(path))),
                                               axis=0) / 255.
            random.shuffle(self.X_train_bg)
            random.shuffle(self.X_train_real)

            self.foreground = np.load("{0}/mask_1_train_256_all.npy".format(path))
            self.foreground = np.concatenate((self.foreground, np.load("{0}/mask_2_train_256_all.npy".format(path))),
                                             axis=0)
            self.foreground = np.concatenate((self.foreground, np.load("{0}/mask_3_train_256_all.npy".format(path))),
                                             axis=0)
            self.foreground = np.concatenate((self.foreground, np.load("{0}/mask_4_train_256_all.npy".format(path))),
                                             axis=0)
            self.foreground = np.concatenate((self.foreground, np.load("{0}/mask_5_train_256_all.npy".format(path))),
                                             axis=0) / 255.
            random.shuffle(self.foreground)
            print('Data Loaded!\n')

    # Discriminator Graph
    def discriminator(self, x, bs):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            net = lrelu(conv2d(x, 32, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(conv2d(net, 32, 4, 4, 2, 2, name='d_conv21'))
            net = lrelu(conv2d(net, 32, 4, 4, 2, 2, name='d_conv22'))
            net = lrelu(conv2d(net, 64, 4, 4, 2, 2, name='d_conv23'))
            net = lrelu(conv2d(net, 64, 4, 4, 2, 2, name='d_conv24'))
            net = lrelu(conv2d(net, 64, 4, 4, 2, 2, name='d_conv25'))
            net = lrelu(conv2d(net, 64, 4, 4, 2, 2, name='d_conv26'))
            net = lrelu(conv2d(net, 128, 4, 4, 2, 2, name='d_conv27'))
            net = lrelu(conv2d(net, 128, 4, 4, 2, 2, name='d_conv28'))
            net = tf.reshape(net, [bs, -1])
            net = lrelu(linear(net, 64, scope='d_fc3'))
            out_logit = linear(net, 1, scope='d_fc4')
            out = tf.nn.tanh(out_logit)
            return out, net

    # Localization Graph , Calculates theta changes
    def localization(self, x, bs):
        with tf.variable_scope("localization", reuse=tf.AUTO_REUSE):
            net = lrelu(conv2d(x, 32, 4, 4, 2, 2, name='g_conv1'))
            net = lrelu(conv2d(net, 32, 4, 4, 2, 2, name='g_conv21'))
            net = lrelu(conv2d(net, 64, 4, 4, 2, 2, name='g_conv22'))
            net = lrelu(conv2d(net, 64, 4, 4, 2, 2, name='g_conv23'))
            net = lrelu(conv2d(net, 64, 4, 4, 2, 2, name='g_conv24'))
            net = lrelu(conv2d(net, 64, 4, 4, 2, 2, name='g_conv25'))
            net = tf.reshape(net, [bs, -1])
            net = lrelu(linear(net, 64, scope='g_fc3'))
            theta = linear(net, 4, scope='g_fc4')
            # return theta
            scale_angle = theta[:, 0] * tf.cos(theta[:, 1])
            angle = tf.sin(theta[:, 1])
            return scale_angle, angle, theta[:, 2], -angle, scale_angle, theta[:, 3]


    # calculate overlapped region  and bone overlapped region
    def overlap(self, b, f):
        back_px = tf.to_float(tf.less(b, 0.001))
        bone_px = tf.to_float(tf.greater(b, .9))
        o_r = f * (1 - back_px) * (1 - bone_px)
        b_o = (f * bone_px)
        return o_r, b_o

    def build_model(self):
        image_dims = [self.height, self.width, self.c_dim]
        fg_dims = [self.height, self.width, self.c_dim + 1]
        bs = self.batch_size


        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')  # abnormal_imgs
        self.fg = tf.placeholder(tf.float32, [bs] + fg_dims, name='foreground')  # masks
        self.bg_img = tf.placeholder(tf.float32, [bs] + image_dims, name='background')  # normal_imgs

        d_losses = 0
        g_losses = 0
        total_overlap= 0
        inputs_gpus = tf.split(self.inputs, self.gpus, axis=0)
        fg_gpus = tf.split(self.fg, self.gpus, axis=0)
        bg_img_gpus = tf.split(self.bg_img, self.gpus, axis=0)

        def default_theta(dims):
            dims = tf.convert_to_tensor([dims,], dtype=tf.int32)
            tensor1 = tf.random_normal(dims, mean=.99, stddev=.15)
            tensor2 = tf.random_normal(dims, mean=-.01, stddev=.15)
            tensor3 = tf.random_uniform(dims, minval=-.2, maxval=.2)
            # tensor4 = tf.random_normal(dims, mean=.01, stddev=.15)
            tensor5 = tf.random_normal(dims, mean=.99, stddev=.15)
            tensor6 = tf.random_uniform(dims, minval=-.2, maxval=.2)
            theta = tf.stack((tensor1, tensor2, tensor3, -tensor2, tensor5, tensor6), axis=1)
            return theta
        reuse = False
        "'Loop for multi gpu training'"
        for gpu_id in range(self.gpus):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

                    cur_bs = inputs_gpus[gpu_id].shape[0]
                    train_theta = default_theta(cur_bs)

                    # output of Discriminator for real images
                    _, D_real_logits = self.discriminator(inputs_gpus[gpu_id],bs=cur_bs)

                    img_concat = tf.concat([bg_img_gpus[gpu_id], fg_gpus[gpu_id]], axis=3)

                    # Learning Localization parameters for the background and foreground pair
                    all_theta = self.localization(img_concat, bs=cur_bs)
                    theta = tf.stack(all_theta, axis=1)
                    # Transforming the foreground based on the localiz. params
                    t_fg = stn(fg_gpus[gpu_id], theta+train_theta)
                    transformed_fg = tf.reshape(t_fg, [cur_bs, self.height, self.width, self.c_dim + 1])
                    fg_img, fg_mask = transformed_fg[:, :, :, :self.c_dim], transformed_fg[:, :, :, self.c_dim:]

                    # find the overlapping region (and region overlapping with bone) and blend them to create fake image

                    overlapped_region1, b_overlap = self.overlap(bg_img_gpus[gpu_id], fg_mask)

                    fake_img =  fg_img * fg_mask * self.merge_coeff  + (bg_img_gpus[gpu_id] * overlapped_region1)*\
                                (1-self.merge_coeff)+bg_img_gpus[gpu_id]  * (1 - overlapped_region1)

                    # output of Discriminator for fake images
                    D_fake, D_fake_logits = self.discriminator(fake_img, bs=cur_bs)

                    # get loss from discriminator
                    d_loss_real = - tf.reduce_mean(D_real_logits)
                    d_loss_fake = tf.reduce_mean(D_fake_logits)

                    d_loss_ = d_loss_real + d_loss_fake

                    """ Gradient Penalty """
                    alpha = tf.random_uniform(shape=inputs_gpus[gpu_id].get_shape(), minval=0., maxval=1.)
                    differences = fake_img - inputs_gpus[gpu_id]
                    interpolates = inputs_gpus[gpu_id] + (alpha * differences)
                    _, D_inter = self.discriminator(interpolates, bs=inputs_gpus[gpu_id].shape[0])
                    gradients = tf.gradients(D_inter, [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

                    d_loss_ += self.lambd * gradient_penalty
                    g_loss_ = - d_loss_fake

                    # exclude area on the bone
                    bone_overlap = tf.reduce_sum(b_overlap / (b_overlap + self.epsilon), reduction_indices=[1, 2])

                    """ Penalty for small/large overlapping area"""
                    overlap_area = tf.reduce_sum(
                        (overlapped_region1) / (overlapped_region1+ self.epsilon),
                        reduction_indices=[1, 2]) - bone_overlap + 0.1

                    bg_area = tf.reduce_sum(bg_img_gpus[gpu_id] / (bg_img_gpus[gpu_id] + self.epsilon),
                                            reduction_indices=[1, 2])
                    percent_bg = tf.random_normal(bg_area.shape, mean=.35, stddev=.05)
                    area_loss_ = tf.reduce_mean(tf.square(percent_bg - overlap_area/(bg_area+ self.epsilon)))
                    g_loss_ += self.area_penalty * area_loss_

                    """ Penalty for large change in theta"""
                    dT_sqnorm = tf.reduce_sum(theta ** 2 + 1e-8, reduction_indices=[1])
                    loss_dT_norm = tf.reduce_mean(dT_sqnorm)
                    g_loss_ += self.theta_penalty* loss_dT_norm

                    d_losses += d_loss_
                    g_losses += g_loss_
                    total_overlap +=overlap_area

            reuse = True

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        self.g_loss = g_losses / self.gpus
        self.d_loss = d_losses / self.gpus
        self.mean_overlap = tf.reduce_mean(total_overlap/self.gpus)
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.d_loss, var_list=d_vars, colocate_gradients_with_ops=True)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 5, beta1=self.beta1, beta2=self.beta2) \
                .minimize(self.g_loss, var_list=g_vars, colocate_gradients_with_ops=True)


        """" Testing """
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
            # brightness = tf.random_normal([self.test_bs] + image_dims, mean=1.05, stddev=.15)

            self.test_bg_img = tf.placeholder(tf.float32, [self.test_bs] + image_dims, name='bkg_test')  # normal_imgs
            self.test_fg = tf.placeholder(tf.float32, [self.test_bs] + fg_dims, name='foreground_test')  # masks

            img_concat = tf.concat([self.test_bg_img, self.test_fg], axis=3)
            all_theta2 = self.localization(img_concat, bs=self.test_bs)
            self.test_theta = tf.stack(all_theta2, axis=1) + default_theta(self.test_bs)
            transformed_fg2 = stn(self.test_fg, self.test_theta)
            fg_img, fg_mask = transformed_fg2[:, :, :, :self.c_dim], transformed_fg2[:, :, :, self.c_dim:]

            overlapped_region, b_o = self.overlap(self.test_bg_img, fg_mask)
            # brightness. Use 1 for no brightness increase
            self.fake_test_images = 1 * ((fg_img * overlapped_region) * self.merge_coeff) + \
                                    (self.test_bg_img * overlapped_region) * (1 - self.merge_coeff) + \
                                    self.test_bg_img * (1 - overlapped_region)
            self.fake_test_images = tf.clip_by_value(self.fake_test_images,0,1)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_sum = d_loss_sum
        self.g_sum = g_loss_sum

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.group + '_G_' + str(self.batch_size),
                                            self.sess.graph)

        # restore check-point if it exits
        if self.resume:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir + '_G_' + self.group)
            if could_load:
                self.iter_ctr = checkpoint_counter
                print(" [*] Previous Generator Load SUCCESS")
            else:
                print(" [*] Previous Generator Load FAIL")

        print("Hyper-parameters <Disc : %d X, Lambda: %d,  Theta Penalty: %g, Learning Rate: %g Beta1: %g Beta2: "
              "%g >" % (self.disc_iters, self.lambd, self.theta_penalty, self.learning_rate, self.beta1,
                                          self.beta2))
        # loop for epoch
        start_time = time.time()

        g_loss = 0
        overlap_ =0
        if len(self.foreground) < self.batch_size:
            mul = np.ceil(self.batch_size/len(self.foreground))
            self.foreground = np.repeat(self.foreground, mul, axis=0)
        perm = np.random.permutation(self.foreground.shape[0])  # for faster shuffle
        self.foreground = self.foreground[perm]
        perm = np.random.permutation(self.foreground.shape[0] - self.batch_size)

        def create_batch(bg, fg, bs,idx):
            rand_bg = random.randint(0, len(bg) - bs - 1)
            batch_bkg = bg[rand_bg:rand_bg + bs]  # Need to randomize
            batch_frg = fg[perm[idx% len(perm)]:perm[idx % len(perm)] + self.batch_size]
            return batch_bkg, batch_frg

        for idx in range(self.iter_ctr, self.iterations + 1):
            rand_real = random.randint(0, len(self.X_train_real) - self.batch_size - 1)
            batch_real = self.X_train_real[rand_real:rand_real + self.batch_size]  # Need to randomize

            batch_bg, batch_fg = create_batch(self.X_train_bg, self.foreground, self.batch_size,idx)
            # update D network
            _, summary_str, d_loss_ = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                    feed_dict={self.inputs: batch_real, self.fg: batch_fg,
                                                               self.bg_img: batch_bg})

            self.writer.add_summary(summary_str, idx)

            # update G network
            if (idx - 1) % self.disc_iters == 0:
                _, summary_str, g_loss,overlap_ = self.sess.run([self.g_optim, self.g_sum, self.g_loss, self.mean_overlap],
                                                               feed_dict={self.fg: batch_fg, self.bg_img: batch_bg})
                self.writer.add_summary(summary_str, idx)

            self.iter_ctr = idx
            # display training status
            if (idx % 100) == 0:
                print("Iteration: [%d]  time: %4.1f, d_loss: %.3f, g_loss: %.3f, overlap: %.3f" % (
                idx, time.time() - start_time, d_loss_, g_loss, overlap_))

            # save test results for every 300 steps
            if np.mod(idx, 1000) == 0:
                batch_bg, batch_fg = create_batch(self.X_test_bg, self.foreground, self.test_bs,idx)

                samples, perturb = self.sess.run([self.fake_test_images, self.test_theta],
                                                 feed_dict={self.test_fg: batch_fg, self.test_bg_img: batch_bg})

                tot_num_samples = min(self.sample_num, self.test_bs)
                manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                print("Theta: " + str(np.mean(perturb,axis=0)))
                save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],'./' +
                            check_folder(self.result_dir + '/' + self.model_dir + self.group) + '/' + self.model_name +
                            '_train_G_{:04d}'.format(idx))
                # save model
                self.save(self.checkpoint_dir + '_G_' + self.group, self.iter_ctr)

        print("Hyper-parameters <Disc : %d X, Lambda: %d,  Theta Penalty: %g, Learning Rate: %g Beta1: %g Beta2: "
              "%g >" % (self.disc_iters, self.lambd, self.theta_penalty, self.learning_rate, self.beta1,
                                          self.beta2))
        print(self.group)
        print(time.time())

    def inference(self, fg, bg):
        fg = fg.reshape(1, self.width, self.height, self.c_dim + 1)
        combination_img = self.sess.run([self.fake_test_images],
                                        feed_dict={self.test_fg: fg, self.test_bg_img: bg})

        return combination_img

    @property
    def model_dir(self):
        return "{}_{}".format(
            self.model_name,
            self.batch_size)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, '_G_' + '.model'), global_step=step)

    # Load to retrain
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            iter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, iter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    # Load for inference
    def load_ckpt(self, checkpoint_path):
        # initialize all variables
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())

        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_path, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
