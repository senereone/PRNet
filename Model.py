import tensorflow as tf
import numpy as np
import time
import imageio
from Network import Network
from logger import Log
from utils import build_log_dir, save_param, merge_images, calc_performance
from data_helper import load_list, load_data_split, load_data_sample
from config import DatasetConfig as dataset_config
from config import ModelConfig as model_config
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class Model(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.train_test_batch_size =config.train_test_batch_size
        self.input_size = model_config.input_size
        self.sample_patch_num = config.sample_patch_num
        self.input_shape = [None] + model_config.input_shape
        self.train_test_input_shape = [config.train_test_batch_size] + model_config.input_shape

    def build_graph(self):
        self.network = Network(self.config)
        self.build_input()
        self.build_model()

    def build_input(self):
        device = '/gpu:%s' % model_config.gpu
        with tf.device(device):
            self.dis_img = tf.placeholder(tf.float32, self.input_shape, name="dis_img")
            self.ref_img = tf.placeholder(tf.float32, self.input_shape, name="ref_img")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")


    def build_model(self):
        self.fake_imgs = self.network.darb(self.dis_img, self.is_training, reuse=False)
        recovery_img = self.fake_imgs[0] - self.fake_imgs[0]
        self.img_weight = tf.random_normal([self.config.net_repet_num, 1])
        self.alpha = tf.random_normal([1])
        self.beta = tf.random_normal([1])

        for i in range(self.config.net_repeat_num):
            recovery_img += self.fake_imgs[i] * self.img_weight[i]
        recovery_image = recovery_img / (self.config.net_repeat_num + 0.0)
        self.fake_img = recovery_image
        print(self.fake_img.shape)

        ref_img = tf.image.convert_image_dtype(self.ref_img, tf.float32)
        fake_img = tf.image.convert_image_dtype(self.fake_img, tf.float32)
        mse_loss1 = (self.config.net_repeat_num) * [None]
        ssim_loss1 = (self.config.net_repeat_num) * [None]
        for i in range(0, self.config.net_repeat_num+1):
            generate_img = tf.image.convert_image_dtype(self.fake_imgs[i], tf.float32)
            mse_loss1[i] = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(self.fake_imgs[i], self.ref_img)) + 1e-6))
            ssim_loss1[i] = -tf.reduce_mean(tf.image.ssim(generate_img, ref_img, max_val=2))
        mse_loss1 = tf.add_n(mse_loss1) * (1.0 / self.config.net_repeat_num)
        ssim_loss1 = tf.add_n(ssim_loss1) * (1.0 / self.config.net_repeat_num)
        mse_loss2 = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(self.fake_img, self.ref_img)) + 1e-6))
        ssim_loss2 = -tf.reduce_mean(tf.image.ssim(fake_img, ref_img, max_val=2))

        self.loss = self.beta * (self.alpha * mse_loss1 + (1-self.alpha) * mse_loss2) + (1-self.beta) * \
                    (self.alpha * ssim_loss1 + (1 - self.alpha) * ssim_loss2)

        self.t_vars = tf.trainable_variables()

    def build_log(self):
        self.log = Log()
        logs = build_log_dir()
        self.param_log = logs[3]
        self.train_log, self.train_test_log = logs[4], logs[5]
        self.model_dir, self.pic_dir = logs[6], logs[7]
        self.test_psnr_ssim_log = logs[8]
        self.all_train_test_psnr_ssim_log = logs[11]

        save_param(self.config, self.param_log)

        line = self.log.line(["time", "global_step", "epoch", "step", "loss"])
        self.log.save_line(line, self.train_log)
        line = self.log.line(["time", "epoch", "loss"])
        self.log.save_line(line, self.train_test_log)
        line = self.log.line(["epoch", "test_step", "PSNR", "SSIM"])
        self.log.save_line(line, self.test_psnr_ssim_log)
        line = self.log.line(["epoch", "PSNR", "SSIM"])
        self.log.save_line(line, self.all_train_test_psnr_ssim_log)

    def train_test(self, sess, epoch, test_ref_data, test_dis_data, train_test_step):
        total_loss = 0.0
        psnr_values_dis_all, ssim_values_dis_all, psnr_values_fake_all, ssim_values_fake_all = [], [], [], []

        steps = len(test_ref_data) // self.train_test_batch_size
        steps = min(steps, train_test_step)
        for step in range(steps):
            batch_test_ref = test_ref_data[step * self.train_test_batch_size: (step + 1) * self.train_test_batch_size]
            batch_test_dis = test_dis_data[step * self.train_test_batch_size: (step + 1) * self.train_test_batch_size]

            batch_test_fake, loss = self.sess.run([self.fake_img, self.loss],
                                                  feed_dict={self.ref_img: batch_test_ref, self.dis_img: batch_test_dis,
                                                             self.is_training: True})
            total_loss += loss

            # save the generated fake image
            manifold_h = int(np.floor(np.sqrt(self.train_test_batch_size)))
            manifold_w = int(np.floor(np.sqrt(self.train_test_batch_size)))


            fake_img = merge_images(batch_test_fake, [manifold_h, manifold_w])
            fake_img = (fake_img + 1) * 127.5
            imageio.imsave(self.log.check_dir(self.pic_dir + "/fake/") + "fake_{:02d}_{:02d}.png".format(epoch, step),
                           np.array(fake_img.astype(np.uint8())))

            ref_img = merge_images(batch_test_ref, [manifold_h, manifold_w])
            ref_img = (ref_img + 1) * 127.5


            dis_img = merge_images(batch_test_dis, [manifold_h, manifold_w])
            dis_img = (dis_img + 1) * 127.5

            # save ref image and dis image
            if epoch == 0:
                imageio.imsave(self.log.check_dir(self.pic_dir + "/ref/") + "ref_{:02d}_{:02d}.png".format(epoch, step),
                               np.array(ref_img.astype(np.uint8())))
                imageio.imsave(self.log.check_dir(self.pic_dir + "/dis/") + "dis_{:02d}_{:02d}.png".format(epoch, step),
                               np.array(dis_img.astype(np.uint8())))

            fake_psnr_values = peak_signal_noise_ratio(ref_img, fake_img, data_range=255)
            psnr_values_fake_all.append(fake_psnr_values)
            fake_ssim_values = structural_similarity(ref_img, fake_img, data_range=255)
            ssim_values_fake_all.append(fake_ssim_values)
            dis_psnr_values = peak_signal_noise_ratio(ref_img, dis_img, data_range=255)
            psnr_values_dis_all.append(dis_psnr_values)
            dis_ssim_values = structural_similarity(ref_img, dis_img, data_range=255)
            ssim_values_dis_all.append(dis_ssim_values)

            line = self.log.line([epoch, step, dis_psnr_values, dis_ssim_values, fake_psnr_values,fake_ssim_values])
            self.log.save_line(line, self.test_psnr_ssim_log)

        avg_losses = total_loss / steps
        psnr_dis = np.mean(psnr_values_dis_all)
        ssim_dis = np.mean(ssim_values_dis_all)
        psnr_fake = np.mean(psnr_values_fake_all)
        ssim_fake = np.mean(ssim_values_fake_all)

        line = self.log.line([epoch, psnr_dis, ssim_dis, psnr_fake, ssim_fake])
        self.log.save_line(line, self.all_train_test_psnr_ssim_log)

        m_time = time.strftime("%Y-%m-%d %X", time.localtime())
        line = self.log.line([m_time, epoch, avg_losses])
        self.log.save_line(line, self.train_test_log)


    def train(self, sess):
        self.build_log()
        self.sess = sess

        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step,
                                                   self.config.lr_decay_steps,
                                                   self.config.lr_decay_rate, staircase=True)
        r_optim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(self.loss, var_list=self.t_vars,
                                                                            global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(self.t_vars, max_to_keep=self.config.max_to_keep)
        if self.config.pretrained_ckpt is not None:
            print("restoring checkpoint from %s ..." % self.config.pretrained_ckpt)
            saver.restore(sess, self.config.pretrained_ckpt)

        # test data
        test_ref_list, test_dis_list = load_list(dataset_config.test_meta)
        test_ref_data, test_dis_data = load_data_split(test_ref_list, test_dis_list,
                                                       split_shape=[self.input_size, self.input_size],
                                                       split_stride=self.input_size, is_gray=dataset_config.is_gray,
                                                       shuffle=False)

        # train data
        train_ref_list, train_dis_list = load_list(dataset_config.train_meta)

        for epoch in range(self.config.epochs):
            train_ref_data, train_dis_data = load_data_sample(train_ref_list, train_dis_list,
                                                              sample_shape=[self.input_size, self.input_size],
                                                              sample_num=self.sample_patch_num,
                                                              is_gray=dataset_config.is_gray,
                                                              shuffle=True)
            steps = len(train_ref_data) // self.batch_size
            for step in range(steps):
                batch_train_ref = train_ref_data[step * self.batch_size: (step + 1) * self.batch_size]
                batch_train_dis = train_dis_data[step * self.batch_size: (step + 1) * self.batch_size]

                m_time = time.strftime("%Y-%m-%d %X", time.localtime())

                _, batch_train_fake, loss = self.sess.run([r_optim, self.fake_img, self.loss],
                                                          feed_dict={self.ref_img: batch_train_ref,
                                                                     self.dis_img: batch_train_dis,
                                                                     self.is_training: True})

                self.m_global_step = self.sess.run(global_step)

                line = self.log.line([m_time, self.m_global_step, epoch, step, loss])
                self.log.save_line(line, self.train_log)

                # do train_test
            if epoch % self.config.train_test_per_epoch == 0:
                self.train_test(sess, epoch, test_ref_data, test_dis_data, self.config.train_test_step)

            # save model
            if epoch % self.config.snapshot_epoch == 0:
                saver.save(self.sess, self.log.check_dir(self.model_dir) + "model.ckpt", epoch)



