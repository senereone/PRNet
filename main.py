import sys
import tensorflow as tf
import numpy as np
from Model import Model
from config import TrainConfig as train_config
from config import TestConfig as test_config
from config import ModelConfig as model_config
from config import DatasetConfig as dataset_config
import imageio
from logger import Log
from utils import build_log_dir, merge_images
from data_helper import load_list, load_data_split, load_data
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


os.environ["CUDA_VISIBLE_DEVICES"] = train_config.gpu

def parse_args():
    if len(sys.argv) <= 1:
        print('usage: python %s <train or test>' % sys.argv[0])
        exit()
    mode = sys.argv[1]
   # if len(sys.argv) == 3:
   #     os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    return mode

def restore_model(sess):
    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, test_config.test_model)

def train():
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        model = Model(train_config)
        model.build_graph()
        model.train(sess)

def test():
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        model = Model(test_config)
        model.build_graph()
        restore_model(sess)

        log = Log()
        logs = build_log_dir()
        all_test_pic_dir = logs[9]
        all_test_psnr_ssim_log = logs[10]
        line = log.line(["dis_PSNR", "dis_SSIM", "fake_PSNR", "fake_SSIM"])
        log.save_line(line, all_test_psnr_ssim_log)

        test_ref_list, test_dis_list = load_list(test_config.test_meta)
        test_ref_data, test_dis_data = load_data(test_ref_list, test_dis_list,
                                                       split_shape=[model_config.input_size,
                                                                    model_config.input_size],
                                                       split_stride=model_config.input_size,
                                                       is_gray=dataset_config.is_gray,
                                                       shuffle=False)

        steps = len(test_ref_data) // test_config.batch_size

        psnr_values_dis_all, ssim_values_dis_all, psnr_values_fake_all, ssim_values_fake_all = [], [], [], []
        for step in range(steps):
            batch_test_ref = test_ref_data[step * test_config.batch_size: (step + 1) * test_config.batch_size]
            batch_test_dis = test_dis_data[step * test_config.batch_size: (step + 1) * test_config.batch_size]

            batch_test_fake = sess.run(model.fake_img, feed_dict={model.ref_img: batch_test_ref,
                                                                  model.dis_img: batch_test_dis,
                                                                  model.is_training: True
                                                                  })
            # save the generated fake image
    
            fake_img = (batch_test_fake + 1) * 127.5

            imageio.imsave(log.check_dir(all_test_pic_dir) + "fake_{:02d}.png".format(step),
                           np.array(fake_img[0].astype(np.uint8())))

            ref_img = (batch_test_ref + 1) * 127.5
            dis_img = (batch_test_dis + 1) * 127.5
            fake_img = np.reshape(fake_img, [fake_img.shape[1]] + [fake_img.shape[2]])
            ref_img = np.reshape(ref_img, [ref_img.shape[1]] + [ref_img.shape[2]])
            dis_img = np.reshape(dis_img, [dis_img.shape[1]] + [dis_img.shape[2]])

            psnr_values_dis = peak_signal_noise_ratio(ref_img, dis_img, data_range=255)
            psnr_values_dis_all.append(psnr_values_dis)
            ssim_values_dis = structural_similarity(ref_img, dis_img, data_range=255)
            ssim_values_dis_all.append(ssim_values_dis)
            psnr_values_fake = peak_signal_noise_ratio(ref_img, fake_img, data_range=255)
            psnr_values_fake_all.append(psnr_values_fake)
            ssim_values_fake = structural_similarity(ref_img, fake_img, data_range=255)
            ssim_values_fake_all.append(ssim_values_fake)

            line = log.line([psnr_values_dis, ssim_values_dis, psnr_values_fake, ssim_values_fake])
            log.save_line(line, all_test_psnr_ssim_log)

        psnr_dis = np.mean(psnr_values_dis_all)
        ssim_dis = np.mean(ssim_values_dis_all)
        psnr_fake = np.mean(psnr_values_fake_all)
        ssim_fake = np.mean(ssim_values_fake_all)
        line = log.line([psnr_dis, ssim_dis, psnr_fake, ssim_fake])
        log.save_line(line, all_test_psnr_ssim_log)

def main():
    mode = parse_args()
    if mode == "train":
        train()
    elif mode == "test":
        test()
    # train()
    # else:
    #     print("mode error")

if __name__ == "__main__":
    main()
