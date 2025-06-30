from ops import *
from config import ModelConfig as model_config


class Network(object):
    def __init__(self, config):
        self.config = config
        self.momentum = 0.9
        self.output_c_dim = model_config.output_c_dim


    def darb(self, image, is_training, reuse=False):
        device = '/gpu:%s' % model_config.gpu
        with tf.device(device):
            with tf.variable_scope("darb", reuse=reuse) as scope:
                input_feature = tf.layers.conv2d(image, kernel_size=3, filters=64, strides=1, padding="SAME")
                input_image = image

                intermediate_images = self.config.net_repeat_num * [None]
                for i in range(self.config.net_repeat_num):
                    net_image = tf.layers.conv2d(input_image, kernel_size=3, filters=64, strides=1, padding="SAME")
                    net_image = lrelu(net_image)
                    net_feature = tf.layers.conv2d(input_feature, kernel_size=3, filters=64, strides=1, padding="SAME")
                    net_feature = lrelu(net_feature)
                    net_temp = net_feature

                    for d in range(10):
                        net_temp1 = tf.layers.conv2d(net_temp, kernel_size=3, filters=64, strides=1, padding="SAME")
                        net_temp1 = lrelu(net_temp1)
                        net_temp = tf.concat([net_temp1, net_temp], axis=3)
                    net_temp = tf.layers.conv2d(net_temp, kernel_size=1, filters=64, strides=1, padding="SAME")
                    net_feature = tf.add(net_feature, net_temp)

                    net_image = tf.layers.conv2d(net_image, kernel_size=3, filters=64, strides=1, padding="SAME")
                    net_image = lrelu(net_image)

                    position_attention = tf.reshape(tf.reshape(tf.reduce_max(net_image, axis=-1),
                                                                             [-1, net_image.get_shape()[1] *
                                                                              net_image.get_shape()[2]]),
                                                    [-1, net_image.get_shape()[1], net_image.get_shape()[2], 1])

                    channel_attention = tf.reshape(tf.reduce_max(net_image, axis=(1, 2)),
                                                   [-1, 1, 1, 64])
                    attention_out = net_feature * position_attention + net_feature * channel_attention
                    net_image = tf.layers.conv2d(net_image, kernel_size=3, filters=self.output_c_dim, strides=1, padding="SAME")
                    net_gradient = tf.layers.conv2d(attention_out, kernel_size=3, filters=self.output_c_dim, strides=1, padding="SAME")

                    net_image = tf.add(net_gradient, net_image)
                    intermediate_images[i] = net_image
                return intermediate_images

