# import tensorflow as tf
# temp=tf.ones((256,32,32,64))
# a=tf.reduce_max(temp, axis=-1)
# print(a.get_shape())
# a=tf.reshape(a,[-1,32*32])
# print(a.get_shape())
# a=tf.nn.softmax(a,axis=-1)
# print(a.get_shape())
# a=tf.reshape(a,[-1,32,32,1])
# print(a.get_shape())
#
# position_attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.reduce_max(temp, axis=-1),
#                                                          [-1, 32*32]), axis=-1),
#                                 [-1, 32,32, 64])
#
# channel_attention = tf.reshape(tf.nn.softmax(tf.reduce_max(temp, axis=(1, 2)), axis=-1), [-1, 1, 1, 64])
#
# print(position_attention.get_shape())
# print(channel_attention.get_shape())

import tensorflow as tf
import numpy as np

input1 = tf.constant([1.0, 2.0, 3.0])
input2 = tf.Variable(tf.random_uniform([3]))
output = tf.add_n([input1, input2])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(input2))
    print(sess.run(input1 + input2))
    print(sess.run(output))
