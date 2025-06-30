import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, x*leak)


def prelu(_x, name):
    """
    Parametric ReLU.
    """
    alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.1),
                             dtype=tf.float32, trainable=True)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg