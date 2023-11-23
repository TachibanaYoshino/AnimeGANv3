import tensorflow as tf
from tools.ops import conv_LADE_Lrelu, External_attention_v3,Conv2D


def G_net(inputs, is_training):
    with tf.variable_scope("base"):

        x0 = conv_LADE_Lrelu(inputs, 32, 7)        # 256

        x1 = conv_LADE_Lrelu(x0, 32, strides=2)    # 128
        x1 = conv_LADE_Lrelu(x1, 64)

        x2 = conv_LADE_Lrelu(x1, 64, strides=2)   # 64
        x2 = conv_LADE_Lrelu(x2, 128)

        x3 = conv_LADE_Lrelu(x2, 128, strides=2, )  # 32
        x3 = conv_LADE_Lrelu(x3, 128, )

    with tf.variable_scope("support"):
        s_x3 = External_attention_v3(x3, is_training)
        s_x4 = tf.image.resize_images(s_x3, [2 * tf.shape(s_x3)[1], 2 * tf.shape(s_x3)[2]])    # 64
        s_x4 = conv_LADE_Lrelu(s_x4, 128)
        s_x4 = conv_LADE_Lrelu(s_x4 + x2, 128)

        s_x5 = tf.image.resize_images(s_x4, [2 * tf.shape(s_x4)[1], 2 * tf.shape(s_x4)[2]])    # 128
        s_x5 = conv_LADE_Lrelu(s_x5, 64)
        s_x5 = conv_LADE_Lrelu(s_x5 + x1, 64)

        s_x6 = tf.image.resize_images(s_x5, [2 * tf.shape(s_x5)[1], 2 * tf.shape(s_x5)[2]])  # 256
        s_x6 = conv_LADE_Lrelu(s_x6, 32)
        s_x6 = conv_LADE_Lrelu(s_x6 + x0, 32)

        s_final = Conv2D(s_x6, filters=3, kernel_size=7, strides=1)
        fake_s = tf.tanh(s_final, name='out_layer')

    with tf.variable_scope("main"):
        m_x3 = External_attention_v3(x3, is_training)
        m_x4 = tf.image.resize_images(m_x3, [2 * tf.shape(m_x3)[1], 2 * tf.shape(m_x3)[2]])  # 64
        m_x4 = conv_LADE_Lrelu(m_x4, 128)
        m_x4 = conv_LADE_Lrelu(m_x4 + x2, 128)

        m_x5 = tf.image.resize_images(m_x4, [2 * tf.shape(m_x4)[1], 2 * tf.shape(m_x4)[2]])  # 128
        m_x5 = conv_LADE_Lrelu(m_x5, 64)
        m_x5 = conv_LADE_Lrelu(m_x5 + x1, 64)

        m_x6 = tf.image.resize_images(m_x5, [2 * tf.shape(m_x5)[1], 2 * tf.shape(m_x5)[2]])  # 256
        m_x6 = conv_LADE_Lrelu(m_x6, 32)
        m_x6 = conv_LADE_Lrelu(m_x6 + x0, 32)

        m_final = Conv2D(m_x6, filters=3, kernel_size=7, strides=1)
        fake_m = tf.tanh(m_final, name='out_layer')

    return fake_s, fake_m