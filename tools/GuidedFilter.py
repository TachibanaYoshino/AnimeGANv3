import tensorflow as tf
import time, os, cv2

import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def diff_x(input, r):
    assert input.shape.ndims == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=2)

    return output


def diff_y(input, r):
    assert input.shape.ndims == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = tf.concat([left, middle, right], axis=3)

    return output


def box_filter(x, r):
    assert x.shape.ndims == 4

    return diff_y(tf.cumsum(diff_x(tf.cumsum(x, axis=2), r), axis=3), r)


def guided_filter(x, y, r, eps=1e-1, nhwc=True):

    """
    @param x: guidance image with value in [0.0 , 1.0]
    @param y: filtering input image with value in [0.0 , 1.0]
    @param r: local window radius : 2, 3, 4, 5
    @param eps: regularization parameter:  0.1**2, 0.2**2, 0.4**2
    @param nhwc: tensor format (default)
    @return:  smooth image by guided filter
    """

    assert x.shape.ndims == 4 and y.shape.ndims == 4

    # data format
    if nhwc:
        x = tf.transpose(x, [0, 3, 1, 2])
        y = tf.transpose(y, [0, 3, 1, 2])

    # shape check
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)

    assets = [tf.assert_equal(x_shape[0],  y_shape[0]),
              tf.assert_equal(x_shape[2:], y_shape[2:]),
              tf.assert_greater(x_shape[2:],   2 * r + 1),
              tf.Assert(tf.logical_or(tf.equal(x_shape[1], 1),
                                      tf.equal(x_shape[1], y_shape[1])), [x_shape, y_shape])]

    with tf.control_dependencies(assets):
        x = tf.identity(x)

    # N
    N = box_filter(tf.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r)

    # mean_x
    mean_x = box_filter(x, r) / N
    # mean_y
    mean_y = box_filter(y, r) / N
    # cov_xy
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    # var_x
    var_x  = box_filter(x * x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b

    if nhwc:
        output = tf.transpose(output, [0, 2, 3, 1])

    return output


if __name__=="__main__":

    print('GuidedFilter:')
    x = tf.placeholder(tf.float32, [1, None, None, 3])
    y = tf.placeholder(tf.float32, [1, None, None, 3])

    output = guided_filter(x, y, 2, 0.005) # eps: 0.1**2, 0.2**2, 0.4**2

    temp = 'GF_res'
    check_folder(temp)
    image_foder = '../dataset/val'
    image_foder = '/media/ada/035ea81c-0b9a-4036-9c2a-a890e6fe0cee/ada/AnimeGANv3-528/samples/AnimeGANv3_6_Hayao_0.5_1.0_10.0_0.5_0/062'
    image_foder = '/media/ada/035ea81c-0b9a-4036-9c2a-a890e6fe0cee/ada/AnimeGANv3-528/samples/AnimeGANv3_4_6_Hayao_0.5_4.0_10.0_0_0/048'


    with tf.Session() as sess:
        for im in os.listdir(image_foder):
            print(im)
            # if im != '11t.jpg': # example img
            if im != '052_c.jpg': # example img
                continue
            path = os.path.join(image_foder, im)
            rgb = img_as_float(imread(path))   # 0.0 ~ 1.0
            gt = img_as_float(imread(path))

            rgb, gt = np.expand_dims(rgb,0), np.expand_dims(gt,0)


            start_time = time.time()
            r = sess.run(output,feed_dict={x:rgb, y:gt})
            end_time = time.time()
            print('\t\tTime: {}'.format(end_time - start_time))

            r = r.squeeze()
            print(r.max(),r.min())
            r = np.asarray(r.clip(0, 1) * 255, dtype=np.uint8)
            # imsave(os.path.join(temp,im), r)

            cv2.imshow('super_seg0',cv2.cvtColor((rgb.squeeze()* 255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imshow('super_seg',cv2.cvtColor(r.astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)