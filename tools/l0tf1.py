################################################################################
# L0 Gradient Norm Image Smoothing Algorithm
# Implemented in: Tensorflow 2.0
#
# Author: Jameson Nguyen (JNRuan)
################################################################################
# ATTRIBUTIONS
#
# [1]. L. Xu, C. Lu, Y. Xu, and J. Jia, "Image Smoothing via L0 Gradient Minimization",
#      ACM Transactions on Graphics, Vol. 30, No. 5 (SIGGRAPH Asia 2011), Dec 2011
#
# [2]. Alexandre Boucaud, “pypher: Python PSF Homogenization kERnels”. Zenodo, 02-Sep-2016.
#
################################################################################

# Disable TF logging for INFO and WARN, keep ERROR.
import os
import cv2,time
import numpy as np
import tensorflow as tf

def zero_pad_fxypsf(psf, shape):
    """
    Pads point spread function (psf) Fx or Fy with zeroes up to target shape.
    The target shape is the shape of the image we want to smooth.
    Keeps original psf functions in the initial positions of the tensor and pads
    the rest of the tensor indices with zeroes.
    This method is a pre-processing step prior to conversion to an optical
    transfer function (OTF).
    Expects one of two psf functions:
    Fx = [[1, -1]]
    Fy = [[1]. [-1]]
    :param psf: Tensor containing a point spread function for padding.
    :param shape tuple(int, int): Target shape from image we are smoothing.
    :return: psf function padded with zeroes with new shape=(shape[0], shape[1])
    """
    if psf.shape[0] == 1:
        # PSF is Fx = [[-1, 1]]
        indices = [[0, 0], [0, 1]]
        psf_padded = tf.SparseTensor(indices, psf[0], shape)
        psf_padded = tf.sparse.to_dense(psf_padded, default_value=0)
    elif psf.shape[0] == 2:
        # PSF is Fy = [[-1], [1]]
        indices = [[0, 0], [1, 0]]
        psf_padded = tf.SparseTensor(indices, psf[:, 0], shape)
        psf_padded = tf.sparse.to_dense(psf_padded, default_value=0)
    return psf_padded


def _fxypsf_to_otf(psf, target):
    """
    _fxypsf_to_otf is an adapted function specifically for the L0 norm algorithm
    which originally made use of a matlab psf2otf function. This function is
    therefore a port to tensorflow based off the matlab psf2otf function via a
    python port [2], but specifically adapted for the image smoothing algorithm.
    Converts point spread function (psf) to optical transfer function (otf). Using
    the fast fourier transform on a padded psf.
    Expects to work with psf of either:
    Fx = [[1, -1]], or
    Fy = [[1], [-1]]
    ============================================================================
    Attribution:
    Original function was a numpy port of a matlab psf2otf function. For a general
    use psf2otf function, recommend using the original numpy port.
    Original python implementation: https://github.com/aboucaud/pypher [2]
    :param psf: Tensor containing a point spread function for padding.
    :param target: Target img for smoothing to compute otf up to target dimensions.
    :return: otf of original provided psf functions Fx or Fy.
    """
    target_shape = (target.shape[0], target.shape[1])

    psf_padded = zero_pad_fxypsf(psf, target_shape)

    # Per matlab implementation, to ensure off-center psf does not later otf,
    # circular shift psf until central pixel is in (0, 0) position.
    for axis, axis_sz in enumerate(psf.shape):
        psf_padded = tf.roll(psf_padded, shift=(-(axis_sz.value) // 2), axis=axis)

    # Calculate otf
    # Cast psf to complex number per tensorflow spec for 2d fast fourier transform.
    psf_padded = tf.cast(psf_padded, dtype=tf.complex64)
    otf = tf.signal.fft2d(psf_padded)
    return otf


def l0_image_smoother(img, _lambda=2e-2, kappa=2.0, beta_max=1e5, ):
    """
    Applies L0 Image Smoothing [1] on target img.
    Lambda is a hyperparameter to tune degree of smoothing.
    By default this is 2e-2, authors recommend a range of [1e-3, 1e-1] [1].
    Usage note: Smaller lambda results in retaining of more of the original details of image.
    Kappa is the scaling factor that scales rate of smoothing,
    smaller kappa scalar results in more iterations and sharper edges.
    Authors recommend range (1, 2].
    Iterations of smoothing based on beta < beta_max.
    With beta initialised as 2 * lambda.
    In addition, beta is incremented at rate beta * kappa each iteration.
    Furthermore, beta is used as a scaling factor for weight per iteration.
    ============================================================================
    Attribution:
    Original code written in matlab kindly provided by authors of algorithm [1].
    Matlab code can be found at authors website: http://www.cse.cuhk.edu.hk/~leojia/projects/L0smoothing/
    :param img: Input image, read in as numpy array.
    :param _lambda: Smoothing parameter for degree of smoothness [1]. Default 2e-2.
    :param kappa: Scale rate of smoothing. Default 2.0
    :param beta_max: Parameter to scale max iterations, each iteration increments beta * kappa.
    :return: Numpy array representing the smoothed image.
    """
    # Store image dimensions for convenience, C is the number of channels
    N, M, C = img.shape
    # Initialise S as float32 of image
    S = img / 255
    # Image needs to be complex64 or complex128 for tensorflows fourier transform.
    psf_fx = tf.constant([[1, -1]], dtype=tf.int8)
    psf_fy = tf.constant([[1], [-1]], dtype=tf.int8)
    otf_fx = _fxypsf_to_otf(psf_fx, S)
    otf_fy = _fxypsf_to_otf(psf_fy, S)

    img_tensor = tf.cast(S, dtype=tf.complex64)
    # Pre-compute numerator and denominator for sub-problems per matlab
    # implementation [1], used for the two subproblems (h-v, S) [1].
    normin1 = tf.signal.fft3d(img_tensor)
    denorm2 = tf.abs(otf_fx) ** 2 + tf.abs(otf_fy) ** 2

    # Convert denominator to 3 channels for colour:
    if C > 1:
        denorm2 = tf.tile(tf.expand_dims(denorm2, 2), [1, 1, C])

    # Initial beta, smooth until beta > beta_max
    beta = 2 * _lambda

    while beta < beta_max:
        denorm = 1 + beta * denorm2

        # H-V SUBPROBLEM per [1]
        # To do this we need to build tensors to add to the base h tensor of zeroes,
        # as the matlab operation in code provided by authors [1] is not possible.
        # Matlab: h = [diff(S,1,2), S(:,1,:) - S(:,end,:)]; where S is the image.

        # Compute h
        h = tf.zeros_like(S)
        # First M-1 columns (axis 1)
        first_hdiff = S[:, 1:] - S[:, :-1]
        # Last Mth column (axis 1)
        last_hdiff = S[:, 0:1, :] - S[:, M - 1:M, :]
        h_diff = tf.concat([
            first_hdiff,
            last_hdiff
        ], axis=1)
        h += h_diff

        # Compute v
        # Matlab: v = [diff(S,1,1); S(1,:,:) - S(end,:,:)]; where S is the image.
        v = tf.zeros_like(S)
        # First N-1 columns (axis 0)
        first_vdiff = S[1:] - S[:-1]
        # Last Nth column (axis 0)
        last_vdiff = S[0:1, :, :] - S[N - 1:N, :, :]
        v_diff = tf.concat([
            first_vdiff,
            last_vdiff
        ], axis=0)
        v += v_diff

        # Reduce sum on channel dimension
        hv = tf.math.pow(h, 2) + tf.math.pow(v, 2)
        t = tf.math.reduce_sum(hv, axis=2)
        # print(t.shape, )
        threshold = _lambda / beta

        # Find indices where t < threshold

        # indices_true = tf.where(t < threshold).eval(session=sess)
        indices_true =t < threshold
        # t_masked = tf.stack([indices_true, indices_true, indices_true], 2)
        t_masked = tf.tile(tf.expand_dims(indices_true, 2), [1, 1, C])
        h =tf.where(t_masked, tf.zeros_like(h), h)
        v = tf.where(t_masked, tf.zeros_like(v), v)

        # S SUB-PROBLEM per [1]
        # Compute normin2, which is the sum of the h and v slices in reverse.
        normin2 = tf.zeros_like(S)
        normin2_h1 = h[:, M - 1:M, :] - h[:, 0:1, :]
        normin2_h2 = h[:, 1:] - h[:, :-1]
        normin_h = tf.concat([
            normin2_h1,
            -normin2_h2
        ], axis=1)
        normin2 += normin_h

        normin2_v1 = v[N - 1:N, :, :] - v[0:1, :, :]
        normin2_v2 = v[1:] - v[:-1]
        normin2_v = tf.concat([
            normin2_v1,
            -normin2_v2
        ], axis=0)
        normin2 += normin2_v

        # Compute FS function [1], matlab code: FS = (Normin1 + beta*fft2(Normin2))./Denormin;
        fft_numer2 = tf.signal.fft3d(tf.cast(normin2, dtype=tf.complex64))
        fs = (tf.cast(normin1, dtype=tf.complex64) + beta * fft_numer2) / tf.cast(denorm, dtype=tf.complex64)

        # Safety net, might not be necessary.
        # if tf.reduce_any(tf.math.is_nan(tf.math.real(fs))).eval():
        #     break

        # Inverse 2D Fast Fourier Transform to restore image based on current smoothing step
        S = tf.signal.ifft3d(fs)
        S = tf.math.real(S)

        beta = beta * kappa

        # Visual indicator that algorithm is working.
        # print(".", end="", flush=True)

    # Rescale
    # S = S.eval()
    # print()
    return S




def main():
    sess = 0
    # with tf.Session() as sess:
    path = '014_a.png'
    image_foder = '/media/ada/035ea81c-0b9a-4036-9c2a-a890e6fe0cee/ada/AnimeGANv3-528/samples/AnimeGANv3_7_4_Hayao_0.5_1.0_10.0_0_0/043/051_b.jpg'
    image_foder1 = '/media/ada/035ea81c-0b9a-4036-9c2a-a890e6fe0cee/ada/AnimeGANv3-528/samples/AnimeGANv3_7_4_Hayao_0.5_1.0_10.0_0_0/043/066_b.jpg'
    with tf.Session() as sess:

        Im = cv2.imread(image_foder)
        Im1 = cv2.imread(image_foder1)
        # Im = np.expand_dims(Im,0)
        t=time.time()
        S = l0_image_smoother( Im, 0.005)
        print(time.time()-t)
        t = time.time()
        S1 = l0_image_smoother( Im1, 0.005)
        print(time.time() - t)
        # cv2.imshow('a',np.squeeze(S))
        cv2.imshow('a', S.eval())
        cv2.imshow('b', S1.eval())
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
