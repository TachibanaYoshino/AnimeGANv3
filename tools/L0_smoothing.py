import time

import numpy as np
import cv2,os
from tqdm import tqdm

opj = os.path.join

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def psf2otf(psf, size):
    if not (0 in psf):
        # Pad the PSF to outsize
        psf = np.double(psf)
        psfsize = np.shape(psf)
        psfsize = np.array(psfsize)
        padsize = size - psfsize
        psf = np.lib.pad(psf, ((0, padsize[0]), (0, padsize[1])), 'constant')
        # Circularly shift otf so that the "center" of the PSF is at the (1,1) element of the array.
        psf = np.roll(psf, -np.array(np.floor(psfsize / 2), 'i'), axis=(0, 1))
        # Compute the OTF
        otf = np.fft.fftn(psf, axes=(0, 1))
        # Estimate the rough number of operations involved in the computation of the FFT.
        nElem = np.prod(psfsize, axis=0)
        nOps = 0
        for k in range(0, np.ndim(psf)):
            nffts = nElem / psfsize[k]
            nOps = nOps + psfsize[k] * np.log2(psfsize[k]) * nffts
            mx1 = (abs(np.imag(otf[:])).max(0)).max(0)
            mx2 = (abs(otf[:]).max(0)).max(0)
            eps = 2.2204e-16
            if mx1 / mx2 <= nOps * eps:
                otf = np.real(otf)
    else:
                otf = np.zeros(size)
    return otf


def L0Smoothing(Im, lamda=2e-2, kappa=2.0):

    S = Im
    betamax = 1e5
    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    N, M, D = np.shape(Im)
    sizeI2D = np.array([N, M])
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)
    Normin1 = np.fft.fft2(S, axes=(0, 1))
    Denormin2 = abs(otfFx) ** 2 + abs(otfFy) ** 2
    if D > 1:
        D2 = np.zeros((N, M, D), dtype=np.double)
        for i in range(D):
            D2[:, :, i] = Denormin2
        Denormin2 = D2
    beta = lamda * 2
    while beta < betamax:
        Denormin = 1 + beta * Denormin2
        # h-v subproblem
        h1 = np.diff(S, 1, 1)
        h2 = np.reshape(S[:, 0], (N, 1, 3)) - np.reshape(S[:, -1], (N, 1, 3))
        h = np.hstack((h1, h2))
        v1 = np.diff(S, 1, 0)
        v2 = np.reshape(S[0, :], (1, M, 3)) - np.reshape(S[-1, :], (1, M, 3))
        v = np.vstack((v1, v2))
        if D == 1:
            t = (h ** 2 + v ** 2) < lamda / beta
        else:
            t = np.sum((h ** 2 + v ** 2), 2) < lamda / beta
            t1 = np.zeros((N, M, D), dtype=np.bool)
            for i in range(D):
                t1[:, :, i] = t
            t = t1
        h[t] = 0
        v[t] = 0
        # S subproblem
        Normin2 = np.hstack((np.reshape(h[:, -1], (N, 1, 3)) - np.reshape(h[:, 0], (N, 1, 3)), -np.diff(h, 1, 1)))
        Normin2 = Normin2 + np.vstack((np.reshape(v[-1, :], (1, M, 3)) - np.
                                       reshape(v[0, :], (1, M, 3)), -np.diff(v, 1, 0)))
        FS = (Normin1 + beta * np.fft.fft2(Normin2, axes=(0, 1))) / Denormin
        S = np.real(np.fft.ifft2(FS, axes=(0, 1)))
        beta *= kappa
    return S


if __name__ == "__main__":
    image_foder = '../dataset/val/35.jpg'
    image_foder = '../dataset/val/2015-04-23 10_12_24.jpg'
    # image_foder = '../dataset/Hayao/style/11t.jpg'
    Im = cv2.imread(image_foder)
    S = L0Smoothing(Im, 0.005)
    S=S.clip(0,1)
    print(type(S[0,0,0]),S.max(),S.min())
    cv2.imshow('a',S.clip(0,1))
    cv2.waitKey(0)

    # image_foder = '../dataset/train_photo'
    # image_foder = '../dataset/Hayao/style'
    # # out_foder = '../dataset/photo_l0Smooth_0.005'
    # # out_foder = '../dataset/Hayao/style_l0Smooth_0.02'
    # # check_folder(out_foder)
    # imgs = os.listdir(image_foder)
    # for x in tqdm(imgs):
    #
    #     img =  cv2.imread(opj(image_foder,x))/255
    #     # S = L0Smoothing(img, 0.02)*255.
    #     t=time.time()
    #     S = L0Smoothing(img, 0.005)*255.
    #     S=S.clip(0,255)
    #     print(time.time()-t)
    #     # cv2.imwrite(opj(out_foder,x), S)




