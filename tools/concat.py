# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from tqdm import tqdm

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def concat(pa, pb, out):
    filenames = [x for x in os.listdir(pa) if os.path.isfile(os.path.join(pa,x)) ]
    out =  check_folder(out)
    c1 = len(filenames)
    i = 0
    print(c1)
    for filepath in tqdm(filenames):
        i += 1
        img_path1 = os.path.join(pa, filepath)
        img_path2 = os.path.join(pb, filepath)
        print(img_path1, img_path2)
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        assert img1.shape == img2.shape
        h,w, c= img1.shape
        cut1 = np.ones((h, 7, 3), dtype='u8') * 255
        im_A1 = np.concatenate([img1, cut1], 1)
        im_AB = np.concatenate([im_A1, img2], 1)
        cv2.imwrite(os.path.join(out , str(i) + '.jpg'), im_AB)


if __name__ == '__main__':
    pa = r'/home/ada/shinkai/2'
    pb = r'/media/ada/035ea81c-0b9a-4036-9c2a-a890e6fe0cee/ada/V3_state of the art/DIV2K_train&valid_LR_bicubic_X2'
    out = r'/home/ada/shinkai/1'
    concat(pa, pb, out)