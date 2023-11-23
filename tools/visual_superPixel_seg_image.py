import cv2, os
import numpy as np
from skimage import segmentation, color

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_simple_superpixel_improve(image_path, seg_num=200):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1, start_label=0,
                                  compactness=10, convert2lab=True)
    image = color.label2rgb(seg_label, image, bg_label=-1, kind='avg')
    return image


def get_superPixel (image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # rgb = cv2.resize(rgb, (256,256))
    # img_seg = segmentation.felzenszwalb(rgb, scale=5, sigma=0.8, min_size=100) # photo_superpixel
    # img_seg = segmentation.felzenszwalb(rgb, scale=200, sigma=0.5, min_size=200)
    img_seg = segmentation.felzenszwalb(rgb, scale=5, sigma=0.8, min_size=50)
    out = color.label2rgb(img_seg, rgb, bg_label=-1, kind='avg')
    return out

if __name__ == '__main__':
    temp = '../dataset/seg_train_5-0.8-50'
    # temp = '../dataset/seg_slic_train_1000'
    check_folder(temp)
    # image_foder = '../dataset/val'
    # image_foder = '../dataset/Hayao/style'
    image_foder = '../dataset/train_photo'

    for i, x in enumerate(os.listdir(image_foder)):
        print(i, x)
        # if x != '2013-11-10 12_45_41.jpg':
        # if x != '4.jpg':
        #     continue
        path = os.path.join(image_foder,x)
        img = get_superPixel(path)
        # img = get_simple_superpixel_improve(path, 1000)
        cv2.imwrite(os.path.join(temp,x), cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR))
        # cv2.imshow('super_seg',cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)



