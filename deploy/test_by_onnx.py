# -*- coding: utf-8 -*-
# @Time    : 2021/8/31 19:20
# @Author  : Xin Chen
# @File    : test_by_onnx.py
# @Software: PyCharm

import onnxruntime as ort
import time, os, cv2,argparse
import numpy as np
pic_form = ['.jpeg','.jpg','.png','.JPEG','.JPG','.PNG']
from glob import glob

def parse_args():
    desc = "Tensorflow implementation of AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_imgs_dir', type=str, default='/home/ada/test_data', help='video file or number for webcam')
    parser.add_argument('-m', '--model_path', type=str, default='models/AnimeGANv3_Hayao_36.onnx',  help='file path to save the modles')
    parser.add_argument('-o', '--output_path', type=str, default='./output/' ,help='output path')
    parser.add_argument('-d','--device', type=str, default='cpu', choices=["cpu","gpu"] ,help='running device')
    return parser.parse_args()

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def process_image(img, model_name):
    h, w = img.shape[:2]
    # resize image to multiple of 8s
    def to_8s(x):
        # If using the tiny model, the multiple should be 16 instead of 8.
        if 'tiny' in os.path.basename(model_name) :
            return 256 if x < 256 else x - x % 16
        else:
            return 256 if x < 256 else x - x % 8
    img = cv2.resize(img, (to_8s(w), to_8s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def load_test_data(image_path, model_name):
    img0 = cv2.imread(image_path).astype(np.float32)
    img = process_image(img0, model_name)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape

def save_images(images, image_path, size):
    images = (np.squeeze(images) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = cv2.resize(images, size)
    cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))

def Convert(input_imgs_path, output_path, onnx ="model.onnx", device="cpu"):
    # result_dir = opj(output_path, style_name)
    result_dir = output_path
    check_folder(result_dir)
    test_files = glob('{}/*.*'.format(input_imgs_path))
    test_files = [ x for x in test_files if os.path.splitext(x)[-1] in pic_form]
    if ort.get_device() == 'GPU' and device == "gpu":
        session = ort.InferenceSession(onnx, providers = ['CUDAExecutionProvider','CPUExecutionProvider',])
    else:
        session = ort.InferenceSession(onnx, providers=['CPUExecutionProvider', ])
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name

    begin = time.time()
    for i, sample_file  in enumerate(test_files) :
        t = time.time()
        sample_image, shape = load_test_data(sample_file, onnx)
        image_path = os.path.join(result_dir,'{0}'.format(os.path.basename(sample_file)))
        fake_img = session.run(None, {x : sample_image})
        save_images(fake_img[0], image_path, (shape[1], shape[0]))
        print(f'Processing image: {i}, image size: {shape[1], shape[0]}, ' + sample_file, f' time: {time.time() - t:.3f} s')
    end = time.time()
    print(f'Average time per image : {(end-begin)/len(test_files)} s')

if __name__ == '__main__':

    # onnx_file = 'AnimeGANv3_Hayao_36.onnx'
    # input_imgs_path = 'pic'
    # output_path = 'AnimeGANv3_Hayao_36'
    # Convert(input_imgs_path, output_path, onnx_file)

    arg = parse_args()
    Convert(arg.input_imgs_dir, arg.output_path, arg.model_path, arg.device)

