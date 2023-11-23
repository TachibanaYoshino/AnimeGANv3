import argparse
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils import check_folder
import tensorflow as tf
from net import generator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def parse_args():
    desc = "AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoint/AnimeGANv3_Shinkai', help='Directory name to save the checkpoints')
    parser.add_argument('--save_dir', type=str, default='../checkpoint/generator_v3_Shinkai_weight', help='what style you want to get')
    return parser.parse_args()

def save(saver, sess, checkpoint_dir, model_name):
    save_path = os.path.join(checkpoint_dir, model_name + '.ckpt')
    saver.save(sess, save_path, write_meta_graph=True)
    return  save_path

def main(checkpoint_dir, save_dir):

    ckpt_dir = check_folder(save_dir)

    placeholder = tf.placeholder(tf.float32, [None, None, None, 3], name='AnimeGANv3_input')
    with tf.variable_scope("generator"):
        _ = generator.G_net(placeholder, is_training=True)
    with tf.variable_scope("generator", reuse=True):
        _ = generator.G_net(placeholder, is_training=False)

    variables = tf.contrib.framework.get_variables_to_restore()
    # only base tail
    generator_var = [var for var in variables if var.name.startswith('generator')  and ('main' in var.name or 'base' in var.name) and 'Adam' not in var.name and 'support' not in var.name]
    # the whole generator
    # generator_var = [var for var in variables if var.name.startswith('generator')  and 'Adam' not in var.name ]
    saver = tf.train.Saver(generator_var)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = ckpt_name.split('-')[-1]
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")
            return

        info = save(saver, sess, ckpt_dir, ckpt_name)

        print(f'save over : {info} ')



if __name__ == '__main__':
    arg = parse_args()
    print(arg.checkpoint_dir)
    main(arg.checkpoint_dir, arg.save_dir)
