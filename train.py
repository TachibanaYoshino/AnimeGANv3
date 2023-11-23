from AnimeGANv3_shinkai import AnimeGANv3
import argparse
from tools.utils import *
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

"""parsing and configuration"""
def parse_args():
    desc = "AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)
    # parser.add_argument('--style_dataset', type=str, default='Hayao', help='dataset_name')
    parser.add_argument('--style_dataset', type=str, default='Shinkai', help='dataset_name')

    parser.add_argument('--init_G_epoch', type=int, default=5, help='The number of epochs for generator initialization')
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch size')
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')
    parser.add_argument('--load_or_resume', type=str.lower, default="load", choices=["load", "resume"], help='load is used for fine-tuning and resume is used to continue training.')

    parser.add_argument('--init_G_lr', type=float, default=2e-4, help='The generator learning rate')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='The learning rate')

    # ---------------------------------------------
    parser.add_argument('--img_size', type=int, nargs='+', default=[256,256], help='The size of image: H and W')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""train"""
def train():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
    if len(args.img_size) ==1:
        args.img_size = [args.img_size, args.img_size]

    # open session
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,inter_op_parallelism_threads=8,
                               intra_op_parallelism_threads=8,gpu_options=gpu_options)) as sess:
    # with tf.Session() as sess:
        gan = AnimeGANv3(sess, args)
        # build graph
        gan.build_train()
        # show network architecture
        show_all_variables()
        # start train
        gan.train()
        print("----- Training finished! -----")


if __name__ == '__main__':
    start_time = time.time()
    train()
    print("start time :", time.strftime("%Y %b %d %H:%M:%S %a",time.localtime(start_time)))
    print("end time :", time.strftime("%Y %b %d %H:%M:%S %a",time.localtime(time.time())))