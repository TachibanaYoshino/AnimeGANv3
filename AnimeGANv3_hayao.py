from tools.ops import *
from tools.utils import *
from glob import glob
import time
import numpy as np
from joblib import Parallel, delayed
from skimage import segmentation, color
from net import generator
from net.discriminator import D_net
from tools.data_loader import ImageGenerator
from tools.GuidedFilter import guided_filter
from tools.L0_smoothing import L0Smoothing


class AnimeGANv3(object) :
    def __init__(self, sess, args):
        self.model_name = 'AnimeGANv3'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.style_dataset

        self.epoch = args.epoch
        self.init_G_epoch = args.init_G_epoch

        self.batch_size = args.batch_size
        self.save_freq = args.save_freq
        self.load_or_resume = args.load_or_resume

        self.init_G_lr = args.init_G_lr
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr

        self.img_size = args.img_size
        self.img_ch = args.img_ch
        """ Discriminator """
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        self.val_real = tf.placeholder(tf.float32, [1, None, None, self.img_ch], name='val_input')

        self.real_photo = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='real_photo')
        self.photo_superpixel = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='photo_superpixel')

        self.fake_superpixel = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='fake_superpixel')
        self.fake_NLMean_l0 = tf.placeholder(tf.float32,[self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='fake_NLMean_l0')

        self.anime = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='anime_image')
        self.anime_smooth = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='anime_smooth_image')

        self.real_generator = ImageGenerator('./dataset/train_photo', self.img_size, self.batch_size)
        self.anime_image_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/style'), self.img_size, self.batch_size)
        self.anime_smooth_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/smooth'), self.img_size, self.batch_size)
        self.dataset_num = max(self.real_generator.num_images, self.anime_image_generator.num_images)

        print()
        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# init_G_epoch : ", self.init_G_epoch)
        print("# training image size [H, W] : ", self.img_size)
        print("# init_G_lr,g_lr,d_lr : ", self.init_G_lr,self.g_lr,self.d_lr)
        print()

    def generator(self, x_init, is_training, reuse=False, scope="generator"):

        with tf.variable_scope(scope, reuse=reuse):
            fake_s, fake_m =  generator.G_net(x_init, is_training)
            return fake_s, fake_m

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
            return D_net(x_init, self.sn, ch=32, reuse=reuse, scope=scope)

    ##################################################################################
    def build_train(self):

        """ Define Generator, Discriminator """
        self.generated_s,  self.generated_m = self.generator(self.real_photo, is_training=True)
        self.generated = self.tanh_out_scale(guided_filter(self.sigm_out_scale(self.generated_s),self.sigm_out_scale(self.generated_s), 2, 0.01)) #0.25**2

        """for val"""
        self.val_generated_s, self.val_generated_m = self.generator(self.val_real, is_training=False, reuse=True)
        self.val_generated = self.tanh_out_scale(guided_filter(self.sigm_out_scale(self.val_generated_s), self.sigm_out_scale(self.val_generated_s), 2, 0.01))  # 0.25**2

        # gray maping
        self.fake_sty_gray = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(self.generated))
        self.anime_sty_gray = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(self.anime))
        self.gray_anime_smooth = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(self.anime_smooth))

        # support
        gray_anime_smooth_logit = self.discriminator(self.gray_anime_smooth)
        anime_gray_logit = self.discriminator(self.anime_sty_gray,  reuse=True, )
        fake_gray_logit = self.discriminator(self.fake_sty_gray, reuse=True, )
        # main
        generated_m_logit = self.discriminator(self.generated_m, scope="discriminator_main")
        fake_NLMean_logit = self.discriminator(self.fake_NLMean_l0, reuse=True, scope="discriminator_main")

        """ Define Loss """
        # init G
        self.Pre_train_G_loss = con_loss(self.real_photo, self.generated) + con_loss(self.real_photo, self.generated_m)

        # gan
        """support"""
        self.con_loss =  con_loss(self.real_photo, self.generated, 0.5)
        # self.s22, self.s33, self.s44  = style_loss_decentralization_3(self.anime_sty_gray, self.fake_sty_gray,  [0.1, 1., 9.])
        self.s22, self.s33, self.s44  = style_loss_decentralization_3(self.anime_sty_gray, self.fake_sty_gray,  [0.1, 5., 25.])
        self.sty_loss = self.s22  + self.s33 +  self.s44

        self.rs_loss =  region_smoothing_loss(self.fake_superpixel, self.generated, 0.2 ) \
                        + VGG_LOSS(self.photo_superpixel, self.generated) * 0.2


        self.color_loss =  Lab_color_loss(self.real_photo, self.generated, 10. )
        self.tv_loss  = 0.001 * total_variation_loss(self.generated)

        self.g_adv_loss = generator_loss(fake_gray_logit)
        self.G_support_loss = self.g_adv_loss + self.con_loss + self.sty_loss   + self.rs_loss +  self.color_loss +self.tv_loss
        self.D_support_loss = discriminator_loss(anime_gray_logit, fake_gray_logit) \
                            + discriminator_loss_346(gray_anime_smooth_logit) * 2.0
        """main"""
        self.tv_loss_m = 0.001 * total_variation_loss(self.generated_m)
        self.p4_loss = VGG_LOSS(self.fake_NLMean_l0, self.generated_m) * 0.5
        self.p0_loss = L1_loss(self.fake_NLMean_l0, self.generated_m) * 50.
        self.g_m_loss = generator_loss_m(generated_m_logit) * 0.02

        self.G_main_loss = self.g_m_loss + self.p0_loss + self.p4_loss + self.tv_loss_m
        self.D_main_loss = discriminator_loss_m(fake_NLMean_logit, generated_m_logit) * 0.1

        self.Generator_loss =  self.G_support_loss +  self.G_main_loss
        self.Discriminator_loss = self.D_support_loss + self.D_main_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        # init G
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.init_G_optim = tf.train.AdamOptimizer(self.init_G_lr, beta1=0.5, beta2=0.999).minimize(self.Pre_train_G_loss, var_list=G_vars)
        ###
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.G_optim = tf.train.AdamOptimizer(self.g_lr , beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
            self.D_optim = tf.train.AdamOptimizer(self.d_lr , beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        #
        self.Summary_G_init_loss = tf.summary.scalar("G_init", self.Pre_train_G_loss)
        #
        self.Summary_G_adv = tf.summary.scalar("G_adv", self.g_adv_loss)
        self.Summary_G_con_loss = tf.summary.scalar("con_loss", self.con_loss)
        self.Summary_G_rs_loss = tf.summary.scalar("rs_loss", self.rs_loss)
        self.Summary_G_sty_loss = tf.summary.scalar("sty_loss", self.sty_loss)
        self.Summary_G_color_loss = tf.summary.scalar("color_loss", self.color_loss)
        self.Summary_G_tv_loss = tf.summary.scalar("tv_loss", self.tv_loss)

        self.Summary_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.Summary_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

        #------
        self.pretrianed_G_merge = tf.summary.merge([self.Summary_G_init_loss])
        self.GD_loss_merge = tf.summary.merge([self.Summary_G_loss,self.Summary_G_adv, self.Summary_G_con_loss, self.Summary_G_rs_loss, self.Summary_G_sty_loss,self.Summary_G_color_loss,self.Summary_G_tv_loss, self.Summary_D_loss])

    def train(self):
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())
        # saver to save model
        variables = tf.contrib.framework.get_variables_to_restore()
        variables_to_resotre = [v for v in variables if 'Adam' not in v.name]
        self.saver_load = tf.train.Saver(var_list=variables_to_resotre, max_to_keep=self.epoch)
        self.saver = tf.train.Saver(max_to_keep=self.epoch)

        # summary writer
        # self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        """ Input Image"""
        real_photo_op, anime_img_op, anime_smooth_op  = self.real_generator.load_images(), self.anime_image_generator.load_images(), self.anime_smooth_generator.load_images()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = checkpoint_counter + 1
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            print(" [!] Load failed...")

        # loop for epoch
        steps = int(self.dataset_num / self.batch_size)
        for epoch in range(start_epoch, self.epoch):
            for idx in range(steps):
                start_time = time.time()
                real_photo, anime_, anime_smooth_ = self.sess.run([real_photo_op, anime_img_op, anime_smooth_op])
                train_feed_dict = {
                    self.real_photo:real_photo[0],
                    self.photo_superpixel: real_photo[1],
                    self.anime:anime_[0],
                    self.anime_smooth:anime_smooth_[0],
                }

                """ pre-training G """
                if epoch < self.init_G_epoch :
                    _, init_loss, summary_str = self.sess.run([self.init_G_optim,self.Pre_train_G_loss, self.pretrianed_G_merge], feed_dict = train_feed_dict)
                    # self.writer.add_summary(summary_str, epoch)
                    step_time = time.time() - start_time
                    print("Epoch: %3d, Step: %5d / %5d, time: %.3fs, ETA: %.2fs, Pre_train_G_loss: %.6f" % (
                           epoch, idx, steps, step_time, step_time*(steps-idx+1), init_loss))

                    """style transfer"""
                else:

                    """ Update G """
                    # output fake image
                    inter_out_s, inter_out= self.sess.run([self.generated_s, self.generated], feed_dict=train_feed_dict)
                    # superpixel_batch = self.get_simple_superpixel(inter_out, seg_num=200)
                    superpixel_batch = self.get_seg(inter_out)
                    fake_NLMean_batch = self.get_NLMean_l0(inter_out_s)

                    train_feed_dict.update(
                        {
                            self.fake_superpixel: superpixel_batch,
                            self.fake_NLMean_l0: fake_NLMean_batch,
                        }
                    )

                    _, G_loss, G_support_loss, g_adv_loss, con_loss, rs_loss, sty_loss, s22, s33, s44, color_loss, tv_loss, \
                               G_main_loss, g_m_loss, p0_loss,p4_loss,tv_loss_m = self.sess.run([self.G_optim,
                                                                                                          self.Generator_loss,
                                                                                                          self.G_support_loss,
                                                                                                          self.g_adv_loss,
                                                                                                          self.con_loss,
                                                                                                          self.rs_loss,
                                                                                                          self.sty_loss, self.s22, self.s33, self.s44,
                                                                                                          self.color_loss,
                                                                                                          self.tv_loss,
                                                                                                          self.G_main_loss,
                                                                                                          self.g_m_loss,
                                                                                                          self.p0_loss,
                                                                                                          self.p4_loss,
                                                                                                          self.tv_loss_m
                                                                                                          ], feed_dict = train_feed_dict)

                    """ Update D """
                    _, D_loss, D_support_loss, D_main_loss, summary_str = self.sess.run([self.D_optim,
                                                            self.Discriminator_loss,
                                                            self.D_support_loss,
                                                            self.D_main_loss,
                                                            self.GD_loss_merge],
                                                            feed_dict=train_feed_dict)
                    # self.writer.add_summary(summary_str, epoch)
                    step_time = time.time() - start_time
                    info = f'Epoch: {epoch:3d}, Step: {idx:5d} /{steps:5d}, time: {step_time:.3f}s, ETA: {step_time*(steps-idx+1):.2f}s, ' + \
                           f'D_loss:{D_loss:.3f} ~ G_loss: {G_loss:.3f} || ' + \
                           f'G_support_loss: {G_support_loss:.6f}, g_s_loss: {g_adv_loss:.6f}, con_loss: {con_loss:.6f}, rs_loss: {rs_loss:.6f}, sty_loss: {sty_loss:.6f}, s22: {s22:.6f}, s33: {s33:.6f}, s44: {s44:.6f}, color_loss: {color_loss:.6f}, tv_loss: {tv_loss:.6f} ~ D_support_loss: {D_support_loss:.6f} || ' + \
                           f'G_main_loss: {G_main_loss:.6f}, g_m_loss: {g_m_loss:.6f}, p0_loss: {p0_loss:.6f}, p4_loss: {p4_loss:.6f}, tv_loss_m: {tv_loss_m:.6f} ~ D_main_loss: {D_main_loss:.6f}'
                    print(info)
            # 2---------------------------------------------------------------------------------

            if (epoch + 1) >= self.init_G_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, epoch)

            if (epoch + 1) >= self.init_G_epoch:
                """ Result Image """
                val_files = glob('./dataset/{}/*.*'.format('val'))
                save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
                check_folder(save_path)
                for i, sample_file in enumerate(val_files):
                    print('val: '+ str(i) + sample_file)
                    sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                    val_real,test_s1, test_s0, test_m = self.sess.run([self.val_real,self.val_generated,self.val_generated_s,self.val_generated_m ],feed_dict = {self.val_real:sample_image} )
                    save_images(val_real, save_path+'{:03d}_a.jpg'.format(i))
                    save_images(test_s1, save_path+'{:03d}_b.jpg'.format(i))
                    save_images(test_s0, save_path+'{:03d}_c.jpg'.format(i))
                    save_images(test_m, save_path+'{:03d}_d.jpg'.format(i))


    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def get_seg(self, batch_image):
        def get_superpixel(image):
            image = (image + 1.) * 127.5
            image = np.clip(image, 0, 255).astype(np.uint8)  # [-1. ,1.] ~ [0, 255]
            image_seg = segmentation.felzenszwalb(image, scale=5, sigma=0.8, min_size=50)
            image = color.label2rgb(image_seg, image,  bg_label=-1, kind='avg').astype(np.float32)
            image = image / 127.5 - 1.0
            return image
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(delayed(get_superpixel) (image) for image in batch_image)
        return np.array(batch_out)

    def get_simple_superpixel(self, batch_image, seg_num=200):
        def process_slic(image):
            seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1, start_label=0,compactness=10, convert2lab=True)
            image = color.label2rgb(seg_label, image, bg_label=-1, kind='avg')
            return image
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)(image )for image in batch_image)
        return np.array(batch_out)

    def get_NLMean_l0(self, batch_image, ):
        def process_revision(image):
            image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
            image = cv2.fastNlMeansDenoisingColored(image, None, 5, 6, 5, 7)
            image = L0Smoothing(image/255, 0.005).astype(np.float32) * 2. - 1.
            return image.clip(-1., 1.)
        num_job = np.shape(batch_image)[0]
        batch_out = Parallel(n_jobs=num_job)(delayed(process_revision)(image) for image in batch_image)
        return np.array(batch_out)


    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir) # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path) # first line
            if  "resume" == self.load_or_resume :
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            else:
                self.saver_load.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


    def to_lab(self, x):
        """
        @param x: image tensor  [-1.0, 1.0]
        # @return:  image tensor  [-1.0, 1.0]
        @return:  image tensor  [0.0, 1.0]
        """
        x = (x + 1.0) / 2.0
        x = rgb_to_lab(x)
        y = tf.concat([tf.expand_dims(x[:, :, :, 0] / 100.,-1), tf.expand_dims((x[:, :, :, 1]+128.)/255.,-1), tf.expand_dims((x[:, :, :, 2]+128.)/255.,-1)], axis=-1)
        return y


    def sigm_out_scale(self, x):
        """
        @param x: image tensor  [-1.0, 1.0]
        @return:  image tensor  [0.0, 1.0]
        """
        # [-1.0, 1.0]  to  [0.0, 1.0]
        x = (x + 1.0) / 2.0
        return  tf.clip_by_value(x, 0.0, 1.0)


    def tanh_out_scale(self, x):
        """
        @param x: image tensor  [0.0, 1.0]
        @return:  image tensor  [-1.0, 1.0]
        """
        # [0.0, 1.0]   to  [-1.0, 1.0]
        x = (x - 0.5) * 2.0
        return  tf.clip_by_value(x,-1.0, 1.0)
