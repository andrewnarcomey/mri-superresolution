import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
from collections import defaultdict
import logging, scipy
import math
import tensorflow as tf
import tensorlayer as tl
from model2 import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every


def train(train_lr_imgs, train_hr_imgs):
    ## create folders to save result images and trained model
    checkpoint_dir = "models_checkpoints"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder(dtype='float32', shape=(batch_size, 512, 512, 1), name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder(dtype='float32', shape=(batch_size, 512, 512, 3), name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg
    net_vgg, vgg_target_emb = Vgg19_simple_api(input = (t_target_image_224 + 1) / 2, reuse=False, nchannels=3)
    _, vgg_predict_emb = Vgg19_simple_api(input = (t_predict_image_224 + 1) / 2, reuse=True, nchannels=3)


    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])

    tl.files.assign_params(sess, params, net_vgg)

    ###============================= TRAINING ===============================###
    
    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    start_time = time.time()
    for epoch in range(0, n_epoch_init):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        step_time = None
        for idx in range(0, len(train_hr_imgs), batch_size):
            if idx % 1000 == 0: step_time = time.time()
            b_imgs_hr = train_hr_imgs[idx:idx + batch_size]
            b_imgs_lr = train_lr_imgs[idx:idx + batch_size]
            b_imgs_hr = np.asarray(b_imgs_hr).reshape((batch_size,512,512,3))
            b_imgs_lr = np.asarray(b_imgs_lr).reshape((batch_size,512,512,1))

            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})

            if idx % 1000 == 0:
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

            total_mse_loss += errM
            n_iter += 1
            
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## save model
        tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
    print("G init took: %4.4fs" % (time.time() - start_time))

    ###========================= train GAN (SRGAN) =========================###
    start_time = time.time()
    epoch_losses = defaultdict(list)
    iter_losses = defaultdict(list)

    for epoch in range(0, n_epoch):
        ## update learning rate
        if epoch != 0 and decay_every != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        step_time = None
        for idx in range(0, len(train_hr_imgs), batch_size):
            if idx % 1000 == 0: step_time = time.time()
            b_imgs_hr = train_hr_imgs[idx:idx + batch_size]
            b_imgs_lr = train_lr_imgs[idx:idx + batch_size]
            b_imgs_hr = np.asarray(b_imgs_hr).reshape((batch_size,512,512,3))
            b_imgs_lr = np.asarray(b_imgs_lr).reshape((batch_size,512,512,1))
            
            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})
            ## update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})

            if idx % 1000 == 0:
                print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
                    (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)

            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

            iter_losses['d_loss'].append(errD)
            iter_losses['g_loss'].append(errG)
            iter_losses['mse_loss'].append(errM)
            iter_losses['vgg_loss'].append(errV)
            iter_losses['adv_loss'].append(errA)


        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)
        epoch_losses['d_loss'].append(total_d_loss)
        epoch_losses['g_loss'].append(total_g_loss)

        ## save model
        tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
        tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
    print("G train took: %4.4fs" % (time.time() - start_time))

    ## create visualizations for losses from training
    plot_total_losses(epoch_losses)
    plot_iterative_losses(iter_losses)
    for loss, values in epoch_losses.items():
        np.save(checkpoint_dir + "/epoch_" + loss + '.npy', np.asarray(values))
    for loss, values in iter_losses.items():
        np.save(checkpoint_dir + "/iter_" + loss + '.npy', np.asarray(values))
    print("[*] saved losses")

def evaluate(data, n_patients_train, eval_model, save_imgs=False):
    ## create folders for checkpoint and results
    checkpoint_dir = "models_checkpoints"
    results_dir = None
    if eval_model == '/g_srgan.npz':
        results_dir = "srgan_results"
    else:
        results_dir = "srresnet_results"
    tl.files.exists_or_mkdir(results_dir)
    
    ###========================== RESTORE G =============================###
    t_image = tf.placeholder('float32', [1, 512, 512, 1], name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + eval_model, network=net_g)

    ###======================= EVALUATION =============================###
    counter, imgs_evald, total_mse = 0, 0, 0

    for patient, values in data.items():
        if counter >= n_patients_train:
            print("[] Evaluating patient " + patient + " files")
            tl.files.exists_or_mkdir(results_dir + "/" + patient)
            valid_lr_imgs = values[0]
            valid_hr_imgs = values[1]
            patient_mse = 0
            for i in range(len(valid_lr_imgs)):
                valid_lr_img = valid_lr_imgs[i]
                valid_hr_img = valid_hr_imgs[i]
                valid_lr_img = np.asarray(valid_lr_img).reshape((512,512,1))
                valid_hr_img = np.asarray(valid_hr_img).reshape((512,512,3))
                out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})

                curr_mse = mse(out.reshape(512,512,3), valid_hr_img)
                imgs_evald += 1
                patient_mse += curr_mse
                total_mse += curr_mse

                if save_imgs:
                    tl.vis.save_image(out[0], results_dir + "/" + patient + "/" + str(patient) + "_" + str(i) + '_valid_gen.png')
                    tl.vis.save_image(valid_lr_img, results_dir + "/" + patient + "/" + str(patient) + "_" + str(i) + '_valid_lr.png')
                    tl.vis.save_image(valid_hr_img, results_dir + "/" + patient + "/" + str(patient) + "_" + str(i) + '_valid_hr.png')

                if i % 100 == 0: print("Batch " + str((i/float(100))) + "/" + str(math.ceil(len(valid_lr_imgs)/float(100))))

            patient_mse /= len(valid_lr_imgs)
            print("Average MSE: " + str(patient_mse))

        counter += 1

    total_mse /= imgs_evald
    print("[*] Evaluation -- total MSE: " + str(total_mse))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    data = load_info(path=config.data_path)

    n_patients_train = len(data) - config.n_eval
    train_lr_imgs = []
    train_hr_imgs = []
    counter = 0
    for patient, values in data.items():
        if counter < n_patients_train:
            for lr_val in values[0]:
                train_lr_imgs.append(lr_val)
            for hr_val in values[1]:
                train_hr_imgs.append(hr_val)
        counter += 1

    if tl.global_flag['mode'] == 'srgan':
        train(train_lr_imgs, train_hr_imgs)
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate(data, n_patients_train, '/g_srgan.npz')
        evaluate(data, n_patients_train, '/g_srgan_init.npz')
    else:
        raise Exception("Unknown --mode")
