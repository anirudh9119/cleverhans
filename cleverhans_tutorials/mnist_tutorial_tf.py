#!/usr/bin/env python
"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_cifar10 import data_cifar10
from cleverhans.utils_svhn import data_svhn
from cleverhans.utils_tf import model_train_2, model_eval_2
from cleverhans.attacks import FastGradientMethod, MadryEtAl
from cleverhans.utils import AccuracyReport, set_log_level

#import os
#import math
FLAGS = flags.FLAGS

#from autoencoder_tied_arch import autoencoder, get_output
#from classifier_basic import autoencoder, get_output
#from autoencoder_pspace import autoencoder, get_output
#from autoencoder_condrec import autoencoder, get_output, make_basic_fc
#from autoencoder_modelmatch import autoencoder, get_output, make_basic, compute_rec_error
from conv_autoencoder_mm import autoencoder, get_output, make_basic, compute_rec_error


def create_adv_by_name(model, x, attack_type, sess, dataset, y=None, **kwargs):
    attack_names = {'FGSM': FastGradientMethod,
                    'MadryEtAl': MadryEtAl,
                    }

    if attack_type not in attack_names:
        raise Exception('Attack %s not defined.' % attack_type)

    attack_params_shared = {
        #'mnist': {'eps': .3, 'eps_iter': 1.2, 'clip_min': 0., 'clip_max': 1.,'nb_iter':40},
        'mnist': {'eps': 1.0, 'eps_iter': 1.2, 'clip_min': 0., 'clip_max': 1.,
                  'nb_iter': 40},
        'cifar10': {'eps': 8./255, 'eps_iter': 0.01, 'clip_min': 0.,
                    'clip_max': 1., 'nb_iter': 20},
        'svhn': {'eps': 1.0, 'eps_iter': 1.2, 'clip_min': 0., 'clip_max': 1.,
                    'nb_iter': 40},
    }

    with tf.variable_scope(attack_type):
        attack_class = attack_names[attack_type]
        attack = attack_class(model, sess=sess)

        # Extract feedable and structural keyword arguments from kwargs
        fd_kwargs = attack.feedable_kwargs.keys() + attack.structural_kwargs
        params = attack_params_shared[dataset].copy()
        params.update({k: v for k, v in kwargs.items() if v is not None})
        params = {k: v for k, v in params.items() if k in fd_kwargs}

        if '_y' in attack_type:
            params['y'] = y
        logging.info(params)
        adv_x = attack.generate(x, **params)

    return adv_x



def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=20, batch_size=128,
                   learning_rate=0.001,
                   attack_name='MadryEtAl',
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64,
                   dataset='mnist',
                   num_threads=None):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    if dataset == 'cifar10':
        train_end = 50000
    elif dataset == "svhn":
        train_end = 604388
        test_end = 26032

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    if dataset == "mnist":
        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
        num_features = 28*28
        shape_in = [-1,28*28]
    elif dataset == "cifar10":
        X_train, Y_train, X_test, Y_test = data_cifar10()
        num_features = 32*32*3
        shape_in = [-1,32*32*3]
    elif dataset == "svhn":
        X_train, Y_train, X_test, Y_test = data_svhn()
        num_features = 32*32*3
        shape_in = [-1,32*32*3]
    else:
        raise Exception()

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    #model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    rng = np.random.RandomState([2017, 8, 30])


    if clean_train:
        x= tf.reshape(x, [-1, num_features])
        encoder, encoder_b, decoder_b, autoencoder_params, corrupt_prob = autoencoder(dataset,dimensions=[512, 128])
        model = make_basic(encoder, encoder_b, decoder_b, autoencoder_params,input_shape=(None,num_features))
        preds,hpreclean,hpostclean = get_output(model, x, encoder, encoder_b, decoder_b, autoencoder_params)

        cost = compute_rec_error(hpreclean,hpostclean)

        #preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc,rec_error,re_true,re_false = model_eval_2(
                sess, x, y, corrupt_prob, preds, X_test, Y_test, args=eval_params, rec_cost=cost, x_shape_in=shape_in)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)
            print('rec error on legitimate examples: %0.4f' % rec_error)
            print('rec error on legitimate examples corr class (%0.4f) and incorr. class (%0.4f)' % (re_true,re_false))
        model_train_2(sess, x, y, corrupt_prob, preds, X_train, Y_train, dataset, evaluate=evaluate,rec_cost=cost,
                    args=train_params, rng=rng)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval_2(
                sess, x, y, corrupt_prob, preds, X_train, Y_train, rec_cost=cost, args=eval_params, x_shape_in=shape_in)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph

        adv_x = create_adv_by_name(model, x, attack_name, sess, 'mnist')
        adv_x = tf.reshape(adv_x, [-1, num_features])
        preds_adv,hpreadv,hpostadv = get_output(model, adv_x, encoder, encoder_b, decoder_b, autoencoder_params)

        cost = compute_rec_error(hpreadv,hpostadv)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc,rec_error,re_true,re_false = model_eval_2(sess, x, y, corrupt_prob, preds_adv, X_test, Y_test, rec_cost=cost, args=eval_par, x_shape_in=shape_in)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        print('rec error adv->adv on adversarial examples: %0.4f\n' % rec_error)
        print('rec error adv->adv examples corr class (%0.4f) and incorr. class (%0.4f)' % (re_true,re_false))
        report.clean_train_adv_eval = acc

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval_2(sess, x, y, corrupt_prob, preds_adv, X_train,
                             Y_train, args=eval_par, x_shape_in=shape_in)
            report.train_clean_train_adv_eval = acc

        print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    #x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    x= tf.reshape(x, [-1, num_features])
    #model_2 = make_basic_cnn(nb_filters=nb_filters)
    with_denoising = True
    print("using denoising for training adversarial", with_denoising)
    assert with_denoising == True

    if with_denoising:
        encoder_2, encoder_b_2, decoder_b_2, autoencoder_params, corrupt_prob_2 = autoencoder(dataset, dimensions=[512, 320])
        model_2 = make_basic(encoder_2, encoder_b_2, decoder_b_2, autoencoder_params,input_shape=(None,num_features))

    cost_2 = 0
    corrupt_prob_2 = tf.placeholder(tf.float32, [1])
    adv_x_2 = create_adv_by_name(model_2, x, attack_name, sess, 'mnist')
    adv_x_2 = tf.reshape(adv_x_2, [-1, num_features])

    if with_denoising == False:
        preds_2 = model_2(x)
        if not backprop_through_attack:
            adv_x_2 = tf.stop_gradient(adv_x_2)
        preds_2_adv = model_2(adv_x_2)
    else:
        encoder_2, encoder_b_2, decoder_b_2, autoencoder_params, corrupt_prob_2 = autoencoder(dataset, dimensions=[512, 320])
        preds_2,hpreclean,hpostclean = get_output(model_2, x, encoder_2, encoder_b_2, decoder_b_2, autoencoder_params)
        cost_2 = compute_rec_error(hpreclean,hpostclean)
        if not backprop_through_attack:
            adv_x_2 = tf.stop_gradient(adv_x_2)
        preds_2_adv,hpreadv,hpostadv = get_output(model_2, adv_x_2, encoder_2, encoder_b_2, decoder_b_2, autoencoder_params)
        cost_2_adv = compute_rec_error(hpreclean,hpostadv)


    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy,rec_error,re_true,re_false = model_eval_2(sess, x, y, corrupt_prob_2, preds_2, X_test, Y_test,rec_cost=cost_2,
                              args=eval_params,x_shape_in=shape_in)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        print('Test rec error clean->clean on legitimate examples: %0.4f' % rec_error)
        print('rec error on legitimate examples corr class (%0.4f) and incorr. class (%0.4f)' % (re_true,re_false))
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy, rec_error,re_true,re_false = model_eval_2(sess, x, y, corrupt_prob_2, preds_2_adv, X_test, Y_test, rec_cost=cost_2_adv,
                              args=eval_params,x_shape_in=shape_in)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        print('Test rec error adv->clean on adversarial examples: %0.4f' % rec_error)
        print('rec error adv->clean corr class (%0.4f) and incorr. class (%0.4f)' % (re_true,re_false))
        report.adv_train_adv_eval = accuracy

    # Perform and evaluate adversarial training
    model_train_2(sess, x, y, corrupt_prob_2, preds_2, X_train, Y_train, dataset,
                rec_cost=cost_2+cost_2_adv, predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, rng=rng)

    # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval_2(sess, x, y, corrupt_prob_2, preds_2, X_train, Y_train,
                              args=eval_params,x_shape_in=shape_in)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval_2(sess, x, y, corrupt_prob_2, preds_2_adv, X_train,
                              Y_train, args=eval_params,x_shape_in=shape_in)
        report.train_adv_train_adv_eval = accuracy

    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   attack_name='MadryEtAl', #'FGSM'
                   #attack_name='FGSM',
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   dataset=FLAGS.dataset)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 3, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    flags.DEFINE_string('dataset', 'mnist', "The dataset to train and evaluate on")
    tf.app.run()



