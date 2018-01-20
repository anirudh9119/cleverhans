
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_train_2, model_eval, model_eval_2
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn, make_basic_fc
from cleverhans.utils import AccuracyReport, set_log_level

import math
FLAGS = flags.FLAGS

print("Importing autoencoder classifier!")

def corrupt(x):
    """Take an input tensor and add uniform masking.
    """
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))


def autoencoder(dimensions=[512, 256, 64]):
    """Build a deep denoising autoencoder w/ tied weights.
    """
    # input to the network
    h = tf.placeholder(tf.float32, [None, dimensions[0]], name='h')

    # I'll change this to 1 during training
    # Put it back to 0.
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(h) * corrupt_prob + h * (1 - corrupt_prob)

    # Build the encoder
    encoder = []
    encoder_b = []
    decoder_b = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        encoder_b.append(b)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # latent representation
    #z = current_input
    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
        decoder_b.append(b)
    # now have the reconstruction through the network
    #y = current_input
    # cost function measures pixel-wise difference
    #cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
    return encoder, encoder_b, decoder_b, corrupt_prob
            #{'x': x, 'z': z, 'y': y,
            #'corrupt_prob': corrupt_prob,
            #'cost': cost}

def get_output(model, x, encoder, encoder_b, decoder_b):
        #x= tf.reshape(x, [-1, 784])
    h_input_to_dae_ = model.layers[1].fprop(model.layers[0].fprop(x))
    output = tf.nn.tanh(tf.matmul(h_input_to_dae_, encoder[0]) + encoder_b[0])
    output_ = tf.nn.tanh(tf.matmul(output, tf.transpose(encoder[0])) + decoder_b[0])
    presoftmax_ = model.layers[2].fprop(output_)
    preds = model.layers[3].fprop(presoftmax_)
    cost = tf.sqrt(tf.reduce_mean(tf.square(h_input_to_dae_ - output_)))
    return cost, preds




