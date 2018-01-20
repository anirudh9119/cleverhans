
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
from cleverhans_tutorials.tutorial_models import Linear, ReLU, Softmax, MLP
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.model import Model

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
    h_input_to_dae_ = model.layers['a1'].fprop(model.layers['l1'].fprop(x))
    output = tf.nn.tanh(tf.matmul(h_input_to_dae_, encoder[0]) + encoder_b[0])
    output_ = tf.nn.tanh(tf.matmul(output, tf.transpose(encoder[0])) + decoder_b[0])

    h2 = model.layers['a2'].fprop(model.layers['l2'].fprop(output_))

    presoftmax_ = model.layers['logits'].fprop(h2)
    preds = model.layers['probs'].fprop(presoftmax_)
    cost = tf.sqrt(tf.reduce_mean(tf.square(h_input_to_dae_ - output_)))
    return cost, preds

class MLP_Classifier_Condrec(Model):
    def __init__(self, input_shape):
        super(MLP_Classifier_Condrec, self).__init__()

        self.layers = {}

        self.layers['l1'] = Linear(512)
        self.layers['a1'] = ReLU()
        self.layers['l2'] = Linear(512)
        self.layers['a2'] = ReLU()

        self.layers['logits'] = Linear(10)
        self.layers['probs'] = Softmax()

        self.layers['l1'].set_input_shape(input_shape)
        self.layers['logits'].set_input_shape((None, 512))

        self.layers['l2'].set_input_shape((None, 512))

        #self.layers['l1'].set_input_shape(input_shape)
        #self.layers['a1']
        #self.layers['logits'] = Linear(10)
        #self.layers['probs'] = Softmax()
        #layer.set_input_shape(input_shape)

    def fprop(self, x, set_ref=False):
        states = {}

        states['l1'] = self.layers['l1'].fprop(x)
        states['a1'] = self.layers['a1'].fprop(states['l1'])
        #states['l2'] = self.layers['l2'].fprop(states['a1'])
        #states['a2'] = self.layers['a2'].fprop(states['l2'])

        states['logits'] = self.layers['logits'].fprop(tf.concat(states['a2'],axis=1))
        states['probs'] = self.layers['probs'].fprop(states['logits'])

        return states

def make_basic_fc(nb_classes=10,
                  input_shape=(None, 784)):
    print("Using classifier module defined in condrec module")
    layers = [Linear(512),
              ReLU(),
              #Linear(512),
              #ReLU(),
              Linear(nb_classes),
              Softmax()]

    model = MLP_Classifier_Condrec(input_shape)

    return model




