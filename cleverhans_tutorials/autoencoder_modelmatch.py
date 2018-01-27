
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

def gaussian_noise(x,std=0.1):
    return x + tf.cast(tf.random_normal(shape=tf.shape(x),stddev=std), tf.float32)

def compute_rec_error(hpre,hpost):
    return tf.reduce_mean(tf.square(tf.stop_gradient(hpre) - hpost),axis=1)

def autoencoder(dataset,dimensions=[512, 256, 64]):
    """Build a deep denoising autoencoder w/ tied weights.
    """
    
    if dataset == "cifar10":
        num_features = 32*32*3
    elif dataset == "mnist":
        num_features = 28*28
    
    # input to the network
    h = tf.placeholder(tf.float32, [None, dimensions[0]], name='h')

    # I'll change this to 1 during training
    # Put it back to 0.
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(h) * corrupt_prob + h * (1 - corrupt_prob)

    autoencoder_params = {}

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
    

    autoencoder_params['x_w1'] = tf.Variable(tf.random_uniform([num_features, 512],-1.0 / math.sqrt(512),1.0 / math.sqrt(512)))
    autoencoder_params['x_w2'] = tf.Variable(tf.random_uniform([512, 784],-1.0 / math.sqrt(512),1.0 / math.sqrt(512)))

    return encoder, encoder_b, decoder_b, autoencoder_params, corrupt_prob
            #{'x': x, 'z': z, 'y': y,
            #'corrupt_prob': corrupt_prob,
            #'cost': cost}

def h_autoencoder(inp,encoder,encoder_b,decoder_b,autoencoder_params):

    output = tf.nn.tanh(tf.matmul(inp, encoder[0]) + encoder_b[0])
    output_ = tf.nn.tanh(tf.matmul(output, tf.transpose(encoder[0])) + decoder_b[0])

    return output_


def get_output(model, x, encoder, encoder_b, decoder_b, autoencoder_params,return_state_map=False,autoenc_x=False,scope=""):
    #x= tf.reshape(x, [-1, 784])

    if autoenc_x:
        xa = tf.nn.leaky_relu(tf.matmul(x, autoencoder_params['x_w1']))
        xuse = tf.matmul(xa, autoencoder_params['x_w2'])
    else:
        xuse = x

    h_input_to_dae_ = tf.nn.relu(model.layers['l1'].fprop(xuse))
    
    output_ = h_autoencoder(gaussian_noise(h_input_to_dae_),encoder,encoder_b,decoder_b,autoencoder_params)
    
    output_blockin = h_autoencoder(gaussian_noise(tf.stop_gradient(h_input_to_dae_)),encoder,encoder_b,decoder_b,autoencoder_params)

    if autoenc_x:
        cost += tf.sqrt(tf.reduce_mean(tf.square(x - xrec)))

    #h2 = model.layers['a2'].fprop(model.layers['l2'].fprop(tf.concat([output_],axis=1)))

    presoftmax_ = model.layers['logits'].fprop(output_)
    preds = model.layers['probs'].fprop(presoftmax_)

    if return_state_map:
        return {'logits' : presoftmax_, 'probs' : preds}
    else:
        return preds,h_input_to_dae_,output_blockin

class MLP_Classifier_Condrec(Model):
    def __init__(self, input_shape, encoder, encoder_b, decoder_b, autoencoder_params):
        super(MLP_Classifier_Condrec, self).__init__()

        self.layers = {}

        self.layers['l1'] = Linear(512)
        #self.layers['a1'] = LeakyReLU()

        self.layers['l2'] = Linear(512)
        self.layers['a2'] = ReLU()

        self.layers['logits'] = Linear(10)
        self.layers['probs'] = Softmax()

        self.layers['l1'].set_input_shape(input_shape)
        #self.layers['l2'].set_input_shape((None,512))
        self.layers['logits'].set_input_shape((None, 512))

        self.encoder = encoder
        self.encoder_b = encoder_b
        self.decoder_b = decoder_b
        self.autoencoder_params = autoencoder_params

    def fprop(self, x, set_ref=False):
        states = get_output(self, x, self.encoder, self.encoder_b, self.decoder_b, self.autoencoder_params, return_state_map=True)

        return states

def make_basic(encoder, encoder_b, decoder_b,autoencoder_params,nb_classes=10,
                  input_shape=(None, 784)):
    print("Using classifier module defined in condrec module")

    model = MLP_Classifier_Condrec(input_shape, encoder, encoder_b, decoder_b,autoencoder_params)

    return model




