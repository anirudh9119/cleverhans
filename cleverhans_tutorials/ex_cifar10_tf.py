from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import math
#from cleverhans.utils_keras import cnn_model
from cleverhans.utils_tf import model_train, model_eval#, batch_eval
from cleverhans_tutorials.tutorial_models import model_cnn#make_basic_fc#, make_basic_cnn
from cleverhans.utils import AccuracyReport#, set_log_level
from cleverhans.attacks import FastGradientMethod

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/tmp', 'Directory storing the saved model.')
flags.DEFINE_string(
    'filename', 'cifar10.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')


def corrupt(x):
    """Take an input tensor and add uniform masking.
    """
    return tf.add(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
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

def get_conv_output(model, x, encoder, encoder_b, decoder_b):
     #h_input_to_dae_ = model.layers[1].fprop(model.layers[0].fprop(x))
     output = tf.nn.tanh(tf.matmul(x, encoder[0]) + encoder_b[0])
     output_ = tf.nn.tanh(tf.matmul(output, tf.transpose(encoder[0])) + decoder_b[0])
     #presoftmax_ = model.layers[2].fprop(output_)
     #preds = model.layers[3].fprop(presoftmax_)
     #cost = tf.sqrt(tf.reduce_mean(tf.square(tf.stop_gradient(x) - output_)))
     cost = tf.sqrt(tf.reduce_mean(tf.square(x - output_)))
     return output_,cost#, preds

def data_cifar10():
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def main(argv=None):
    """
    CIFAR10 CleverHans tutorial
    :return:
    """
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get CIFAR10 test data
    X_train, Y_train, X_test, Y_test = data_cifar10()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    batch_size = 128
    rng = np.random.RandomState([2017, 8, 30])
    testing=False
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                    'clip_max': 1.}

    clean_train = True
    encoder, encoder_b, decoder_b, corrupt_prob = autoencoder(dimensions=[512, 300])
    if clean_train:
        model = model_cnn()
        #x= tf.reshape(x, [-1, 784])
        #cost, preds = get_cnn_output(model, x, encoder, encoder_b, decoder_b)

        layer_1_out = model.layers[1].fprop(model.layers[0].fprop(x))
        layer_2_out = model.layers[3].fprop(model.layers[2].fprop(layer_1_out))
        layer_3_out = model.layers[4].fprop(model.layers[3].fprop(layer_2_out))
        layer_4_out = model.layers[6].fprop(model.layers[5].fprop(layer_3_out))

        output_, cost_ = get_conv_output(model, layer_4_out, encoder, encoder_b, decoder_b)
        preds_ = model.layers[8].fprop(model.layers[7].fprop(output_))


        #preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, corrupt_prob, preds_, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            #assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('First phase, Test accuracy on legitimate examples: %0.4f' % acc)
        model_train(sess, x, y, corrupt_prob, preds_, X_train, Y_train, cost_, evaluate=evaluate,
                    args=train_params, rng=rng)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, corrupt_prob, preds_, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        #adv_x = tf.reshape(adv_x, [-1, 784])
        #cost, preds_adv = get_output(model, adv_x, encoder, encoder_b, decoder_b)

        layer_1_out = model.layers[1].fprop(model.layers[0].fprop(adv_x))
        layer_2_out = model.layers[3].fprop(model.layers[2].fprop(layer_1_out))
        layer_3_out = model.layers[4].fprop(model.layers[3].fprop(layer_2_out))
        layer_4_out = model.layers[6].fprop(model.layers[5].fprop(layer_3_out))

        output_, cost_2 = get_conv_output(model, layer_4_out, encoder, encoder_b, decoder_b)
        preds_adv = model.layers[8].fprop(model.layers[7].fprop(output_))



        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, corrupt_prob, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, corrupt_prob, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc




    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    #adv_x = fgsm(x, preds_adv, eps=0.3)
    #eval_params = {'batch_size': FLAGS.batch_size}
    #X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
    #assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the CIFAR10 model on adversarial examples
    #accuracy = model_eval(sess, x, y, preds_adv, X_test_adv, Y_test,
    #                      args=eval_params)
    #print('Test accuracy on adversarial examples: ' + str(accuracy))

    without_denoising = True#False
    cost_2 = 0
    corrupt_prob_2 = tf.placeholder(tf.float32, [1])
    #fgsm2 = FastGradientMethod(model_2, sess=sess)
    #adv_x_2 = fgsm2.generate(x, **fgsm_params)
    model_2 = model_cnn()#img_rows=32, img_cols=32, channels=3)
    fgsm2 = FastGradientMethod(model_2, sess=sess)
    adv_x_2 = fgsm2.generate(x, **fgsm_params)


    if without_denoising == False:
        preds_2 = model_2(x)
        #if not backprop_through_attack:
        #    adv_x_2 = tf.stop_gradient(adv_x_2)
        preds_2_adv = model_2(adv_x_2)
    else:
        #encoder_2, encoder_b_2, decoder_b_2, corrupt_prob_2 = autoencoder(dimensions=[512, 300])

        layer_1_out = model_2.layers[1].fprop(model_2.layers[0].fprop(x))
        layer_2_out = model_2.layers[3].fprop(model_2.layers[2].fprop(layer_1_out))
        layer_3_out = model_2.layers[4].fprop(model_2.layers[3].fprop(layer_2_out))
        layer_4_out = model_2.layers[6].fprop(model_2.layers[5].fprop(layer_3_out))

        output_, cost_2 = get_conv_output(model_2, layer_4_out, encoder, encoder_b, decoder_b)
        preds_2 = model_2.layers[8].fprop(model_2.layers[7].fprop(output_))

        #if not backprop_through_attack:
        #    adv_x_2 = tf.stop_gradient(adv_x_2)
         #cost_2, preds_2_adv = get_output(model_2, adv_x_2, encoder_2, encoder_b_2, decoder_b_2)
        layer_1_out = model_2.layers[1].fprop(model_2.layers[0].fprop(adv_x_2))
        layer_2_out = model_2.layers[3].fprop(model_2.layers[2].fprop(layer_1_out))
        layer_3_out = model_2.layers[4].fprop(model_2.layers[3].fprop(layer_2_out))
        layer_4_out = model_2.layers[6].fprop(model_2.layers[5].fprop(layer_3_out))

        output_, cost_2 = get_conv_output(model_2, layer_4_out, encoder, encoder_b, decoder_b)
        preds_2_adv = model_2.layers[8].fprop(model_2.layers[7].fprop(output_))

    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, corrupt_prob_2, preds_2, X_test, Y_test,
                               args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

         # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, corrupt_prob_2, preds_2_adv, X_test,
                               Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy

     # Perform and evaluate adversarial training
    model_train(sess, x, y, corrupt_prob_2, preds_2, X_train, Y_train,
                 cost_2, predictions_adv=preds_2_adv, evaluate=evaluate_2,
                 args=train_params, rng=rng)

     # Calculate training errors
    if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, corrupt_prob_2, preds_2, X_train, Y_train,
                               args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, corrupt_prob_2, preds_2_adv, X_train,
                               Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracy

    return report



if __name__ == '__main__':
    app.run()
