
import keras
import numpy as np
from keras.utils import np_utils
import numpy.random as rng
import time

def data_svhn():
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets

    import scipy.io as sio
    train_file = svhn_file_train = "/u/lambalex/data/svhn/train_32x32.mat"
    extra_file = svhn_file_extra = '/u/lambalex/data/svhn/extra_32x32.mat'
    test_file = svhn_file_test = '/u/lambalex/data/svhn/test_32x32.mat'

    t0 = time.time()

    train_object = sio.loadmat(train_file)
    extra_object = sio.loadmat(extra_file)
    test_object = sio.loadmat(test_file)

    print(time.time()-t0, "time to load from file")

    t0 = time.time()

    train_X = np.asarray(train_object["X"], dtype = 'uint8')
    extra_X = np.asarray(extra_object["X"], dtype = 'uint8')
    test_X = np.asarray(test_object["X"], dtype = 'uint8')

    train_X = train_X.transpose(3,0,1,2)
    extra_X = extra_X.transpose(3,0,1,2)
    test_X = test_X.transpose(3,0,1,2)

    train_Y = np.asarray(train_object["y"], dtype = 'uint8')
    extra_Y = np.asarray(extra_object["y"], dtype = 'uint8')
    test_Y = np.asarray(test_object["y"], dtype = 'uint8')

    print(time.time()-t0, "time to pull into np and transpose")

    t0 = time.time()

    train_Y -= 1
    extra_Y -= 1
    test_Y -= 1

    print("trainx shape", train_X.shape, "extra X shape", extra_X.shape)

    all_train_X = np.vstack((train_X, extra_X))
    all_train_Y = np.vstack((train_Y, extra_Y))

    print("making perm")

    permutation = rng.permutation(all_train_X.shape[0])
    print("rand drawn")
    all_train_X = all_train_X[permutation]
    all_train_Y = all_train_Y[permutation]

    print(time.time()-t0, "time to sub, stack, permute")

    print("permuting done")

    X_train = all_train_X
    y_train = all_train_Y

    X_test = test_X
    y_test = test_Y

    t0 = time.time()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print(time.time()-t0, "time to cast to float32")
    print("casting done")
    t0 = time.time()
    X_train /= 255
    X_test /= 255

    #print(X_train.dtype)
    #print(X_train.max())

    print(time.time()-t0, "time to divide")
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    t0=time.time()
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print(time.time()-t0, "time to go to categorical")
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":

    data_svhn()


