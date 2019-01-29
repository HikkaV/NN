import numpy as np
import keras
import scipy
from scipy import ndimage
from Settings import *
import os
class Generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, abs_path=path_to_train,  batch_size=batch, dim=(50,50), n_channels=n_channels,
                 n_classes=n_classes, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = os.listdir(list_IDs)
        self.abs_path = abs_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def load_labels(self, path):
        """used to load labels"""
        if path.__contains__('car'):
            label = 0
        else:
            label=1
        return label

    def preprocess_input(self, path):
        """preprocessing input images from path
        resizing each image to a 4dim vector with params (1, 50, 50, 3)
        """
        image = np.array(ndimage.imread(self.abs_path+'/'+path, flatten=False))
        my_image = scipy.misc.imresize(image, size=self.dim).reshape(
            (1, 50, 50, 3))
        return my_image

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples""" # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, path in enumerate(list_IDs_temp):
            # Store sample
            X[i,]=self.preprocess_input(path)
            # Store class
            y[i] = self.load_labels(path)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)