from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers.core import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import keras
from Settings import *
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from Generators import Generator
import os
import json
import pandas as pd


class NN(object):

    def __init__(self, epochs=n_epochs, steps=batch, dropout=last_dropout, path=path_to_model):
        """
        initialize train and test set and their labels
        :param img_size: the reshaped size of images
        """
        self.dropout = dropout
        self.path_to_model = path
        self.steps = steps
        self.epochs = epochs
        self.img_size = img_size

    def init_nn(self):
        """
        building the model using keras, making 6 conv2D layers with the same activation function = relu
        the same kernel size =3 , means the quantity of filters
        adding maxpooling with the matrix size 2x2 to boost the speed of fitting and minimize the chance of overfitting
        dense layer is a fully connected layer. dense_1 has  the same neurons as the image  pixels (50x50)
        the dense_2 layer is an output layer with 2 neurons related to car and cat
        in compile , using binary_crossentropy as we deal with 2 classes of images
        :return: returns a made model
        """

        model = Sequential()
        model.add(Conv2D(32, kernel_size, padding="same",
                         input_shape=(self.img_size, self.img_size, 3), activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(64, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(2500))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        # softmax classifier
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
        # return the constructed network architecture
        return model

    def fit_nn(self):
        """
        fitting model
        :param epochs: quantity of epochs
        :param batch_size: quantity of pics to be educated with per sec
        """

        model = self.init_nn()
        train_generator = Generator(path_to_train)
        test_generator = Generator(path_to_test, abs_path=path_to_test)
        callback = keras.callbacks.ModelCheckpoint(self.path_to_model, monitor='val_loss', verbose=1,
                                                   save_best_only=False,
                                                   save_weights_only=False, mode='min', period=1)
        callback_List = [callback]
        history = model.fit_generator(generator=train_generator, callbacks=callback_List, epochs=self.epochs, verbose=1,
                                      validation_data=test_generator,
                                      shuffle=True, steps_per_epoch=self.steps, initial_epoch=0,
                                      use_multiprocessing=True,
                                      workers=12)

        self.save_history(history)

    def show_stats(self):
        stats = pd.read_json(path_to_history)
        print(stats)

    def save_model(self):
        """save your model """
        self.model.save(path_to_model)

    def save_history(self, history):

        with open(path_to_history, 'w') as f:
            json.dump(history.history, f)

    def load_model(self):
        """load model if it exists"""
        if os.path.exists(self.path_to_model):
            model = keras.models.load_model(self.path_to_model)
            return model

    def predict(self, path):
        """
        reshapes your input image into the valid format to make prediction
        :param path: path to your pic
        :param model: saved model
        makes a prediction
        """
        model = self.load_model()
        if os.path.exists(path):
            image = np.array(ndimage.imread(path, flatten=False))
            my_image = scipy.misc.imresize(image, size=(img_size, img_size)).reshape(
                (1, img_size, img_size, 3))
            tmp = model.predict_classes(my_image)
            if tmp[0] == 1:
                print(str(tmp[0]) + "  it's a cat")
            else:
                print(str(tmp[0]) + "  it's a car")
            plt.imshow(image)
            plt.show()
