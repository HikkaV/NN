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
import datetime


class NN(object):

    def __init__(self, epochs=n_epochs, eta=learning_rate, steps=batch, dropout=last_dropout, path=path_to_model):
        """
        initialize train and test set and their labels
        :param img_size: the reshaped size of images
        """
        self.eta = eta
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
        model.add(Conv2D(64, kernel_size, padding="same",
                         input_shape=(img_size, img_size, 3), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(64, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(256, kernel_size, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Dropout(0.3))
        model.add(Conv2D(256, kernel_size, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10000))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        model.add(Dense(n_classes, activation='softmax'))

        adam = keras.optimizers.Adam(lr=self.eta, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
        self.score = model.summary()
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
        callback = keras.callbacks.ModelCheckpoint(path_to_model, monitor='val_acc', verbose=1,
                                                   save_best_only=False,
                                                   save_weights_only=False, mode='max', period=1)
        callback_List = [callback]
        history = model.fit_generator(generator=train_generator, callbacks=callback_List, epochs=self.epochs, verbose=1,
                                      validation_data=test_generator,
                                      shuffle=True, steps_per_epoch=self.steps, initial_epoch=0,
                                      use_multiprocessing=True,
                                      workers=12)

        NN.save_history(self, history)

    def show_stats(self):
        stats = pd.read_json(self.history_path)
        print(stats)
        # arch = pd.read_json(self.arch_path)
        # print (arch)
    def save_model(self):
        """save your model """
        self.model.save(path_to_model)

    def save_history(self, history):
        now = datetime.datetime.now()
        additional_history = {'epochs': self.epochs,
                              'batch': batch,
                              'eta': self.eta,
                              'kernel_size': kernel_size[0]


                              }
        additional_history.update(history.history)
        self.history_path = history_path = path_to_history + ":" + str(now.year) + ":" + str(now.month) + ":" + str(now.hour) + ":" + str(now.minute) + ".json"
        with open(history_path, 'w') as f:
            json.dump(additional_history, f)
        # self.arch_path = path_to_acrh+":" + str(now.year) + ":" + str(now.month) + ":" + str(now.hour) + ":" + str(now.minute)+'.json'
        # with open (self.arch_path, 'w') as v:
        #      v.write(self.score)
    def load_model(self):
        """load model if it exists"""
        if os.path.exists(self.path_to_model):
            model = keras.models.load_model(self.path_to_model)
            return model

    def predict(self, path, model):
        """
        reshapes your input image into the valid format to make prediction
        :param path: path to your pic
        :param model: saved model
        makes a prediction
        """
        model = model
        if os.path.exists(path):
            image = np.array(ndimage.imread(path, flatten=False))
            my_image = scipy.misc.imresize(image, size=(img_size, img_size)).reshape(
                (1, img_size, img_size, 3))
            tmp = model.predict_classes(my_image)
            if tmp[0] == 1:
                print(str(tmp[0]) + "  it's a dog")
            elif tmp[0] == 0:
                print(str(tmp[0]) + "  it's a car")
            elif tmp[0] == 2:
                print(str(tmp[0]) + " it's a cat")

            plt.imshow(image)
            plt.show()
