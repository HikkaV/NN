from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers.core import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import keras
from settings import *
import os
import json
import pandas as pd
import datetime
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import transform
import helper


class NN(object):

    def __init__(self, epochs=n_epochs, train_batches=batch, val_batch=val_steps, crossval_batches=crossval_batch,
                 eta=learning_rate, dropout=last_dropout,
                 path=path_to_model):
        """
        initialize all necessary params

        """
        self.train_batch = train_batches
        self.val__batch = val_batch
        self.datagen = ImageDataGenerator(rescale=1. / 255)
        self.validation_generator = self.datagen.flow_from_directory(directory=path_to_test,
                                                                     target_size=dim, color_mode='rgb',
                                                                     batch_size=self.val__batch,
                                                                     class_mode='categorical',
                                                                     shuffle=True,
                                                                     seed=random_state)
        self.train_generator = self.datagen.flow_from_directory(directory=path_to_train,
                                                                target_size=dim, color_mode='rgb',
                                                                batch_size=self.train_batch,
                                                                class_mode='categorical',
                                                                shuffle=True,
                                                                seed=random_state)
        self.crossval_generator = self.datagen.flow_from_directory(
            directory=path_to_crossval,
            target_size=(img_size, img_size),
            color_mode="rgb",
            batch_size=crossval_batches,
            class_mode='categorical',
            shuffle=True,
            seed=random_state
        )
        self.test_generator = self.datagen.flow_from_directory(
            directory=path_to_predict,
            target_size=(img_size, img_size),
            color_mode="rgb",
            batch_size=1,
            class_mode=None,
            shuffle=False,
            seed=random_state
        )
        self.classes = self.validation_generator.class_indices
        self.classes = dict((v, k) for k, v in self.classes.items())
        self.save_dict()
        self.eta = eta
        self.dropout = dropout
        self.path_to_model = path
        self.epochs = epochs
        self.img_size = img_size
        self.now = datetime.datetime.now()
        self.abs_model_path = None
        self.model_arch = None
        self.history_path = None
        self.date = "." + str(self.now.year) + "-" + str(self.now.month) + "-" + str(self.now.day) + "_" + str(
            self.now.hour) + '-' + str(self.now.minute)

    def init_nn(self):
        """
        building the model using keras, making 6 conv2D layers with the same activation function = relu
        the same kernel size =3 , means the quantity of filters
        adding maxpooling with the matrix size 2x2 to boost the speed of fitting and minimize the chance of overfitting
        dense layer is a fully connected layer. dense_1 has  the same neurons as the image  pixels (100x100)
        the dense_2 layer is an output layer with 3 neurons related to car , cat and dog
        in compile , using binary_crossentropy as we deal with 3 classes of images
        :return: returns a made model
        """

        model = Sequential()
        model.add(Conv2D(32, kernel_size, padding="same",
                         input_shape=(img_size, img_size, 3), activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(32, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(32, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(64, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(64, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(64, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(64, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size, padding="same", activation='relu'))
        model.add(BatchNormalization(axis=3))
        model.add(Conv2D(256, kernel_size, padding="same", activation='relu'))
        model.add(Conv2D(256, kernel_size, padding="same", activation='relu'))
        model.add(Conv2D(256, kernel_size, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, kernel_size, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, kernel_size, padding="same", activation='relu'))
        model.add(Conv2D(512, kernel_size, padding="same", activation='relu'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(dense_layer))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        model.add(Dense(n_classes, activation='softmax'))

        adam = keras.optimizers.Adam(lr=self.eta, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
        # return the constructed network architecture
        return model

    def fit_nn(self, args):

        """
        fitting model

        """

        model = self.init_nn()

        self.abs_model_path = path_to_model + self.date + ".h5"
        callback = keras.callbacks.ModelCheckpoint(self.abs_model_path, monitor='val_acc', verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=False, mode='max', period=1)
        callback_List = [callback]
        history = model.fit_generator(generator=self.train_generator, callbacks=callback_List, epochs=self.epochs,
                                      verbose=1,
                                      validation_data=self.validation_generator,
                                      validation_steps=self.val__batch,
                                      shuffle=True, steps_per_epoch=self.train_batch, initial_epoch=0,
                                      use_multiprocessing=True,
                                      workers=12)

        NN.save_history(self, history, model)
        if args.condition:
            self.show_stats()

    def evaluate(self, args):
        model = self.load_model(args.path)
        score = model.evaluate_generator(self.crossval_generator, steps=crossval_batch, max_queue_size=10, workers=10,
                                         use_multiprocessing=False)
        print ('acc : ' + str(score[1]))

    def show_stats(self):
        """
        prints results of training per epoch
        from json file using pandas lib

        """
        stats = pd.read_json(self.history_path)
        print(stats)

    def save_model(self):
        """save your model """

        self.model.save_weights(self.abs_model_path)
    def save_history(self, history, model):
        """
        saves history of training and other features to json file
        :param history: history of training
        :param model: saved model
        :return:
        """
        data = model.to_json()
        self.model_arch = 'model' + self.date + ".json"
        with open(self.model_arch, 'w') as z:
            z.write(data)
        additional_history = {'epochs': self.epochs,
                              'batch': batch,
                              'eta': self.eta,
                              'kernel_size': kernel_size[0]

                              }
        additional_history.update(history.history)
        self.history_path = history_path = path_to_history + self.date + ".json"
        with open(history_path, 'w') as f:
            json.dump(additional_history, f)

    def save_dict(self):
        """

        saves a dict with a labels to json file
        """
        if not os.path.exists(path_to_labels):
            with open(path_to_labels, 'w') as f:
                json.dump(self.classes, f)

    def load_model(self, path):
        """load model if it exists"""
        if self.abs_model_path is not None:
            path = self.abs_model_path
        if os.path.exists(path):
            model = keras.models.load_model(path)
            return model

    def predict_on_single_image(self, args):

        '''
        predicts a class of a single input image
        :param args.filename: path to the pic to predict
        :param model: saved model

        '''
        model = self.load_model(args.path)
        np_image = Image.open(args.filename)
        np_image = np.array(np_image).astype('float32') / 255
        np_image = transform.resize(np_image, (img_size, img_size, n_classes))
        np_image = np.expand_dims(np_image, axis=0)
        tmp = model.predict(np_image)
        print(tmp)
        prediction = np.argmax(tmp, axis=1)
        pred = self.classes[prediction[0]]
        helper.plot_single_pic(helper.make_single_pic_to_show(args.filename), pred)

    def predict_pics(self, args):

        """
        reshapes your input image into the valid format to make prediction
        :param model: saved model
        makes a prediction on a list of files in some folder
        """
        model = self.load_model(args.path)
        self.test_generator.reset()
        pred = model.predict_generator(self.test_generator, verbose=1)
        predicted_class_indices = np.argmax(pred, axis=1)
        predictions = [self.classes[k] for k in predicted_class_indices]
        helper.plot_images(helper.make_pics_to_show(), predictions)
