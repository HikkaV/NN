from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers.core import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from Dataset import *
import keras
import Settings
import scipy
from scipy import ndimage

class NN(object):
    def __init__(self, img_size=50):
        """
        initialize train and test set and their labels
        :param img_size: the reshaped size of images
        """
        self.X_train, self.y_train, self.X_test, self.y_test = load_dataset()
        self.img_size = img_size

    def init_nn(self):
        """
        building the model using keras, making 5 conv2D layers with the same activation function = relu
        the same kernel size =3 , means the quantity of filters
        adding maxpooling with the matrix size 2x2 to boost the speed of fitting and minimize the chance of overfitting
        dense layer is a fully connected layer. dense_1 has  the same neurons as the image  pixels (50x50)
        the dense_2 layer is an output layer with 2 neurons related to car and cat
        in compile , using binary_crossentropy as we deal with 2 classes of images
        :return: returns a made model
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=(self.img_size, self.img_size, 3), activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(2500))
        model.add(BatchNormalization())
        model.add(Dropout(0.8))

        # softmax classifier
        model.add(Dense(2))
        model.add(Activation("softmax"))
        model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
        # return the constructed network architecture
        return model

    def fit_nn(self, epochs,  batch_size):
        """
        fitting model
        :param epochs: quantity of epochs
        :param batch_size: quantity of pics to be educated with per sec
        """
        model = self.init_nn()
        model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=epochs, verbose=1,
                  batch_size=batch_size)
        self.model = model

    def save_model(self):
        """save your model """
        self.model.save(Settings.path_to_model)

    def load_model(self):
        """load model if it exists"""
        if os.path.exists(Settings.path_to_model):
            model = keras.models.load_model(Settings.path_to_model)
            return model

    def predict(self, path, model):
        """
        reshapes your input image into the valid format to make prediction
        :param path: path to your pic
        :param model: saved model
        makes a prediction
        """
        if os.path.exists(path):
            image = np.array(ndimage.imread(path, flatten=False))
            my_image = scipy.misc.imresize(image, size=(self.img_size, self.img_size)).reshape((1, self.img_size , self.img_size , 3))
            tmp = model.predict_classes(my_image)
            if tmp[0]==1:
                print(str(tmp[0])+"  it's a cat")
            else:
                print(str(tmp[0])+"  it's a car")