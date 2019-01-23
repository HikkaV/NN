import numpy as np
import os
import h5py
import Settings
import keras


def load_dataset():
    train_dataset = h5py.File(Settings.adress, "r")
    train_set_x_orig = np.array(train_dataset["train_img"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_labels"][:])  # your train set labels

    test_dataset = h5py.File(Settings.adress, "r")
    test_set_x_orig = np.array(test_dataset["test_img"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_labels"][:])  # your test set labels

    train_set_x_orig = train_set_x_orig.reshape(1298, 50, 50, 3)
    test_set_x_orig = test_set_x_orig.reshape(433, 50, 50, 3)
    train_set_y_orig = keras.utils.to_categorical(train_set_y_orig, 2)
    test_set_y_orig = keras.utils.to_categorical(test_set_y_orig, 2)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig
