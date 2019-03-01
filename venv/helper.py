from settings import *
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import transform
import re
import argparse
import nn
from nn import NN

ap = argparse.ArgumentParser()
neuraln = NN()


def parse_args():
    subparsers = ap.add_subparsers()
    fit_parser = subparsers.add_parser('fit_nn', help='fit your neural network')
    fit_parser.add_argument('-c', dest='condition', help='the argument which is responsible for saving the history of '
                                                         'training', required=True, type=bool)
    fit_parser.set_defaults(func=neuraln.fit_nn)

    predict_on_single_parser = subparsers.add_parser('predict_on_single_image',
                                                     help='get a prediction for a single img')
    predict_on_single_parser.add_argument('-p', dest='filename', required=True, type=str, help='path to pic for making '
                                                                                               ' a prediction ')
    predict_on_single_parser.add_argument('-w', dest='path',
                                          help='path to trained model', required=False, type=str,
                                          default='/home/hikkav/environments/my_env/IntestinesNNProjects/biasesAndWeights/model.2019-2-28_23-46.h5')
    predict_on_single_parser.set_defaults(func=neuraln.predict_on_single_image)
    evaluate = subparsers.add_parser('evaluate', help='evaluate your model using cross validation')
    evaluate.add_argument('-p', dest='path', help='path to trained model ', required=False, type=str,
                          default='/home/hikkav/environments/my_env/IntestinesNNProjects/biasesAndWeights/model.2019-2-28_23-46.h5')
    evaluate.set_defaults(func=neuraln.evaluate)
    return ap.parse_args()


def sorted_alphanumeric(data):
    """
    sorts a list in an alphanumeric order
    :param data: some list to sort

    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def make_single_pic_to_show(path):
    """
    converts an image into numpy array to show it using matplotlib lib
    :param path: path to a single img to convert

    """
    img = np.array(ndimage.imread(path, flatten=False))
    return img


def plot_single_pic(img, label):
    """
    shows a single pic with predicted class
    :param img: the converted img
    :param label: it's predicted class

    """
    ax = plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(img)
    plt.text(0.5, -0.1, label, horizontalalignment='center', verticalalignment='center',
             fontsize=15, transform=ax.transAxes)
    plt.show()


