from Settings import *
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def make_pics_to_show():
    z = os.listdir(path_to_predict)[0]
    tmp = os.listdir(path_to_predict + '/' + z)
    tmp.sort()
    list_images = [np.array(ndimage.imread(path_to_predict + '/' + z + '/' + b, flatten=False)) for b in tmp]
    return list_images


def plot_image(images, labels):
    for i in range(0, len(labels)):
        ax = plt.subplot(1, 1, 1)
        plt.axis('off')
        plt.imshow(images[i])
        plt.text(0.5, -0.1, labels[i], horizontalalignment='center', verticalalignment='center',
                 fontsize=15, transform=ax.transAxes)
        plt.show()
