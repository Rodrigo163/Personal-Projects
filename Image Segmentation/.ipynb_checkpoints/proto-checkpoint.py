import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import imageio as iio
import scipy.ndimage as ndi
def plotim(im):
    plt.imshow(im)
    plt.axis('off')