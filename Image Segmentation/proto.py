#prototyping code for thesis
#Current problem: find/estimate middle of filament given grid with pixel intensities
#as first test we'll use a masked single filament
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import imageio as iio
import scipy.ndimage as ndi
# %%

im = iio.imread('masked_single_fil.png')
plt.imshow(im)
