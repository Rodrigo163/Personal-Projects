# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:03:52 2019

@author: pwmadmin
"""

from tkinter import *
import tkinter as tk
from tkinter import filedialog as fd
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from PIL import ImageTk, Image
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage.morphology import skeletonize as skel 
from skimage.morphology import disk
import imageio as iio
import scipy.ndimage as ndi


def plot(plt_x, plt_y):
    plt.figure()
    plt.plot(plt_x, plt_y,'k')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("x")
    plt.ylabel("y'")
    plt.legend(loc="upper right")
    plt.show()

def im_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def pre_processing_dark(multi):    
    #thresholding, masking, filtering
    masked_multi = np.where(multi < np.percentile(multi, 99),0 , multi)
    mask_dilate = ndi.binary_dilation(masked_multi, iterations=1)
    mask_erosion = ndi.binary_erosion(mask_dilate, iterations=2)
    mask_closed = ndi.binary_closing(mask_erosion, iterations=3)
    mask_median = filters.median(mask_closed, disk(2.5))
    multi_final = mask_median
    
    return multi_final

def singlify(list_of_objects, n):
    for object in list_of_objects:
        #first part is to get labels and boxes 
        labels, nobs = ndi.label(object)
        boxes = ndi.find_objects(labels)

        #if there is only one object then no need to modify the picture
        if len(boxes)>1:
            #Then we have to compare their size and get the index of the biggest one
            obs = [np.array(object[boxes[i]]) for i in range(0,nobs-1)]
            obs_sizes = [ob.shape[0]*ob.shape[1] for ob in obs]
            n = int(np.where(obs_sizes == np.max(obs_sizes))[0])

            #now follows the mask that will filter everything but the biggest

            for i in range(0,n-1):
                if i != n:
                    object[boxes[i]].fill(0)
    return list_of_objects 

def backbone(objects):
    binaries = [np.where(ob > 0, 1, 0) for ob in objects]
    backbones = [skel(binary) for binary in binaries]
    binary_backbones = [np.where(ob ==True, 1, 0) for ob in backbones]
    return binary_backbones

def greyoverlap(singles, backbones):
    greyoverlap = [np.where(backbones[i] == 1, -80, 0) + singles[i] for i in range(0, len(singles))]
    return greyoverlap

def Run():
    file1 = fd.askopenfilename()
    global image_tinker
    image_tinker = ImageTk.PhotoImage(file = file1)
    label= Label(master=a, image =image_tinker)
    label.image= image_tinker
   # label.pack(side=TOP)
    
    global im
    im = iio.imread(file1)
    im = color.rgb2gray(im) 
    im = pre_processing_dark(im)
    #global labels
    #global nobjects
    labels, nobjects = ndi.label(im)
    #global boxes
    boxes = ndi.find_objects(labels)
    #global objects
    objects = [np.array(im[boxes[i]]) for i in range(0, nobjects-1) if (np.array(im[boxes[i]]).shape[0]*np.array(im[boxes[i]]).shape[1] > 600)]
    #global singles
    singles = singlify(objects, nobjects)
    #global backbones
    backbones = backbone(singles)
    #global overlaps
    overlaps = greyoverlap(singles, backbones)
    #global n_images
    n_images = len(backbones)
    #global fig
    fig = Figure(figsize = (9, 6))
    for i in range(0,n_images-1):
        fig.add_subplot(i+1, 1, i+1).imshow(overlaps[i])
    #fig.add_subplot(111).imshow(backbones[0])
    canvas = FigureCanvasTkAgg(fig, master=a)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack()
       
    
    #checks
    #texto = backbones[0]
    #Label(master=a, text= texto).pack(side=BOTTOM)
    
#--------------------------------------------------------- 
# Define geometry and GUI properties
a = Tk()
a.title("Filament analysis")
a.geometry("640x960+0+0")

def save_figure():
    file2save = fig  
    file2save = fd.asksaveasfile(mode='w', defaultextension='.png')
    file2save_2 = file2save.name
    fig.savefig(str(file2save_2)) 
    
def _quit():
    a.quit()     # stops mainloop
    a.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
run = Button(a, text="Import Image and analyse", width=20, height=1, bg="lightblue", command=Run).place(x=250, y=380)
exit_button = tk.Button(master=a, text='Quit', command=_quit)
exit_button.pack(side=tk.BOTTOM)
saveButton = Button(a, text="Export figure", width=20, height=1, bg="lightblue", command=save_figure).pack(side=tk.BOTTOM)

#-----------------------------------------------------------------------------------------------------------------------
# Headers
heading = Label(a, text="Thesis stuff", font=("arial", 30, "bold"), fg="steelblue").pack() 


a.mainloop()
