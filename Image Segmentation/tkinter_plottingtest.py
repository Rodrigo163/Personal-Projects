import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog as fd
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage.morphology import skeletonize as skel 
from skimage.morphology import disk
import imageio as iio
import scipy.ndimage as ndi
import math




def show_plot():
    fig = Figure(figsize=(5, 4), dpi=100)
    t = np.arange(0, 3, .01)
    fig.add_subplot(121).plot(t, 2 * np.sin(2 * np.pi * t))
    fig.add_subplot(122).plot(t, 6 * np.sqrt(2 * np.pi * t))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=1)
    canvas.mpl_connect("key_press_event", on_key_press)

def show_image():
    image = Image.open("welcome.jpg")
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(image=photo,width=400, height=400)
    label.image = photo # keep a reference!
    label.grid(row=1, column=1)
    
def open_file():
    #users selects a file and the image is displayed
    global file1
    file1 = fd.askopenfilename()
    global im
    im = ImageTk.PhotoImage(file = file1)
    image_label= tk.Label(master=root, image =im)
    image_label.image= im
    image_label.grid(row=1, column=1)
    
def pre_processing_dark():    
    #thresholding, masking, filtering
    global multi
    multi = file1
    multi = iio.imread(file1)
    multi = color.rgb2gray(multi)
    masked_multi = np.where(multi < np.percentile(multi, 99),0 , multi)
    mask_dilate = ndi.binary_dilation(masked_multi, iterations=1)
    mask_erosion = ndi.binary_erosion(mask_dilate, iterations=2)
    mask_closed = ndi.binary_closing(mask_erosion, iterations=3)
    mask_median = filters.median(mask_closed, disk(2.5))
    multi_final = mask_median
    
    fig = Figure(figsize=(5, 4), dpi=100)
    fig.add_subplot(111).imshow(multi_final)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=2)
    canvas.mpl_connect("key_press_event", on_key_press)
    
root = tk.Tk()
root.wm_title("Embedding in Tk")
root.minsize(500,500)
openfile_button= tk.Button(root, text="Open file", command=open_file)
openfile_button.grid(row=0, column=0)

plot2_button= tk.Button(root, text="change plot", command=show_diff_plot)
plot2_button.grid(row=2, column=0)

image_button= tk.Button(root, text="display image", command=show_image)
image_button.grid(row=3, column=0)

prepro_button= tk.Button(root, text="preprocess image", command=pre_processing_dark)
prepro_button.grid(row=4, column=0)

def on_key_press(event):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas)



def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


quit_button = tk.Button(master=root, text="Quit", command=_quit)
quit_button.grid(row=5, column=0)

tk.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.