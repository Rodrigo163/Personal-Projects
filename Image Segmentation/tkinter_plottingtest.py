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


def singlify(list_of_objects):
    for single_object in list_of_objects:
        #first part is to get labels and boxes on each single object
        labels, nobs = ndi.label(single_object)
        boxes = ndi.find_objects(labels)

        #if there is only one object then no need to modify the picture
        if nobs>1: #if we have two or more objects 
            #Then we have to compare their size and get the index of the biggest one
            obs = [np.array(single_object[boxes[i]]) for i in range(0,nobs)]
            obs_sizes = [ob.shape[0]*ob.shape[1] for ob in obs]
            n = int(np.where(obs_sizes == np.max(obs_sizes))[0])

            #now follows the mask that will filter everything but the biggest

            for i in range(0,nobs):
                if i != n:
                    single_object[boxes[i]].fill(0)
    return list_of_objects 

def backbone(objects):
    binaries = [np.where(ob > 0, 1, 0) for ob in objects]
    backbones = [skel(binary) for binary in binaries]
    binary_backbones = [np.where(ob ==True, 1, 0) for ob in backbones]
    return binary_backbones

def greyoverlap(singles, backbones):
    greyoverlap = [np.where(backbones[i] == 1, -80, 0) + singles[i] for i in range(0, len(singles))]
    return greyoverlap

def coordinates(backboneimage):
    #the idea is to get first an arrays like [(x1,y1), (x2, y2), ...] from the boolean or binary output of the backbone function
    locations = np.where(test_output==True)
    coords = [[locations[0][i], locations[1][i]] for i in range(0, len(locations[0]))]
    
    #now to match JFils format
    f = open("snake.txt", "w+")
    f.write("#\r")
    f.write("0\r")
    for i in range(0, len(coords)):
        f.write("1\t"+ str(i) + "\t" + "%d\t" % (coords[i][0]) + "%d\t" % (coords[i][1]) + "0\r")
    f.close()

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


def analyse():
    #label, box, singlify, backbone and greyoverlap all in one
    labels, nobjects = ndi.label(multi_final)
    boxes = ndi.find_objects(labels)
    objects = [np.array(multi_final[boxes[i]]) for i in range(0,nobjects-1) if (np.array(multi_final[boxes[i]]).shape[0]*np.array(multi_final[boxes[i]]).shape[1] > 600)]
    singles =  singlify(objects)
    backbones = backbone(singles)
    global finals
    finals = greyoverlap(singles, backbones)
    
    #plotting
    global track
    track =0 
    global fig
    fig = Figure(figsize=(5, 4), dpi=100)
    a = fig.add_subplot(111, frameon=False)
    a.imshow(finals[track])
    canvas = FigureCanvasTkAgg(fig, master=figures_menu)
    canvas.draw()
    canvas.get_tk_widget().grid(row=track, column=0)
    canvas.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Snake", command=save_figure)
    save_figure_button.grid(row=track, column=1)
    #skip_figure_button = tk.Button(figures_menu, text="Skip Snake", command = skip_figure)
    #skip_figure_button.grid(row=track, column=2)
    
tracking = 0

def update_track():
    global tracking
    tracking = tracking +1
    track_button = tk.Button(figures_menu, text="track : "+ str(tracking), command=update_track)
    track_button.grid(row=4, column=0)
    
    

def open_file():
    #users selects a file and the image is displayed
    global file1
    file1 = fd.askopenfilename()
    global im
    im = ImageTk.PhotoImage(file = file1)
    image_label= tk.Label(master=root, image =im)
    image_label.image= im
    image_label.grid(row=1, column=1)

def save_figure():
    labels, nobjects = ndi.label(multi_final)
    boxes = ndi.find_objects(labels)
    objects = [np.array(multi_final[boxes[i]]) for i in range(0,nobjects-1) if (np.array(multi_final[boxes[i]]).shape[0]*np.array(multi_final[boxes[i]]).shape[1] > 600)]
    singles =  singlify(objects)
    backbones = backbone(singles)
    global finals
    finals = greyoverlap(singles, backbones)
    
    #getting fig
    global fig
    fig = Figure(figsize=(5, 4), dpi=100)
    a = fig.add_subplot(111, frameon=False)
    a.imshow(finals[track])
    
    file2save=fig
    file2save = fd.asksaveasfile(mode='w', defaultextension='.png')
    file2save_2 = file2save.name
    fig.savefig(str(file2save_2)) 
    
    #updating fig 
    track=track+1
    fig = Figure(figsize=(5, 4), dpi=100)
    a = fig.add_subplot(111, frameon=False)
    a.imshow(finals[track])
    canvas = FigureCanvasTkAgg(fig, master=figures_menu)
    canvas.draw()
    canvas.get_tk_widget().grid(row=track, column=0)
    canvas.mpl_connect("key_press_event", on_key_press)
    
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
    global multi_final
    multi_final = mask_median
    
    global fig
    fig = Figure(figsize=(5, 4), dpi=100)
    a = fig.add_subplot(111, frameon=False)
    a.imshow(multi_final)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=2)
    canvas.mpl_connect("key_press_event", on_key_press)


    
    
    
root = tk.Tk()
root.wm_title("Embedding in Tk")
root.minsize(500,500)

file_preproc_menu = tk.Frame(root)
file_preproc_menu.grid(row = 0, column=0)

figures_menu = tk.Frame(root)
figures_menu.grid(row=0, column=1)

openfile_button= tk.Button(file_preproc_menu, text="Open file", command=open_file)
openfile_button.grid(row=0, column=0)

prepro_button= tk.Button(file_preproc_menu, text="preprocess image", command=pre_processing_dark)
prepro_button.grid(row=1, column=0)

start_analysis_button = tk.Button(file_preproc_menu, text="Start Analysis", command=analyse)
start_analysis_button.grid(row=2, column=0)
    
track_button = tk.Button(figures_menu, text="track : 0", command=update_track)
track_button.grid(row=4, column=0)

    
def on_key_press(event):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas)



def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


quit_button = tk.Button(file_preproc_menu, text="Quit", command=_quit)
quit_button.grid(row=3, column=0)

tk.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.