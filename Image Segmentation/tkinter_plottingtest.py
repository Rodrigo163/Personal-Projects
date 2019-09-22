import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import matplotlib
import os
from PIL import Image, ImageTk
from tkinter import filedialog as fd
import skimage.data as data
import skimage.segmentation as seg
from sklearn.preprocessing import normalize
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage.morphology import skeletonize as skel 
from skimage.morphology import disk
import imageio as iio
import scipy.ndimage as ndi
from scipy import interpolate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import math

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

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

track =0
finals = np.array([])
backbones = np.array([])
fig = Figure(figsize=(5, 4), dpi=100)
figpix = Figure(figsize=(3, 3), dpi=100)
figpoly = Figure(figsize=(3.5, 3.5), dpi=100)
figspl = Figure(figsize=(3.5, 3.5), dpi=100)
nfilaments=0
x = np.array([])
y = np.array([])
xpoly = np.array([])
ypoly = np.array([])
xspl= np.array([])
yspl= np.array([])
coordspix = np.array([])
coordspoly = np.array([])
coordsspl = np.array([])
locations = np.array([])

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
    im = Image.open(file1)
    im = im.resize((400,400), Image.ANTIALIAS)
    name = file1.split("/")[-1].split(".")[0]+'.PNG'
    im.save(name)
    im = ImageTk.PhotoImage(file = name)
    image_label = tk.Label(master=file_preproc_menu, image =im)
    image_label.image= im #keeping reference
    image_label.grid(row=0, column=1)

    
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
    
    canvas = FigureCanvasTkAgg(fig, master=file_preproc_menu)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=1)
    canvas.mpl_connect("key_press_event", on_key_press)

def pre_processing_general():    
    #thresholding, masking, filtering
    global multi
    multi = file1
    multi = iio.imread(file1)
    multi = color.rgb2gray(multi)
    multi = normalize(multi, norm='l1')
    multi = np.where(multi < np.percentile(multi, 95),0 , multi)
    multi = ndi.binary_dilation(multi, iterations=2)
    multi = ndi.binary_erosion(multi, iterations=1)
    multi = ndi.binary_opening(multi, iterations=1)
    
    global multi_final
    multi_final = multi
    
    global fig
    fig = Figure(figsize=(5, 4), dpi=100)
    a = fig.add_subplot(111, frameon=False)
    a.imshow(multi_final)
    
    canvas = FigureCanvasTkAgg(fig, master=file_preproc_menu)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=1)
    canvas.mpl_connect("key_press_event", on_key_press)


def skip_filament():
    global track
    global finals
    global fig
    global nfilaments
    global backbones
    global figpix
    global figpoly
    global figspl
    global x
    global y
    global xploy
    global ypoly
    global xspl
    global yspl
    global coordspix
    global coordspoly
    global coordsspl
    global locations
    #====================================================================================
    #UPDATE TRACKERS AND COORDSPIX
    track=track+1
    nfilaments = nfilaments -1
    locations = np.where(backbones[track]==True)
    coordspix = [np.array([locations[0][i], locations[1][i]]) for i in range(0, len(locations[0]))]
    ordered = sorted(coordspix, key=lambda k: [k[1], k[0]])
    ordered = [np.flip(x) for x in ordered]
    coordspix = np.copy(ordered)
    #====================================================================================
    #GETTING new COORDSPOLY
    xpoly = [coordspix[i][0] for i in range(0, len(coordspix))]
    ypoly = [coordspix[i][1] for i in range(0, len(coordspix))]
    #poly coeffs
    z = np.polyfit(xpoly,ypoly,10)
    #poly function
    p = np.poly1d(z)
    ypolyfinal = p(xpoly)
    #====================================================================================
    #GETTING new COORDSSPL
    
    xspl = [coordspix[i][0] for i in range(0, len(coordspix))]
    yspl = [coordspix[i][1] for i in range(0, len(coordspix))]
    coordsspl = np.copy(coordspix)
    n_chunks = 5
    x_chunks = chunkIt(xspl, n_chunks)
    y_chunks = chunkIt(yspl, n_chunks)
    ysplfinal = []
    
    for i in range(0,n_chunks):
        tck = interpolate.splrep(x_chunks[i], y_chunks[i], s=13)
        ynew = interpolate.splev(x_chunks[i], tck, der=0)
        ysplfinal.extend(ynew)
    
    #updating output coords
    for i in range(0, len(coordsspl)):
        if math.isnan(i):
            coordsspl[i].astype(float)
            coordsspl[i][1] = ysplfinal[i]
    xspl = [coordsspl[i][0] for i in range(0, len(coordsspl))]
    yspl = [coordsspl[i][1] for i in range(0, len(coordsspl))]

    #====================================================================================
    #UPDATE PLOTS

    #update figpix
    apix = figpix.add_subplot(111, frameon=False)
    apix.imshow(finals[track])
    canvaspix = FigureCanvasTkAgg(figpix, master=figures_menu)
    canvaspix.draw()
    canvaspix.get_tk_widget().grid(row=0, column=0)
    canvaspix.mpl_connect("key_press_event", on_key_press)
    counter_button = tk.Button(figures_menu, text=str(nfilaments)+" remaining filaments")
    counter_button.grid(row=0, column=3)
    
    #update figpoly
    figpoly = Figure(figsize=(3, 3), dpi=100)
    apoly = figpoly.add_subplot(111, frameon=False)
    apoly.plot(xpoly, ypoly, 'ro', xpoly, ypolyfinal, 'b', markersize=1)
    canvaspoly = FigureCanvasTkAgg(figpoly, master=figures_menu)
    canvaspoly.draw()
    canvaspoly.get_tk_widget().grid(row=1, column=0)
    canvaspoly.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Polynomial Snake", command=save_polyfigure)
    save_figure_button.grid(row=1, column=1)
    
    #update figspl
    figspl = Figure(figsize=(3, 3), dpi=100)
    aspl = figspl.add_subplot(111, frameon=False)
    aspl.plot(xspl, yspl, 'ro', xspl, ysplfinal, 'b', markersize=1)
    canvasspl = FigureCanvasTkAgg(figspl, master=figures_menu)
    canvasspl.draw()
    canvasspl.get_tk_widget().grid(row=2, column=0)
    canvasspl.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Spline Snake", command=save_polyfigure)
    save_figure_button.grid(row=2, column=1)
    
def save_pixfigure():
    global track
    global finals
    global fig
    global nfilaments
    global backbones
    global figpix
    global figpoly
    global figspl
    global x
    global y
    global xploy
    global ypoly
    global xspl
    global yspl
    global coordspix
    global coordspoly
    global coordsspl
    global locations
    
    #====================================================================================
    #GETTING PIX OUTPUT
    coordsoutput = np.copy(coordspix)
    #writting pix file
    s = open("snake"+str(track)+".txt", "w+")
    s.write("#\rX\rX\rX\rX\rX\rX\rX\rX\rX\r0\r")
    for i in range(0, len(coordsoutput)):
        s.write("1\t"+ str(i) + "\t" + "%5.2f \t" % (coordsoutput[i][0]) + "%5.3f \t" % float((coordsoutput[i][1])) + "0\r")
    s.close()
    #empty elongation file, necessary to run Matlab script
    e = open("elongation"+str(track)+".txt", "w+")
    e.write("#\r#\r#\r#\r#\r#\r#\r#\r#\r\r\r#Snake Data\r#\r1 0")
    e.close()
    #====================================================================================
    #UPDATE TRACKERS AND COORDSPIX
    track=track+1
    nfilaments = nfilaments -1
    locations = np.where(backbones[track]==True)
    coordspix = [np.array([locations[0][i], locations[1][i]]) for i in range(0, len(locations[0]))]
    ordered = sorted(coordspix, key=lambda k: [k[1], k[0]])
    ordered = [np.flip(x) for x in ordered]
    coordspix = np.copy(ordered)
    #====================================================================================
    #GETTING new COORDSPOLY
    xpoly = [coordspix[i][0] for i in range(0, len(coordspix))]
    ypoly = [coordspix[i][1] for i in range(0, len(coordspix))]
    #poly coeffs
    z = np.polyfit(xpoly,ypoly,10)
    #poly function
    p = np.poly1d(z)
    ypolyfinal = p(xpoly)
    #====================================================================================
    #GETTING new COORDSSPL
    
    xspl = [coordspix[i][0] for i in range(0, len(coordspix))]
    yspl = [coordspix[i][1] for i in range(0, len(coordspix))]
    coordsspl = np.copy(coordspix)
    n_chunks = 5
    x_chunks = chunkIt(xspl, n_chunks)
    y_chunks = chunkIt(yspl, n_chunks)
    ysplfinal = []
    
    for i in range(0,n_chunks):
        tck = interpolate.splrep(x_chunks[i], y_chunks[i], s=13)
        ynew = interpolate.splev(x_chunks[i], tck, der=0)
        ysplfinal.extend(ynew)
    
    #updating output coords
    for i in range(0, len(coordsspl)):
        if math.isnan(i):
            coordsspl[i].astype(float)
            coordsspl[i][1] = ysplfinal[i]
    xspl = [coordsspl[i][0] for i in range(0, len(coordsspl))]
    yspl = [coordsspl[i][1] for i in range(0, len(coordsspl))]

    #====================================================================================
    #UPDATE PLOTS

    #update figpix
    apix = figpix.add_subplot(111, frameon=False)
    apix.imshow(finals[track])
    canvaspix = FigureCanvasTkAgg(figpix, master=figures_menu)
    canvaspix.draw()
    canvaspix.get_tk_widget().grid(row=0, column=0)
    canvaspix.mpl_connect("key_press_event", on_key_press)
    counter_button = tk.Button(figures_menu, text=str(nfilaments)+" remaining filaments")
    counter_button.grid(row=0, column=3)
    
    #update figpoly
    figpoly = Figure(figsize=(3, 3), dpi=100)
    apoly = figpoly.add_subplot(111, frameon=False)
    apoly.plot(xpoly, ypoly, 'ro', xpoly, ypolyfinal, 'b', markersize=1)
    canvaspoly = FigureCanvasTkAgg(figpoly, master=figures_menu)
    canvaspoly.draw()
    canvaspoly.get_tk_widget().grid(row=1, column=0)
    canvaspoly.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Polynomial Snake", command=save_polyfigure)
    save_figure_button.grid(row=1, column=1)
    
    #update figspl
    figspl = Figure(figsize=(3, 3), dpi=100)
    aspl = figspl.add_subplot(111, frameon=False)
    aspl.plot(xspl, yspl, 'ro', xspl, ysplfinal, 'b', markersize=1)
    canvasspl = FigureCanvasTkAgg(figspl, master=figures_menu)
    canvasspl.draw()
    canvasspl.get_tk_widget().grid(row=2, column=0)
    canvasspl.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Spline Snake", command=save_polyfigure)
    save_figure_button.grid(row=2, column=1)
    
def save_polyfigure():
    global track
    global finals
    global fig
    global nfilaments
    global backbones
    global figpix
    global figpoly
    global figspl
    global x
    global y
    global xploy
    global ypoly
    global xspl
    global yspl
    global coordspix
    global coordspoly
    global coordsspl
    global locations
    
    #====================================================================================
    #GETTING POLY OUTPUT
    coordsoutput = np.copy(coordspoly)
    #writting pix file
    s = open("snake"+str(track)+".txt", "w+")
    s.write("#\rX\rX\rX\rX\rX\rX\rX\rX\rX\r0\r")
    for i in range(0, len(coordsoutput)):
        s.write("1\t"+ str(i) + "\t" + "%5.2f \t" % (coordsoutput[i][0]) + "%5.3f \t" % float((coordsoutput[i][1])) + "0\r")
    s.close()
    #empty elongation file, necessary to run Matlab script
    e = open("elongation"+str(track)+".txt", "w+")
    e.write("#\r#\r#\r#\r#\r#\r#\r#\r#\r\r\r#Snake Data\r#\r1 0")
    e.close()
    #====================================================================================
    #UPDATE TRACKERS AND COORDSPIX
    track=track+1
    nfilaments = nfilaments -1
    locations = np.where(backbones[track]==True)
    coordspix = [np.array([locations[0][i], locations[1][i]]) for i in range(0, len(locations[0]))]
    ordered = sorted(coordspix, key=lambda k: [k[1], k[0]])
    ordered = [np.flip(x) for x in ordered]
    coordspix = np.copy(ordered)
    #====================================================================================
    #GETTING new COORDSPOLY
    xpoly = [coordspix[i][0] for i in range(0, len(coordspix))]
    ypoly = [coordspix[i][1] for i in range(0, len(coordspix))]
    #poly coeffs
    z = np.polyfit(xpoly,ypoly,10)
    #poly function
    p = np.poly1d(z)
    ypolyfinal = p(xpoly)
    #====================================================================================
    #GETTING new COORDSSPL
    
    xspl = [coordspix[i][0] for i in range(0, len(coordspix))]
    yspl = [coordspix[i][1] for i in range(0, len(coordspix))]
    coordsspl = np.copy(coordspix)
    n_chunks = 5
    x_chunks = chunkIt(xspl, n_chunks)
    y_chunks = chunkIt(yspl, n_chunks)
    ysplfinal = []
    
    for i in range(0,n_chunks):
        tck = interpolate.splrep(x_chunks[i], y_chunks[i], s=13)
        ynew = interpolate.splev(x_chunks[i], tck, der=0)
        ysplfinal.extend(ynew)
    
    #updating output coords
    for i in range(0, len(coordsspl)):
        if math.isnan(i):
            coordsspl[i].astype(float)
            coordsspl[i][1] = ysplfinal[i]
    xspl = [coordsspl[i][0] for i in range(0, len(coordsspl))]
    yspl = [coordsspl[i][1] for i in range(0, len(coordsspl))]

    #====================================================================================
    #UPDATE PLOTS

    #update figpix
    apix = figpix.add_subplot(111, frameon=False)
    apix.imshow(finals[track])
    canvaspix = FigureCanvasTkAgg(figpix, master=figures_menu)
    canvaspix.draw()
    canvaspix.get_tk_widget().grid(row=0, column=0)
    canvaspix.mpl_connect("key_press_event", on_key_press)
    counter_button = tk.Button(figures_menu, text=str(nfilaments)+" remaining filaments")
    counter_button.grid(row=0, column=3)
    
    #update figpoly
    figpoly = Figure(figsize=(3, 3), dpi=100)
    apoly = figpoly.add_subplot(111, frameon=False)
    apoly.plot(xpoly, ypoly, 'ro', xpoly, ypolyfinal, 'b', markersize=1)
    canvaspoly = FigureCanvasTkAgg(figpoly, master=figures_menu)
    canvaspoly.draw()
    canvaspoly.get_tk_widget().grid(row=1, column=0)
    canvaspoly.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Polynomial Snake", command=save_polyfigure)
    save_figure_button.grid(row=1, column=1)
    
    #update figspl
    figspl = Figure(figsize=(3, 3), dpi=100)
    aspl = figspl.add_subplot(111, frameon=False)
    aspl.plot(xspl, yspl, 'ro', xspl, ysplfinal, 'b', markersize=1)
    canvasspl = FigureCanvasTkAgg(figspl, master=figures_menu)
    canvasspl.draw()
    canvasspl.get_tk_widget().grid(row=2, column=0)
    canvasspl.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Spline Snake", command=save_polyfigure)
    save_figure_button.grid(row=2, column=1)
    

def save_splfigure():
    global track
    global finals
    global fig
    global nfilaments
    global backbones
    global figpix
    global figpoly
    global figspl
    global x
    global y
    global xploy
    global ypoly
    global xspl
    global yspl
    global coordspix
    global coordspoly
    global coordsspl
    global locations
    
    #====================================================================================
    #GETTING SPL OUTPUT
    coordsoutput = np.copy(coordsspl)
    #writting pix file
    s = open("snake"+str(track)+".txt", "w+")
    s.write("#\rX\rX\rX\rX\rX\rX\rX\rX\rX\r0\r")
    for i in range(0, len(coordsoutput)):
        s.write("1\t"+ str(i) + "\t" + "%5.2f \t" % (coordsoutput[i][0]) + "%5.3f \t" % float((coordsoutput[i][1])) + "0\r")
    s.close()
    #empty elongation file, necessary to run Matlab script
    e = open("elongation"+str(track)+".txt", "w+")
    e.write("#\r#\r#\r#\r#\r#\r#\r#\r#\r\r\r#Snake Data\r#\r1 0")
    e.close()
    #====================================================================================
    #UPDATE TRACKERS AND COORDSPIX
    track=track+1
    nfilaments = nfilaments -1
    locations = np.where(backbones[track]==True)
    coordspix = [np.array([locations[0][i], locations[1][i]]) for i in range(0, len(locations[0]))]
    ordered = sorted(coordspix, key=lambda k: [k[1], k[0]])
    ordered = [np.flip(x) for x in ordered]
    coordspix = np.copy(ordered)
    #====================================================================================
    #GETTING new COORDSPOLY
    xpoly = [coordspix[i][0] for i in range(0, len(coordspix))]
    ypoly = [coordspix[i][1] for i in range(0, len(coordspix))]
    #poly coeffs
    z = np.polyfit(xpoly,ypoly,10)
    #poly function
    p = np.poly1d(z)
    ypolyfinal = p(xpoly)
    #====================================================================================
    #GETTING new COORDSSPL
    
    xspl = [coordspix[i][0] for i in range(0, len(coordspix))]
    yspl = [coordspix[i][1] for i in range(0, len(coordspix))]
    coordsspl = np.copy(coordspix)
    n_chunks = 5
    x_chunks = chunkIt(xspl, n_chunks)
    y_chunks = chunkIt(yspl, n_chunks)
    ysplfinal = []
    
    for i in range(0,n_chunks):
        tck = interpolate.splrep(x_chunks[i], y_chunks[i], s=13)
        ynew = interpolate.splev(x_chunks[i], tck, der=0)
        ysplfinal.extend(ynew)
    
    #updating output coords
    for i in range(0, len(coordsspl)):
        if math.isnan(i):
            coordsspl[i].astype(float)
            coordsspl[i][1] = ysplfinal[i]
    xspl = [coordsspl[i][0] for i in range(0, len(coordsspl))]
    yspl = [coordsspl[i][1] for i in range(0, len(coordsspl))]

    #====================================================================================
    #UPDATE PLOTS

    #update figpix
    apix = figpix.add_subplot(111, frameon=False)
    apix.imshow(finals[track])
    canvaspix = FigureCanvasTkAgg(figpix, master=figures_menu)
    canvaspix.draw()
    canvaspix.get_tk_widget().grid(row=0, column=0)
    canvaspix.mpl_connect("key_press_event", on_key_press)
    counter_button = tk.Button(figures_menu, text=str(nfilaments)+" remaining filaments")
    counter_button.grid(row=0, column=3)
    
    #update figpoly
    figpoly = Figure(figsize=(3, 3), dpi=100)
    apoly = figpoly.add_subplot(111, frameon=False)
    apoly.plot(xpoly, ypoly, 'ro', xpoly, ypolyfinal, 'b', markersize=1)
    canvaspoly = FigureCanvasTkAgg(figpoly, master=figures_menu)
    canvaspoly.draw()
    canvaspoly.get_tk_widget().grid(row=1, column=0)
    canvaspoly.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Polynomial Snake", command=save_polyfigure)
    save_figure_button.grid(row=1, column=1)
    
    #update figspl
    figspl = Figure(figsize=(3, 3), dpi=100)
    aspl = figspl.add_subplot(111, frameon=False)
    aspl.plot(xspl, yspl, 'ro', xspl, ysplfinal, 'b', markersize=1)
    canvasspl = FigureCanvasTkAgg(figspl, master=figures_menu)
    canvasspl.draw()
    canvasspl.get_tk_widget().grid(row=2, column=0)
    canvasspl.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Spline Snake", command=save_polyfigure)
    save_figure_button.grid(row=2, column=1)
    



def analyse():
    global track
    global finals
    global fig
    global nfilaments
    global backbones
    global figpix
    global figpoly
    global figspl
    global x
    global y
    global xploy
    global ypoly
    global xspl
    global yspl
    global coordspix
    global coordspoly
    global coordsspl
    global locations
    
    #label, box, singlify, backbone and greyoverlap all in one
    labels, nobjects = ndi.label(multi_final)
    boxes = ndi.find_objects(labels)
    objects = [np.array(multi_final[boxes[i]]) for i in range(0,nobjects-1) if (np.array(multi_final[boxes[i]]).shape[0]*np.array(multi_final[boxes[i]]).shape[1] > 600)]
    singles =  singlify(objects)
    backbones = backbone(singles)
    finals = greyoverlap(singles, backbones)
    
    
    nfilaments = len(finals)-1
    
    #plotting figpix and saving coordspix
    locations = np.where(backbones[track]==True)
    coordspix = [np.array([locations[0][i], locations[1][i]]) for i in range(0, len(locations[0]))]
    ordered = sorted(coordspix, key=lambda k: [k[1], k[0]])
    ordered = [np.flip(x) for x in ordered]
    coordspix = np.copy(ordered)
    
    apix = figpix.add_subplot(111, frameon=False)
    apix.imshow(finals[track])
    canvaspix = FigureCanvasTkAgg(figpix, master=figures_menu)
    canvaspix.draw()
    canvaspix.get_tk_widget().grid(row=0, column=0)
    canvaspix.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Snake", command=save_pixfigure)
    save_figure_button.grid(row=0, column=1)
    skip_figure_button = tk.Button(figures_menu, text="Skip Snake", command=skip_filament)
    skip_figure_button.grid(row=0, column=2)
    counter_button = tk.Button(figures_menu, text=str(nfilaments)+" remaining filaments")
    counter_button.grid(row=0, column=3)
    
    
    #plotting figpoly and saving coordspoly
    xpoly = [coordspix[i][0] for i in range(0, len(coordspix))]
    ypoly = [coordspix[i][1] for i in range(0, len(coordspix))]
    #poly coeffs
    z = np.polyfit(xpoly,ypoly,10)
    #poly function
    p = np.poly1d(z)
    ypolyfinal = p(xpoly)
    
    apoly = figpoly.add_subplot(111, frameon=False)
    apoly.plot(xpoly, ypoly, 'ro', xpoly, ypolyfinal, 'b', markersize=1)
    canvaspoly = FigureCanvasTkAgg(figpoly, master=figures_menu)
    canvaspoly.draw()
    canvaspoly.get_tk_widget().grid(row=1, column=0)
    canvaspoly.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Polynomial Snake", command=save_polyfigure)
    save_figure_button.grid(row=1, column=1)
    coordspoly = [np.array([xpoly[i], p(xpoly[i])]) for i in range(0, len(xpoly))]
    
    
    #plotting figspl
    xspl = [coordspix[i][0] for i in range(0, len(coordspix))]
    yspl = [coordspix[i][1] for i in range(0, len(coordspix))]
    coordsspl = np.copy(coordspix)
    n_chunks = 5
    x_chunks = chunkIt(xspl, n_chunks)
    y_chunks = chunkIt(yspl, n_chunks)
    ysplfinal = []
    
    for i in range(0,n_chunks):
        tck = interpolate.splrep(x_chunks[i], y_chunks[i], s=13)
        ynew = interpolate.splev(x_chunks[i], tck, der=0)
        ysplfinal.extend(ynew)
    
    #updating output coords
    for i in range(0, len(coordsspl)):
        if math.isnan(i):
            coordsspl[i].astype(float)
            coordsspl[i][1] = ysplfinal[i]
    xspl = [coordsspl[i][0] for i in range(0, len(coordsspl))]
    yspl = [coordsspl[i][1] for i in range(0, len(coordsspl))]
    
    
    aspl = figspl.add_subplot(111, frameon=False)
    aspl.plot(xspl, yspl, 'ro', xspl, yspl, 'b', markersize=1)
    canvasspl = FigureCanvasTkAgg(figspl, master=figures_menu)
    canvasspl.draw()
    canvasspl.get_tk_widget().grid(row=2, column=0)
    canvasspl.mpl_connect("key_press_event", on_key_press)
    save_figure_button = tk.Button(figures_menu, text="Export Spline Snake", command=save_splfigure)
    save_figure_button.grid(row=2, column=1)
    
    
root = tk.Tk()
root.wm_title("Embedding in Tk")
root.minsize(500,500)

file_preproc_menu = tk.Frame(root)
file_preproc_menu.grid(row = 0, column=0)

figures_menu = tk.Frame(root)
figures_menu.grid(row=0, column=1)

openfile_button= tk.Button(file_preproc_menu, text="Open file", command=open_file)
openfile_button.grid(row=0, column=0)

preprogeneral_button= tk.Button(file_preproc_menu, text="Preprocess image", command=pre_processing_general)
preprogeneral_button.grid(row=1, column=0)

preprodark_button= tk.Button(file_preproc_menu, text="Preprocess dark image", command=pre_processing_dark)
preprodark_button.grid(row=2, column=0)

start_analysis_button = tk.Button(file_preproc_menu, text="Start Analysis", command=analyse)
start_analysis_button.grid(row=3, column=0)
    

    
def on_key_press(event):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas)



def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


quit_button = tk.Button(file_preproc_menu, text="Quit", command=_quit)
quit_button.grid(row=4, column=0)

tk.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.

#further developments
# show pre-processed image without axes