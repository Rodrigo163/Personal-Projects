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
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from scipy.optimize import curve_fit


def mfileopen():
    file1 = fd.askopenfile()
    label = Label(text=file1).pack()
    file2 = file1.name
#    f = open(file2)
    #label2 = Label(text=f.read()).pack()
    w = []
    Gp = []
    Gdp = []
    tan = []
    wlist = []
    Gplist = []
    Gdplist = []
    tanlist = []
    with open(file2, "r") as txtfile:
                content = txtfile.readlines()
                content = content[3:]
                for line in content:
                    line = line.replace("\n", "")
                    line = line.split("\t")
                    if line[0] == "\x00":
                        continue
                    line = list(map(float, line))
                    l0 = line[0]
                    w.append(l0)
                    l1 = line[1]
                    Gp.append(l1)
                    l2 = line[2]
                    Gdp.append(l2)
                    l3 = line[3]
                    tan.append(l3)
                wlist.append(w)
                w = []
                Gplist.append(Gp)
                Gp = []
                Gdplist.append(Gdp)
                Gdp = []
                tanlist.append(tan)
                tan = []
    wlist = np.array(wlist, dtype='float64')
    Gplist = np.array(Gplist, dtype='float64')
    Gdplist = np.array(Gdplist, dtype='float64')
    tanlist = np.array(tanlist, dtype='float64')          
    return wlist[0], Gplist[0], Gdplist[0]
 

def model(x, eps, Lambda):
    N = np.array(list(range(1,n_max + 1)),dtype='float64')
    Len = L * np.ones(n_max)
    lA = Len / N
#---------------------------------------------------------
    tauvec = []
    for n in range(n_max):
        l = lA[n]
        if l > Lambda:
            tau = sigma / (l_p * kBT * math.pi**4 / (l**4)) * np.exp(eps * (l / Lambda - 1))
            tauvec.append(tau)        
        else:
            tau = sigma / (l_p * kBT * math.pi**4 / (l**4))
            tauvec.append(tau)   
    tauvec = np.array(tauvec, dtype='float64')
#---------------------------------------------------------    
    Chi = []
    G_total = []
    for i in range(0,len(x)):
        omega_i = x[i]
        chi = L**4 / (math.pi**4 * l_p**2 * kBT) * np.sum(1 / (N**4 * (1 + 1j * omega_i * tauvec)))
        Chi.append(chi)
    Lam = Lambda * np.ones(len(Chi))
    G = 1 / (5 * xi**2) * Lam / Chi
    Greal = np.array(G.real,dtype='float64')
    Gimag = np.array(G.imag,dtype='float64')
    G_total.append(Greal)
    G_total.append(Gimag)
    tanP = Gimag / Greal
    tanP = np.array(tanP,dtype='float64')   

    return Greal
    #return tanP

def plot_gWLC(omega_in, eps_in, Lambda_in):
    N = np.array(list(range(1,n_max + 1)),dtype='float64')
    Len = L * np.ones(n_max)
    lA = Len / N
#---------------------------------------------------------
    tauvec = []
    for n in range(n_max):
        l = lA[n]
        if l > Lambda_in:
            tau = sigma / (l_p * kBT * math.pi**4 / (l**4)) * np.exp(eps_in * (l / Lambda_in - 1))
            tauvec.append(tau)        
        else:
            tau = sigma / (l_p * kBT * math.pi**4 / (l**4))
            tauvec.append(tau)   
    tauvec = np.array(tauvec, dtype='float64')
#---------------------------------------------------------    
    Chi = []
    for i in range(0,len(omega_in)):
        omega_i = omega_in[i]
        #chi = L**4 / (math.pi**4 * l_p**2 * kBT) * np.sum(1 / (N**4 * (1 + 1j * omega_i * tauvec / 2)))
        chi = L**4 / (math.pi**4 * l_p**2 * kBT) * np.sum(1 / (N**4 * (1 + 1j * math.pi * omega_i * tauvec)))
        Chi.append(chi)
    Lam = Lambda_in * np.ones(len(Chi))
    G = 1 / (5 * xi**2) * Lam / Chi
    Greal = np.array(G.real,dtype='float64')
    Gimag = np.array(G.imag,dtype='float64')
    tanP = Gimag / Greal

    return Greal, Gimag, tanP


def plot(plt_x, plt_y):
    plt.figure()
    plt.plot(plt_x, plt_y,'k')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("x")
    plt.ylabel("y'")
    plt.legend(loc="upper right")
    plt.show()

#--------------------------------------------------------- 
# Define geometry and GUI properties
a = Tk()
a.title("gWLC fitting tool")
a.geometry("640x960+0+0")
#--------------------------------------------------------- 
def Run_Fit():
    eps_0 = float(eps.get())
    Lambda_0 = float(Lambda.get()) * 10**(-6)
    global n_max
    n_max = int(Sampling.get())
    global guess
    guess = np.array([eps_0, Lambda_0 ])
# Environmental parameters:    
    T_env = float(T.get())
    global kBT
    kBT = 1.38064852 * 10**-(23) * (273 + T_env) #(J)
    global sigma
    sigma = float(fric_par.get()) *10**(-3)
    global xi
    xi = float(Mesh_size.get()) * 10**(-6) #(m) Meshsize
    global L
    L = float(Cont_len.get()) * 10**(-6) #(m) Contour length 
    global l_p
    l_p = float(Pers_len.get()) * 10**(-6) #(m) Persistence length
    global w_fin
    w_fin = int(omega_final.get())
#---------------------------------------------------------     
    w, Gp, Gpp = mfileopen()
    f_end = np.where(w > w_fin)  # Search for index where omega >= 10 Hz
    f_end = f_end[0][0]
    w = w[0:f_end]
    Gp = Gp[0:f_end]
    Gpp = Gpp[0:f_end]
    tan_data = Gpp / Gp
#---------------------------------------------------------  
    popt, pcov = curve_fit(model, w, Gp, guess)
    #popt, pcov = curve_fit(model, w, tan_data, guess)
    Gprime, Gdprime, tanGp_Gpp = plot_gWLC(w,popt[0],popt[1])
    print(popt, pcov)
#---------------------------------------------------------   
    # Embed in GUI
    global fig
    fig = Figure(figsize = (9, 6), facecolor = "white")
    axis = fig.add_subplot(111)
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel("Frequency / Hz", fontsize=16)
    axis.set_ylabel("Elastic modulus, Loss modulus / Pa", fontsize=16)
    
    axis.plot(w,Gprime,"-b",label="Elastic modulus")
    axis.plot(w,Gdprime,"-r",label="Loss modulus")
    axis.scatter(w[0:f_end],Gp[0:f_end],color="blue")
    axis.scatter(w[0:f_end],Gpp[0:f_end],color="red")
    axis.legend()
    canvas = FigureCanvasTkAgg(fig, master = a)
    canvas._tkcanvas.pack(side = tk.BOTTOM, fill = tk.BOTH)
    #fig.savefig('plot.png') 
    
    epsilon_fit = popt[0]
    Lambda_fit = str(popt[1])
    cov_eps_Gp = str(pcov[0][0])
    cov_Lam_Gp = str(pcov[0][1])
    cov_eps_Gpp = str(pcov[1][0])
    cov_Lam_Gpp = str(pcov[1][1])
    label_res=Label(a,text="Fitting parameters: ε = %2f , Λ = %s" % (epsilon_fit, Lambda_fit), font=("arial", 10, "bold"), fg="black").place(x=80,y=410)
    label_res=Label(a,text="Covariances for elastic modulus: ε : %s , Λ : %s" % (cov_eps_Gp, cov_Lam_Gp), font=("arial", 10, "bold"), fg="blue").place(x=80,y=430)
    label_res=Label(a,text="Covariances for loss modulus: ε : %s , Λ : %s" % (cov_eps_Gpp, cov_Lam_Gpp), font=("arial", 10, "bold"), fg="red").place(x=80,y=450)
    
    
    
#    save_fig = asksaveasfilename(filetypes=(("PNG Image", "*.png"),("All Files", "*.*")), 
#            defaultextension='.png', title="Window-2")
#    if save_fig:
#        plt.savefig(a)
    
def save_figure():
    file2save = fig  
    file2save = fd.asksaveasfile(mode='w', defaultextension='.png')
    file2save_2 = file2save.name
    fig.savefig(str(file2save_2)) 
    
def _quit():
    a.quit()     # stops mainloop
    a.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    
run = Button(a, text="Import data and run Fit", width=20, height=1, bg="lightblue", command=Run_Fit).place(x=250, y=380)
#saveButton = Button(a, text="Save plot figure", width=20, height=1, bg="lightblue", command=save_figure).place(x=250, y=500)
#saveButton = Button(a, text="Export figure", width=20, height=1, bg="lightblue", command=save_figure).place(x=450, y=930)

exit_button = tk.Button(master=a, text='Quit', command=_quit)
exit_button.pack(side=tk.BOTTOM)
saveButton = Button(a, text="Export figure", width=20, height=1, bg="lightblue", command=save_figure).pack(side=tk.BOTTOM)

#-----------------------------------------------------------------------------------------------------------------------
# Headers
heading = Label(a, text="GLASSY WORMLIKE CHAIN", font=("arial", 30, "bold"), fg="steelblue").pack()
MatPars = Label(a, text="Enter Material Parameters", font=("arial", 10, "bold"), fg="blue").place(x=50,y=150)
ModPars = Label(a, text="Guess Model Parameters", font=("arial", 10, "bold"), fg="blue").place(x=420,y=150)
#-----------------------------------------------------------------------------------------------------------------------
# Parameter placeholders
T = StringVar()
Cont_len = StringVar()
Pers_len = StringVar()
Mesh_size = StringVar()
fric_par = StringVar()
eps = StringVar()
Lambda = StringVar()
Sampling = StringVar()
omega_final = StringVar()
#-----------------------------------------------------------------------------------------------------------------------
# Input instructions material parameters
label_T=Label(a, text = "Temperature \ °C", font=("arial", 10, "bold"), fg="black").place(x=50,y=180)
label_Cont_len=Label(a, text = "Contour length \ μm", font=("arial", 10, "bold"), fg="black").place(x=50,y=220)
label_Pers_len=Label(a, text = "Persistence length \ μm", font=("arial", 10, "bold"), fg="black").place(x=50,y=260)
label_Mesh_size=Label(a, text = "Mesh size  ξ \ μm", font=("arial", 10, "bold"), fg="black").place(x=50,y=300)
label_fric_par=Label(a, text = "Drag coefficient ζ / 1000", font=("arial", 10, "bold"), fg="black").place(x=50,y=340)
# Input instructions model parameters
label_eps=Label(a, text = "ε", font=("arial", 10, "bold"), fg="black").place(x=410,y=180)
label_Lambda=Label(a, text = "Λ \ μm", font=("arial", 10, "bold"), fg="black").place(x=410,y=220)
label_Sampling=Label(a, text = "N", font=("arial", 10, "bold"), fg="black").place(x=410,y=260)
label_omega=Label(a, text = "Apply fit up to final ω / Hz", font=("arial", 10, "bold"), fg="black").place(x=410,y=300)
#-----------------------------------------------------------------------------------------------------------------------
# Entry boxes material parameters
T_=Entry(a,textvariable=T).place(x=250,y=180)
Cont_len_=Entry(a,textvariable=Cont_len).place(x=250,y=220)
Pers_len_=Entry(a,textvariable=Pers_len).place(x=250,y=260)
Mesh_size_=Entry(a,textvariable=Mesh_size).place(x=250,y=300)
fric_par_=Entry(a,textvariable=fric_par).place(x=250,y=340)
# Entry boxes model parameters
eps_=Entry(a,textvariable=eps).place(x=460,y=180)
Lambda_=Entry(a,textvariable=Lambda).place(x=460,y=220)
Sampling_=Entry(a,textvariable=Sampling).place(x=460,y=260)
omega_fin_ = Entry(a,textvariable=omega_final).place(x=460,y=340)

    
#button = Button(text="Import Data",width = 30,command = mfileopen).place(x=220,y=75)
#button = Button(text="Import Data",width = 30,command = mfileopen).pack()

a.mainloop()
