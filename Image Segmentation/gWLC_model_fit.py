# -*- coding: utf-8 -*-
"""
@author: Mollenkopf
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# GUESS PARAMETERS EPSILON AND LAMBDA
eps_0 = 12
Lambda_0 = 2 * 10**(-6)
guess = np.array([eps_0, Lambda_0 ])
# Environmental parameters:
kBT = 1.38064852 * 10**-(23) * (273 + 20) #(J)
#---------------------------------------------------------
# Set network parameters
sigma = 2 * 10**(-3) #(Pa s)
xi = 0.28 * 10**(-6) #(m) Meshsize
L = 16 * 10**(-6) #(m) Contour length 
l_p = 9 * 10**(-6) #(m) Persistence length
#---------------------------------------------------------
# Set model parameters
n_max = 1000
#Lambda = 2.05 * 10**(-6) #(m) Interaction length
#eps = 13
f = 0
f_E = l_p * kBT * math.pi**2 / L**2
omega_ini = -2
omega_fin = 2
omega_sample = 94
#---------------------------------------------------------
omega = np.logspace(omega_ini, omega_fin, omega_sample)
#---------------------------------------------------------
# Data input. Output: G', G'', tan(G'/G'')
def importtxt(filename):
    w = []
    Gp = []
    Gdp = []
    tan = []
    wlist = []
    Gplist = []
    Gdplist = []
    tanlist = []
    with open(filename, "r") as txtfile:
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
#---------------------------------------------------------
# Model: Input epsilon and Lambda
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
    for i in range(0,len(x)):
        omega_i = x[i]
        chi = L**4 / (math.pi**4 * l_p**2 * kBT) * np.sum(1 / (N**4 * (1 + 1j * omega_i * math.pi * tauvec / 2)))
        #chi = L**4 / (math.pi**4 * l_p**2 * kBT) * np.sum(1 / (N**4 * (1 + 1j * omega_i * tauvec / 2)))
        Chi.append(chi)
    Lam = Lambda * np.ones(len(Chi))
    G = 1 / (5 * xi**2) * Lam / Chi
    #G = np.array(G,dtype='float64')       
    return G.real
#---------------------------------------------------------
# Subfunctions
    
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


def main():
    w, Gp, Gpp = importtxt('actin-nhs-peg.txt')
    w = w[0:84]
    Gp = Gp[0:84]
    Gpp = Gpp[0:84]
    
    
    plt.figure(1)
    plt.plot(w,Gp,color='blue')
    plt.plot(w,Gpp,color='orange')
    plt.yscale('log')
    plt.xscale('log')


    plt.figure(2)
    plt.scatter(w,Gp,color='blue', label='Data')
    popt, pcov = curve_fit(model, w, Gp, guess)
    plt.plot(w, model(w, *popt), 'k-',linewidth=3,color='red', label='fit gWLC'+str(popt))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    Gprime, Gdprime, tanGp_Gpp = plot_gWLC(w,popt[0],popt[1])
    plt.figure(3)
    plt.plot([], [], ' ', label="Fit gWLC: " + "eps="+str(popt[0])+"\n" "      Lambda="+str(popt[1]))
    plt.yscale('log')
    plt.xscale('log')   
    plt.scatter(w,Gp,color='blue', label='Data Gp')
    plt.scatter(w,Gpp,color='green', label='Data Gpp')
    plt.plot(w,Gprime,color='blue', label='Gp')
    plt.plot(w,Gdprime,color='green', label='Gpp')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
#    r = np.where(w > 10)  # Search for index where omega >= 10 Hz
#    print(r[0][0])
# TODO: Read from data: omega_ini, omega_fin, sample size omega
if __name__ == "__main__":
    main()
