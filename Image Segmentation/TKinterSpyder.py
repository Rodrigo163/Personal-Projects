# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:07:21 2019

@author: rodrigo_lopez
"""

# %%
from tkinter import *
from tkinter import filedialog as fd

root = Toplevel()
root.title("Filament Analysis GUI")
file1 = fd.askopenfilename()
im = PhotoImage(file = file1)
label= Label(root, image =im)
label.pack()
root.mainloop()



#%%
from tkinter import *
from tkinter import filedialog as fd
root = Toplevel()
root.title("Filament Analysis GUI")

fil = fd.askopenfile()
im = PhotoImage(file=fil)
label= Label(root, image =im)
label.pack()

root.mainloop()