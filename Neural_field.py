# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 07:53:01 2021

@author: kevin
"""

import numpy as np
import scipy as sp
from scipy.optimize import minimize
import itertools
import copy

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=35) 
matplotlib.rc('ytick', labelsize=35) 

import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

# %% Neural fields
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% parameters
L = 20
T = 10
tau = 0.1
a = 3
J0 = .1
noise = 0.01
dt = 0.1
k = 1
lt = int(T/dt)

# %% settings
def RR(x,y,R):
    JJ = np.zeros((L,L))
    for xx in range(L):
        for yy in range(L):
            JJ[xx,yy] = J0/(2*np.pi*a)**0.5 * np.exp(-((x-xx)**2+(y-yy)**2)/2/a**2)
    return np.sum(JJ*R)

#def NL(x):
#    up = x.copy()
#    up[np.where(up<0)[0]] = 0
#    up = up**2
#    return up / (1+k*np.sum(up))
def NL(x):
    nl = 1/(1+np.exp(x))
    return nl

# %% dynamics
    
U = np.zeros((L,L,lt))+0.1
R = np.zeros((L,L,lt))+0.1
In = np.random.randn(L,L,lt)*1.  #set better input later

for tt in range(0,lt-1):
    R[:,:,tt] = NL(U[:,:,tt])  #activation
    for xx in range(1,L-1):
        for yy in range(1,L-1):
            JJ = RR(xx,yy,R[:,:,tt])  #recurrent signal
            U[xx,yy,tt+1] = U[xx,yy,tt] + 1/tau*dt*(-U[xx,yy,tt] + JJ + In[xx,yy,tt]) \
            + np.sqrt(dt*np.random.rand())*noise  #potential


# %% Dynamic mode decomposition
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%



