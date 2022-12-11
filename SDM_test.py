#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:20:00 2019

@author: kschen
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from pylab import *
import time

#from brian2 import *
#from brian2.units.allunits import henry

#%matplotlib inline

plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']
plt.rcParams.update({'font.family': 'serif', 
                     'font.serif':['Computer Modern']})

# %% Resonate-and-fire neuron
### parameters
lam = 1  #Hz
tau_m = 1000  #in ms
gl = 30  #nS
C = 4  #nF
w = 31.4  #rad/s
f  =5  #Hz
T  = 200  #ms
L = 1  #Mhenry
El = -60  #mV
Vt = -58
Ut = 0
### map to R&F
ll = gl/C
ww = 2*np.pi/T

### 


# %% SDM test
import matplotlib.image as img
ph = 0.1
file_name = '/home/kschen/Downloads/jonathanpillow.jpg'
image = img.imread(file_name)
file_name2 = '/home/kschen/Downloads/andrew Leifer.jpg'
image2 = img.imread(file_name2)
M = np.squeeze(image[50:200,50:200,0])
M2 = np.squeeze(image2[50:200,0:150,0])
WI = np.random.rand(len(M)**2,len(M)**2)
WI[WI>ph] = 1
WI[WI<=ph] = 0
mm = np.vstack((M.reshape(-1),M2.reshape(-1))).T
WH = mm @ mm.T @ WI #np.outer(mm,mm) @ WI
###test with noisy cue pattern
noise = np.random.rand(len(mm))
zz = M.reshape(-1) + noise*100
def NL(X,th):
    X[X<th]=0
    X[X>=th]=1
    return X
u = WH @ NL((WI.T @ zz),100)
plt.imshow(u.reshape(M.shape))
plt.figure()
plt.imshow(zz.reshape(M.shape))
