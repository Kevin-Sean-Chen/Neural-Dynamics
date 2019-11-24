#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:53:58 2019

@author: kschen
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
#%matplotlib inline

# %% Lorentz attractor
def Lorenz_dt(xyz, dt):
    """
    Lorenz attractor in 3-D with parameters from the paper
    """
    x, y, z = xyz
    dx = (10*(y-x))*dt
    dy = (x*(28-z)-y)*dt
    dz = (x*y-(8/3)*z)*dt
    return dx,dy,dz

def Lorenz_model(rep):
    """
    Repeat some samples with ran length T and time step dt
    """
    
def Delay_embedding(X,d):
    """
    N dimentional time series X with d step embedding
    """

def Mode_decomp(Y):
    """
    Dynamic mode decomposition for the embedded time series
    """
    
def NN_prediction():
    