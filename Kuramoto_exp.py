# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 01:19:24 2021

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=60) 
matplotlib.rc('ytick', labelsize=60)

# %% 
N = 10
T = 100
dt = 0.1
time = np.arange(0,T,dt)
lt = len(time)
K = 3.5
eps = .1
ths = np.zeros((N,lt))
#Aij = np.random.randint(0,2,size=(N,N))
Aij = np.random.randn(N,N)
ww = np.random.rand(N)*2*np.pi
for tt in range(lt-1):
    ang_i, ang_j = np.meshgrid(ths[:,tt], ths[:,tt])
    temp = ths[:,tt] + dt*(ww + K/N*(Aij * np.sin((ang_j-ang_i))).sum(0)) + np.random.randn(N)*eps
    ths[:,tt+1] = np.mod(temp,np.pi*2)
# %%
plt.figure()
plt.imshow(ths,aspect='auto')

# %%
cc = np.cov(ths)
plt.figure()
plt.plot(Aij.reshape(-1),cc.reshape(-1),'o')
plt.xlabel('True',fontsize=50)
plt.ylabel('Corr',fontsize=50)
