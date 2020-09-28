# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:01:55 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dotmap as DotMap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5

# %% functions

def flipkernel(k):
    """
    flipping kernel to resolve temporal direction
    """
    return np.squeeze(np.fliplr(k[None,:])) ###important for temporal causality!!!??
    
def kernel(theta, pad):
    """
    Given theta weights and the time window for padding,
    return the kernel contructed with basis function
    """
    nb = len(theta)
    basis = basis_function1(pad, nb)  #construct basises
    k = np.dot(theta, basis.T)  #construct kernels with parameter-weighted sum
    return flipkernel(k)

def basis_function1(nkbins, nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    #nBases = 3
    #nkbins = 10 #binfun(duration); # number of bins for the basis functions
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.4),(nBases,1))  #take log for nonlinear time
    dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
    width = 5.*dbcenter # width of each bump
    bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)  #raise-cosine function formula
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] 
    #plt.plot(np.array(BBstm).T)
    return np.array(BBstm).T

def NL(x):
    return 1/(1+np.exp(-x))
# %% neural network settings
nneuron = 5
pad = 100  #window for kernel
nbasis = 7  #number of basis
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T  #basis function used for kernel approximation
T = 10000  #length of simulation
coeff = np.random.randn(nneuron,nneuron,nbasis)  #random basis functions
Ws = np.zeros((nneuron,nneuron,pad))
v1 = np.array([1,0,1,0,1])#np.random.randn(nneuron)
v2 = np.array([0,1,0,1,0])#
vw = np.random.randn(nbasis)
ww = np.outer(v1,v2)/nneuron*2 + np.random.randn(nneuron,nneuron)*.1  #low-rank structure
for ii in range(nneuron):
    for jj in range(nneuron):
#        Ws[ii,jj,:] = np.dot(coeff[ii,jj,:],Ks)*1  #random kernels
        Ws[ii,jj,:] = ww[ii,jj]*np.dot(vw,Ks)*1  # fixed kernels with low-rank weights
#        Ws[ii,jj,:] = ww[ii,jj]*np.dot(np.random.randn(nbasis),Ks)*1  #random kernels with low-rank weights

us = np.random.randn(nneuron,T)*0.01
us_k = np.random.randn(nneuron,T)*0.01
rt = np.zeros_like(us)
rt_k = np.zeros_like(us)
dt = 0.01

# %% neural dynamics
for tt in range(pad,T-1):
    rt_k[:,tt] = NL(np.einsum("ijk,jk->i",Ws,us_k[:,tt-pad:tt])/pad)
    rt[:,tt] = NL(np.dot(Ws[:,:,10],us[:,tt]))
    us_k[:,tt+1] = us_k[:,tt] + dt*(-us_k[:,tt] + rt_k[:,tt]) + np.random.randn(nneuron)*.1
    us[:,tt+1] = us[:,tt] + dt*(-us[:,tt] + rt[:,tt]) + np.random.randn(nneuron)*.1
plt.figure()
plt.subplot(121)
plt.imshow(rt_k,aspect='auto')
plt.subplot(122)
plt.imshow(rt,aspect='auto')

# %% linear analysis
lamb = np.linalg.eigvals(Ws[:,:,10])
plt.figure()
plt.scatter(np.real(lamb),np.imag(lamb))

# %%
###############################################################################
# Rate RNN
###############################################################################
# %% neural network settings
nneuron = 6
T = 10000  #length of simulation
Ws = np.zeros((nneuron,nneuron))  #weights
v1 = np.array([1,0,1,0,1,0])#np.random.randn(nneuron)
v2 = np.array([0,1,0,1,0,1])#
vw = np.random.randn(nbasis)
ww = np.outer(v1,v1)*0.5 + np.outer(v2,v2)*0.5 + np.random.randn(nneuron,nneuron)*.01  #low-rank structure

us = np.random.randn(nneuron,T)*0.01
rt = np.zeros_like(us)
dt = 0.01

# %% neural dynamics
for tt in range(pad,T-1):
    rt[:,tt] = NL(ww @ us[:,tt])
    us[:,tt+1] = us[:,tt] + dt*(-us[:,tt] + rt[:,tt]) + np.random.randn(nneuron)*.5
plt.figure()
plt.imshow(rt,aspect='auto')

# %% auto-corr
def autocorrelation (x) :
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x-np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:int(x.size/2)]/np.sum(xp**2)

plt.figure()
plt.plot(autocorrelation(rt[1,:])[:5000])

# %% Dwell-time calculation!