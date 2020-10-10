# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:01:55 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dotmap as DotMap
import itertools

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

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
T = 50000  #length of simulation
Ws = np.zeros((nneuron,nneuron))  #weights
v1 = np.array([1,0,1,0,1,0])#np.random.randn(nneuron)
v2 = np.array([0,1,0,1,0,1])#
vw = np.random.randn(nbasis)
bias1 = 0.1  #biased towards pattern1
rr = 5  #reccurent strength
bb = 0.05  #input strenght
ww = rr*(np.outer(v1,v1)*(bias1) + np.outer(v2,v2)*(1-bias1)) + np.random.randn(nneuron,nneuron)*.01  #low-rank structure

us = np.random.randn(nneuron,T)*0.01
rt = np.zeros_like(us)
dt = 0.01

# %% neural dynamics
for tt in range(pad,T-1):
    rt[:,tt] = NL(ww @ us[:,tt])
    us[:,tt+1] = us[:,tt] + dt*(-us[:,tt] + rt[:,tt]) + v2*bb + np.random.randn(nneuron)*1.
plt.figure()
plt.imshow(rt,aspect='auto')
plt.colorbar()
plt.xlabel('time steps')
plt.ylabel('cell')

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
import ssm
# %% based on HMM
obs = rt[:4,:].T
obs_dims = obs.shape[1]
N_iters = 50
num_states = 2  #assuming transition between two patterns
hmm = ssm.HMM(num_states, obs_dims, num_iters='gaussian')
hmm_lls = hmm.fit(obs, method='em',num_iters=N_iters)

# %%
plt.figure()
plt.plot(hmm_lls)
most_lls = hmm.most_likely_states(obs)
plt.figure()
plt.subplot(211)
plt.plot(most_lls)
plt.xlim([0,len(most_lls)])
plt.subplot(212)
plt.imshow(obs.T,aspect='auto')

learned_transition_mat = hmm.transitions.transition_matrix
print(learned_transition_mat)

# %% MLE-based $$$

# %% based on ML/corr
ML = np.zeros((2,T))
for tt in range(0,T):
    ML[0,tt], ML[1,tt] = np.dot(v1,rt[:,tt]), np.dot(v2,rt[:,tt])
    
# %% based on HMM
obs = ML.T #ML[1,:][None,:].T #
obs_dims = obs.shape[1]
N_iters = 100
num_states = 2  #assuming transition between two patterns
hmm = ssm.HMM(num_states, obs_dims, num_iters='gaussian')
hmm_lls = hmm.fit(obs, method='em',num_iters=N_iters)

# %%
plt.figure()
plt.plot(hmm_lls)
most_lls = hmm.most_likely_states(obs)
plt.figure()
plt.plot(most_lls)

learned_transition_mat = hmm.transitions.transition_matrix
print(learned_transition_mat)
print(np.log(learned_transition_mat[0,1]/learned_transition_mat[1,0]))

# %%
plt.figure()
plt.subplot(211)
plt.plot(ML[0,:])
plt.xlim([0,len(most_lls)])
plt.ylabel('correlation')
plt.subplot(212)
plt.plot(most_lls)
plt.xlim([0,len(most_lls)])
plt.ylabel('state assigned')
plt.xlabel('time steps')

# %% dwell time
difs = np.diff(most_lls)
pos = np.where(difs==1)[0]
durs = np.diff(pos)
plt.hist(durs,50)
pos = np.where(difs==-1)[0]
durs = np.diff(pos)
plt.hist(durs,50,alpha=0.5)

# %% enumerating free energy
U = 0
kbT = 1
spins = list(itertools.product([-1, 1], repeat=nneuron))
Esf = np.zeros(len(spins))
Psf = np.zeros(len(Esf))
Esi = np.zeros(len(spins))
Psi = np.zeros(len(Esi))
for ii in range(len(spins)):
    vv = np.array(spins[ii])
    Esf[ii] = -0.5* vv @ ww @ vv - 0*v2 @ vv +0* np.ones(nneuron)*U @ vv
    Psf[ii] = np.exp(-1/kbT*Esf[ii])
    Esi[ii] = -0.5* vv @ ww @ vv - bb*v2 @ vv + 0*np.ones(nneuron)*U @ vv
    Psi[ii] = np.exp(-1/kbT*Esi[ii])

# computing free-energy
Zf = sum(Psf)
Psf = Psf/Zf
Zi = sum(Psi)
Psi = Psi/Zi
print(-kbT*np.log(Zf)+kbT*np.log(Zi))
