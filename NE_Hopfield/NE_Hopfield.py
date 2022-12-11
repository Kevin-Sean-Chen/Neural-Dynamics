# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 06:11:26 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import LinearRegression
import random

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

# %% Functions for NE Hopfield
def sigma(h,T):
    """
    Stochastic nonlinearity following Boltzmann machine
    """
    P = 1/(1+np.exp(-2*h/T))
    ll = len(h)
    rand = np.random.rand(ll)
    pos = np.where(P-rand>0)[0]
    r = -np.ones(ll)
    r[pos] = +1
    return r

def network_2state(N,alpha,gamma,eta):
    """
    Returns network weight matrix W that is NxN, constructed with two patterns and a bias term gammma
    Another term alpha governs the strength of overlapping between two patterns
    The last term eta controls the strength of a unrelated Gaussian noise
    Return the matrix and two patterns
    """
    #targets
    Vi = np.random.randint(0,2,N)
    Vi[Vi==0] = -1
    Vf = np.random.randint(0,2,N)
    Vf[Vf==0] = -1
    #weights
    weighted = gamma
    W = (1-weighted)*np.outer(Vi,Vi) + weighted*np.outer(Vf,Vf) + np.outer(Vi,Vf)*alpha + np.random.randn(N,N)*eta
    W = W-np.diag(W)
    return W, Vi, Vf

def work_protocol(W,U,kbT,bb, Vi, Vf, rep,max_it):
    """
    Work performed during forward and reverse trajectories between two patterns, 
    given the network W, kbT temperature, threshold U,
    repititions, and max iteration number, and the increasing rate of forcing protocol
    """
    #trials
    rep = rep
    max_it = max_it
    # physical
    kbT = kbT  #temperature
    U = U  #threshold
    bb = bb  #input weights
    N = W.shape[0]
    #dynamics and protocol
    wf = np.zeros(rep)
    wr = np.zeros(rep)
    
    for rr in range(rep):
        ###forward (i->f)
        overlap_f = 0  #initial overlap
        u = np.matmul(W, Vi)  #inital current
        #u = Vi.copy()
        Et = -0.5*Vi @ W @ Vi #0 #intial energy
        ww = 0  #counting steps
        I = 0  #initial input
        while overlap_f != N:
            Vc = sigma(u-U, kbT)  #voltage
            Ii = bb*Vf  #np.matmul(W,Vf)  #input current
            I = Ii + I
            u = np.matmul(W,Vc) + I  #Vf  #current
            overlap_f = int(np.dot(Vf,Vc))  #overlapping
            #u = sigma(np.matmul(W,u) + Vf - U,kbT)
            dw = np.dot(Vc,I) - Et  #work is difference of energy as a function of input change
            Et = np.dot(Vc,I)  #update E
            wf[rr] = wf[rr] + np.dot(Vc,Ii) #+dw #work trajectory
            ww = ww + 1                   
            if ww>max_it:
                break
        
        ###reverse (f->i)
        overlap_r = 0
        u = np.matmul(W, Vf)
        Et = -0.5*Vf @ W @ Vf
        ww = 0
        I = 0
        while overlap_r != N:
            Vc = sigma(u-U, kbT)  #voltage
            Ii = bb*Vi #np.matmul(W,Vi)  #input current
            I = Ii + I
            u = np.matmul(W,Vc) + I  #current
            overlap_r = int(np.dot(Vi,Vc))
            dw = np.dot(Vc,I) - Et
            Et = np.dot(Vc,I)
            wr[rr] = wr[rr] + np.dot(Vc,Ii)
            ww = ww + 1
            if ww>max_it:
                break
        wr[rr] = -wr[rr]  #as P(-W)
    return wf, wr

# %% analysis
def numeircal_dF(W,kbT,U, Vi,Vf):
    N = W.shape[0]
    spins = list(itertools.product([-1, 1], repeat=N))
    Esf = np.zeros(len(spins))
    Psf = np.zeros(len(Esf))
    Esi = np.zeros(len(spins))
    Psi = np.zeros(len(Esi))
    for ii in range(len(spins)):
        vv = np.array(spins[ii])
        Esf[ii] = -0.5* vv @ W @ vv - 1*Vf @ vv + np.ones(N)*U @ vv
        Psf[ii] = np.exp(-1/kbT*Esf[ii])
        Esi[ii] = -0.5* vv @ W @ vv - 1*Vi @ vv + np.ones(N)*U @ vv
        Psi[ii] = np.exp(-1/kbT*Esi[ii])
    
    # computing free-energy
    Zf = sum(Psf)
    Psf = Psf/Zf
    Zi = sum(Psi)
    Psi = Psi/Zi
    dF = (-kbT*np.log(Zf)+kbT*np.log(Zi))
    
    return dF

def CFT_cross(wf,wr,binsize):
    """
    Given the distribution of forward and reverse work performed and the bin size,
    calculate the free-energy difference by line crossing method
    """
    xm,xM = min(np.hstack([wf,wr])),max(np.hstack([wf,wr]))
    com_bin = np.arange(xm,xM,binsize)
    nwf,_= np.histogram(wf,bins=com_bin,normed=True)
    nwr,_= np.histogram(wr,bins=com_bin,normed=True)
    y_x = np.log(nwf/nwr)
    
    xs = com_bin[:-1]
    pos_inf = np.where(np.isinf(y_x)==False)[0]
    ys = y_x[pos_inf]
    xs = xs[pos_inf]
    pos_real = np.where(np.isnan(ys)==False)[0]
    ys = ys[pos_real]
    xs = xs[pos_real]
    reg = LinearRegression().fit(xs[:].reshape(-1, 1), ys[:].reshape(-1, 1))
    dF_CFT = -reg.intercept_/reg.coef_

    return dF_CFT    


# %% Experiments
###recordings
scans = np.arange(0.1,1,0.1)  #scanning parameter  #np.array([0.1,0.9])#
Wdiss = np.zeros(len(scans))
Ferr = np.zeros(len(scans))
true_dF = np.zeros(len(scans))
CFT_dF = np.zeros(len(scans))
rep = 3000
max_it = 1000
###network
N = 15
U = 0
kbT = 15
bb = 0.005
#alpha = 0.1
gamma = 0.2
eta = 0
binsize = 0.05

W_,Vi,Vf = network_2state(N,0,gamma,eta)
for ii in range(len(scans)):
    ###exp
    alpha = scans[ii]
    W = W_ + np.outer(Vi,Vf)*alpha
#    W,Vi,Vf = network_2state(N,alpha,gamma,eta)
    wf,wr = work_protocol(W,U,kbT,bb, Vi, Vf, rep,max_it)
    ###analysis
    true_dF[ii] = numeircal_dF(W,kbT,U, Vi,Vf)
    CFT_dF[ii] = CFT_cross(wf,wr,binsize)
    Wdiss[ii] = np.mean(wf)-true_dF[ii]
    
    
# %%
plt.figure()
plt.plot(scans, np.flipud(np.abs(true_dF-CFT_dF)[:,None]),'o',markersize=15)
plt.xlabel(r'$\alpha$',fontsize=40)
plt.ylabel('$|\delta F|$',fontsize=40)
# %%
plt.figure()
plt.plot(scans, Wdiss,'o',markersize=15)
plt.xlabel(r'$\alpha$',fontsize=40)
plt.ylabel('$W_{diss}$',fontsize=40)

# %%
#plt.figure()
#plt.plot(xs, ys,'o',label='data')
#plt.hlines(0,-.75,1.0,linestyle='dashed',label='$y=0$')
#plt.plot(xs,y_,'-',label='fitting')
##plt.plot(np.cumsum(nwr))
#plt.xlabel('$W$',fontsize=30)
#plt.ylabel('$log(P_f(W)/P_r(-W))$',fontsize=30)
#plt.legend(fontsize=25)