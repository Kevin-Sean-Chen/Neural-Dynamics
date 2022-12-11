# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 23:59:25 2022

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools
from scipy.signal import correlate

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %% functions
def Current(h,J,s):
    theta = h + J @ s
    return theta

def Transition(si,thetai,beta):
    P = np.exp(-si*thetai*beta)/(2*np.cosh(thetai*beta))
    rand = np.random.rand()
    if P>rand:
        s_ = -si.copy()
    else:
        s_ = si.copy()
    return s_

def Kinetic_Ising(X,Phi,S,J,kbT):
    beta = 1/kbT
    N = len(S)
    Ht = X*Phi
    Theta = Current(Ht, J, S)
    S_ = np.zeros(N)
    for ss in range(N):
        S_[ss] = Transition(S[ss],Theta[ss],beta)
    return S_

def Learning(J,S_,S):
    J_ = (1-tau)*J + mu_ss*np.outer(S_,S_) - mu_s*np.outer(S, J@S)
    return J_

# %% dynamics
mu_ss,mu_s,tau = .1, .1,  0.1
kbT = 1
N = 10
T = 3000
S = np.random.randint(0,2,size=(N,T))
S[S==0]=-1
J = np.zeros((N,N,T))
J0 = np.random.randn(N,N)
J[:,:,0] = J0
for tt in range(0,T-1):
    S[:,tt+1] = Kinetic_Ising(0,0,S[:,tt],J[:,:,tt],kbT)
    J[:,:,tt+1] = Learning(J[:,:,tt],S[:,tt+1],S[:,tt])

# %%
plt.figure()
plt.imshow(S[:,:2000],aspect='auto')
    
