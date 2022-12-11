# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:53:43 2021

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

# %%
# %% Stimuli
T = 5000
D = 3
smooth = 100
noise = np.random.randn(T,D)
X = noise.copy()
for ii in range(D):
    X[:,ii] = np.convolve(noise[:,ii],np.ones(smooth),'same')/smooth *1

# %% Network settings
N = 20
Phi = np.random.randn(D,N)
J = np.random.randn(N,N)/N**0.5
J = 0.5*(J + J.T)  #make symmetric
J = J-np.diag(J)
#vv = np.random.randn(N)
#J = np.outer(vv,vv)*0.1 + np.random.randn(N,N)*0.05
#J = J_nMF.copy()

def Current(h,J,s):
    theta = h + J @ s
    return theta

def Transition(si,thetai,beta):
    P = np.exp(-si*thetai*beta)/(2*np.cosh(thetai*beta))
    rand = np.random.rand()
    if P>rand:
        s_ = -si.copy()
    elif P<=rand:
        s_ = si.copy()
    return s_

# %% Dynamics
def Kinetic_Ising(X,Phi,J,kbT):
    beta = 1/kbT
    N, T = Phi.shape[1], X.shape[0]
    S = np.ones((N,T))
    for tt in range(0,T-1):
        Ht = X[tt,:] @ Phi
        Theta = Current(Ht, J, S[:,tt])
        for ss in range(N):
            S[ss,tt+1] = Transition(S[ss,tt],Theta[ss],beta)
    return S

kbT = 1
S = Kinetic_Ising(X,Phi,J,kbT)

# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% FORCE
iters = 100
gamma = 0.001
P_ = np.zeros((N,N))#np.linalg.pinv(np.cov(S))
np.fill_diagonal(P_, np.ones(N)*1)
J_ = np.random.randn(N,N)
targ_X = np.linalg.pinv(Phi) @ X.T
for ii in range(iters):
    S = Kinetic_Ising(X,Phi,J_,kbT)
    for tt in range(1,T):
        dJ = gamma*np.outer((targ_X[:,tt] - S[:,tt]) , (P_ @ S[:,tt]) )
        #gamma*np.einsum("ik,jk->ij",(targ_X - S), (P_ @ S))  #############
        J_ = J_ + dJ
        P_ = P_ - (P_ @ (np.outer(S[:,tt],S[:,tt])) @ P_) / (1+ S[:,tt].T @ (P_ @ S[:,tt]))   #############
    
# %%
plt.figure()
plt.plot(J[:],J_[:],'o')

# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Original FORCE algorithm
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% setting
alpha = 1.0
dt = 0.1
P_ = np.zeros((N,N))#np.linalg.pinv(np.cov(S))
np.fill_diagonal(P_, np.ones(N)*1/alpha)
J_ = np.random.randn(N,N)
targ_X = np.linalg.pinv(Phi) @ X.T
V = np.random.randn(N)
R = np.random.randn(N)
Vd = np.random.randn(N)
Rd = np.random.randn(N)
p = 1.0
g = 1.5				# g greater than 1 leads to chaotic networks.
scale = 1.0/np.sqrt(p*N)
J_ = np.random.randn(N,N)*g*scale
Jd = J_.copy()

# %% iteration
for tt in range(0,3):
    ##plastic dynamics
    V = V - dt*V + dt*(J_ @ R)
    R = np.tanh(V)
    Z = Phi @ R
    ##driven dynamics
    Vd = Vd - dt*Vd + dt*(Jd @ Rd) + dt*(X[tt,:] @ Phi)
    Rd = np.tanh(Vd)
    ##error
    ###err = Z - X[tt,:]
    err = J_ @ V - Jd @ Vd - X[tt,:] @ Phi
    ##update inverse matrix
    k = P_ @ R
    rPr = R.T @ k
    c = 1.0/(1.0+rPr)
    P_ = P_ -k @ np.outer(k, c)
    ##update matix
    ###dw = -np.sum(err)*k*c  # + np.repeat(dw[:,None],N,1) # (err* P_ @ R)
    J_ = J_ + np.outer(err, k)
    #J_ = J_ + np.sum(err)* P_ @ R

# %%
    
PS_fF = PS_fF - xP * k;
w_fF = w_fF - (z_fF - f_fF) * k;
        
J_err = (np.dot(J,r) - np.dot(Jd,rd) 
								-np.dot(w_targ,targ[t:(t+1),:].T) - np.dot(w_hint, hints[t:(t+1),:].T))
# ...need FULL-FORCE

xP = PS * r;
k = (1 + r' * xP)\xP';

PS = PS - xP * k;
w = w - (z - f) * k;