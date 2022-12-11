# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 16:04:38 2021

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.special import gammaln
import sklearn
from sklearn.metrics import r2_score
import scipy.spatial as sps

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=60) 
matplotlib.rc('ytick', labelsize=60) 

# %% Kinetic Ising model
###############################################################################
# %% Stimuli
T = 3000
D = 3
smooth = 10
noise = np.random.randn(T,D)
X = noise.copy()
for ii in range(D):
    X[:,ii] = np.convolve(noise[:,ii],np.ones(smooth),'same')/smooth *1
    #X[:,ii] = np.convolve(noise[:,1],np.ones(smooth),'same') + np.random.randn(T)*3  #correlated

# %% Network settings
N = 100
g = 1
Phi = np.random.randn(D,N)
J = np.random.randn(N,N)/np.sqrt(N)*g
#J = J-np.diag(J)
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

kbT = .1
S = Kinetic_Ising(X,Phi,J,kbT)

plt.figure()
plt.imshow(S, aspect='auto')
plt.xlabel('$time$',fontsize=40)
plt.ylabel('$cell$',fontsize=40)

# %%
###############################################################################
# %%
#S[S<0] = 0
#m = np.mean(S, axis=1)
#ds = S - m[:,None]
#C = np.cov(ds.T, rowvar=False, bias=True)
#Cinv = np.linalg.inv(C)
#W = np.random.randn(N,N)
#its = 100
#for nn in range(N):
#    s1 = S[nn,:]
#    H = s1.copy()
#    for ii in range(its):
#        dH = H - np.mean(H)
#        Hs_avg = np.dot(ds, dH)/T
#        wi = np.dot(Hs_avg.T, Cinv)   #(iii)
#        H = np.dot(wi, S)   #(i)
#        S_H = np.tanh(H)
#        
#        H *= np.divide(s1, S_H, out=np.ones_like(s1), where=S_H!=0)   #(ii)
#    W[nn,:] = wi
#    
#plt.figure()
#plt.plot(J.reshape(-1), W.reshape(-1), 'o')
    

# %%
def fem(s):
    l,n = np.shape(s)
    m = np.mean(s[:-1],axis=0)
    ds = s[:-1] - m
    l1 = l-1

    c = np.cov(ds,rowvar=False,bias=True)
    c_inv = np.linalg.inv(c)
    dst = ds.T

    W = np.empty((n,n)) #; H0 = np.empty(n)
    
    nloop = 100

    for i0 in range(n):
        s1 = s[1:,i0]
        h = s1
        cost = np.full(nloop,100.)
        for iloop in range(nloop):
            h_av = np.mean(h)
            hs_av = np.dot(dst,h-h_av)/l1
            w = np.dot(hs_av,c_inv)
            #h0=h_av-np.sum(w*m)
            h = np.dot(s[:-1,:],w[:]) # + h0
            
            s_model = np.tanh(h)
            cost[iloop]=np.mean((s1[:]-s_model[:])**2)
                        
            if cost[iloop] >= cost[iloop-1]: break
                       
            h *= np.divide(s1,s_model, out=np.ones_like(s1), where=s_model!=0)
            #t = np.where(s_model !=0.)[0]
            #h[t] *= s1[t]/s_model[t]
            
        W[i0,:] = w[:]
        #H0[i0] = h0
    return W #,H0 

WW = fem(S.T)
plt.figure()
plt.plot(J.reshape(-1), WW.reshape(-1), 'o')

