#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:53:58 2019

@author: kschen
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
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

def Lorenz_model(T,dt):
    """
    Samples with ran length T and time step dt
    """
    lt = int(T/dt)
    Xt = np.zeros((lt,3))
    Xt[0,:] = np.random.randn(3)*10  #initial condition within the dynamic range
    for tt in range(lt-1):
        dx,dy,dz = Lorenz_dt(Xt[tt,:], dt)
        Xt[tt+1,:] = Xt[tt,:]+[dx,dy,dz]
    return Xt
    
def Lorenz_plot(tr):
    """
    Plot 3D trajectory given a 3XT matrix of 3D time series
    """
    ax = plt.axes(projection='3d')
    ax.plot(tr[:,0], tr[:,1], tr[:,2], '-b')
    
def Delay_embedding(X,K):
    """
    N dimentional time series X  (d x T) with K step embedding
    """
    T = X.shape[0]
    d = X.shape[1]
    ut = 50  #size of time blocks
    Y = []  #np.zeros((T-K+1, K*d))
    for kk in range(0,K):
        Y.append(X[(K-kk)*ut:T-kk*ut,0:d])
    return np.concatenate(Y,axis=1)

def Mode_decomp(Y,m):
    """
    Dynamic mode decomposition for the embedded time series
    Reconstruct with rank cutoff at m
    """
    uu,ss,vv = np.linalg.svd(Y,full_matrices=False)  #mode decomposition
    Y_ = uu[:,:m] @ np.diag(ss[:m]) @ vv[:m,:]
    return Y_
    
def NN_prediction(X,X_,tau):
    """
    Compare time series prediction along time window tau
    """
    MSE = np.sum((X-X_)**2)**0.5  #sum of MSE of trajectories
    return MSE

def DM_embedding(Y,ll):
    """
    Test for Diffusion map embedding of transition matrix with kerenel length ll
    """
    
    return

def Diffusion_kernel(xt,xtt,ll):
    """
    Two points xt transitioning to xtt with length scale ll
    """
    k = np.exp(np.linalg.norm((xt-xtt)**2)/ll**2) 
    return k

def find_centroids(Xt,c):
    """
    Take time series and return c number of centroids in d-dimension space
    """
    kmeans = KMeans(n_clusters=c).fit(Xt)
    return kmeans.cluster_centers_

def min_MSE(Xt,Xtt,theta):
    """
    Given kernel and the input-output state-space, return weights to minimize MSE
    """
    l_, centroids = theta
    phi = np.array([Diffusion_kernel(Xt,cc,l_) for cc in centroids])
    phi = phi/np.sum(phi)
    w = np.linalg.pinv(phi.T @ phi) @ phi.T @ Xtt
    return w

# %% analysis
T = 200
dt = 0.01
Es = np.zeros((40,8))
for kk in range(0,40):
    for mm in range(0,8):
        Xt = Lorenz_model(T,dt)
        Y = Delay_embedding(Xt,kk+1)
        Y_ = Mode_decomp(Y,mm+1)
        Es[kk,mm] = NN_prediction(Y,Y_,0)


