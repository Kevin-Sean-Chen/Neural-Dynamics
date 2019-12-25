# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 13:47:51 2019

@author: kevin
"""

import numpy as np
import numpy.matlib
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC
#import Adaptive_Subunits

sns.set_style("white")
sns.set_context("talk")

# %% parameters
dt = 0.1
T = 1000
time = np.arange(0,T,dt)
lt = len(time)
N = 20  #number of neurons
x = np.zeros((N,lt))  #neural activity
x[:,0] = np.random.randn(N)  #initial activity
v = np.random.randn(N)  #input connection
v = v/np.linalg.norm(v)  #unit norm input strength
###stimuli
s = np.zeros((N,lt)) #input to the network
s = np.random.randn(N,lt)
s = ndimage.uniform_filter1d(s, 100, 1)  #smooth in time (can do space too)
eps = 0.1  #noise level
W = np.zeros((N,N,lt))  #all weights through time
W[:,:,0] = np.random.randn(N,N)*0.01 #inital connection

###symmetrical
tempM = np.random.randn(N,N)
tempM = 0.5*(tempM + tempM.T)
W[:,:,0] = tempM*0.1

###functional feedforward
#tempM = np.random.randn(1,N)
#T = np.dot(tempM.T,tempM)
#uu,ss,vv = np.linalg.svd(T)
#tempM2 = np.random.randn(N,N)
#ss = np.diag(ss) + np.triu(tempM2, k=1)*50  #with feedforward term in Schur decomposition
#T = uu @ ss @ vv
#T = T/N
#W[:,:,0] = T*0.1

eta = 0.05
# %% neural dynamics
for tt in range(0,lt-1):
    ###continuous time
#    x[:,tt+1] = x[:,tt] + dt*(W[:,:,tt] @ x[:,tt] + v*s[:,tt] + eps*np.random.randn(N))
#    W[:,:,tt+1] = W[:,:,tt] + dt*(-1/100*W[:,:,tt] + eta*np.outer(x[:,tt],x[:,tt]))
    ###discrete time
    x[:,tt+1] = W[:,:,tt] @ x[:,tt] + v*s[:,tt] + eps*np.random.randn(N)
    W[:,:,tt+1] = (1-eta)*W[:,:,tt] - eta*np.outer(x[:,tt],x[:,tt])
    
# %% Fisher information
Wst = np.squeeze(W[:,:,0])  #temporary for fixed W
C = eps*np.array([np.linalg.matrix_power(Wst,kk) @ np.linalg.matrix_power(Wst,kk).T for kk in range(0,W.shape[2])]).sum(axis=0)
Cn = np.linalg.pinv(C)
ks = 20
Jk = np.zeros(ks)
for kk in range(1,ks+1):
    Wk = np.linalg.matrix_power(Wst,kk)
    Jk[kk-1] = v.T @ Wk.T @ Cn @ Wk @ v
    
plt.plot(Jk,'-o')

# %% Fisher information w/ dynamic W
C = eps*np.array([np.linalg.matrix_power(W[:,:,kk],kk) @ np.linalg.matrix_power(W[:,:,kk],kk).T for kk in range(0,W.shape[2])]).sum(axis=0)
#C = np.cov(x)
Cn = np.linalg.pinv(C)
ks = 100
Jk = np.zeros(ks)
for kk in range(1,ks+1):
    Wk = np.linalg.matrix_power(W[:,:,kk],kk)
    Jk[kk-1] = v.T @ Wk.T @ Cn @ Wk @ v
    
plt.figure()
plt.plot(Jk,'-o')