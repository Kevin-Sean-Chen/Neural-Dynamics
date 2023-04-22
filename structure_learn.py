#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:08:19 2023

@author: kschen
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import itertools
import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)

# %% creating graphs
N = 3
n_comb = N*(N-1)
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=n_comb))
nG = len(combinations)

def vec2mat(vec):
    mat = np.zeros((N,N))
    row_indices, col_indices = np.triu_indices(N,k=1)
    mat[row_indices, col_indices] = vec[:len(vec)//2]
    row_indices, col_indices = np.tril_indices(N,k=-1)
    mat[row_indices, col_indices] = vec[len(vec)//2:]
    return mat

G = []
for ii in range(nG):
    G.append(vec2mat(combinations[ii]))
    
# %% neural patterns
T = 200
dt = 0.1
tau = 1
syn = 1.
noise = .01
frozen_noise = np.random.randn(N,T)*noise

def neuraldynamics(W):
    rt = np.zeros((N,T))
    for tt in range(T-1):
        rt[:,tt+1] = rt[:,tt] + dt*(-rt[:,tt]/tau + syn*W @ np.tanh(rt[:,tt])) + frozen_noise[:,tt]
    return rt

rG = []
for ii in range(nG):
    rG.append(neuraldynamics(G[ii]))
    
# %% similarity
simM = np.zeros((nG, nG))
for ii in range(nG):
    for jj in range(nG):
        r1,r2 = neuraldynamics(G[ii]), neuraldynamics(G[jj])
        simM[ii,jj] = np.corrcoef(r1.reshape(-1),r2.reshape(-1))[0][1]
        
# %% try to finc M matrix
def cluster_corr(corr_array, inplace=False):
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

# %%
plt.imshow(simM, aspect='auto')
plt.imshow(cluster_corr(simM), aspect='auto')
plt.xlabel('gaph ID',fontsize=30)
plt.ylabel('gaph ID',fontsize=30)

# %% graph analysis
graph_dis = np.zeros_like(simM)
for ii in range(nG):
    for jj in range(nG):
        graph_dis[ii,jj] = np.linalg.norm(G[ii]-G[jj])

plt.plot(simM, graph_dis,'k.',alpha=0.5)
plt.ylim([0.8,2.3])
plt.ylabel('Euclidean',fontsize=30)
plt.xlabel('correlation',fontsize=30)

# %% find features
def obj(ww):
    M = ww.reshape(n_comb, n_comb)
    loss = np.linalg.norm(-simM - dg@M@dg.T)
    return loss

def obj2(ww):
    M = ww.reshape(n_comb, n_comb)
    loss = 0
    for ii in range(nG):
        loss = loss + np.linalg.norm(-simM - dg[ii,:,:]@M@dg[:,jj,:].T)
    return loss

def off_diag(mat):
    return mat[np.where(~np.eye(mat.shape[0], dtype=bool))]

dg = np.zeros((nG, nG, n_comb))  # not nG**2...
kk = 0
for ii in range(nG):
    for jj in range(nG):
        vi,vj = off_diag(G[ii]) , off_diag(G[jj])
        # dg[kk,:] = vi-vj
        dg[ii,jj,:] = vi - vj
        kk = kk+1
        
# %%
init_w = np.zeros(n_comb**2)
result = minimize(obj2, init_w)

# %%
M = result.x.reshape(n_comb, n_comb)
m_dis = np.zeros_like(simM)
for ii in range(nG):
    for jj in range(nG):
        vi,vj = off_diag(G[ii]) , off_diag(G[jj])
        ddg = vi-vj
        m_dis[ii,jj] = ddg @ M @ ddg

plt.plot(simM, m_dis,'k.',alpha=0.5)
plt.ylabel('learned M',fontsize=30)
plt.xlabel('correlation',fontsize=30)

# %% decomposition
# L = np.linalg.cholesky(M)
L, low = cho_factor(M)

plt.figure()
plt.title('M',fontsize=30)
plt.imshow(M)
plt.figure()
plt.title('L',fontsize=30)
plt.imshow(L)