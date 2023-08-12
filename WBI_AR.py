# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 23:38:40 2023

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 
import h5py

# %% load dynamics Keito
fname = 'WT_Stim'
f = h5py.File('C:/Users/kevin/Downloads/'+fname+'.mat')
print(f[fname].keys())
all_trace = []
for ii in range(len(f[fname]['traces'])):
    temp = f[fname]['traces'][ii,0]
    all_trace.append(f[temp][:])

plt.figure()
plt.imshow(all_trace[0],aspect='auto')

# %% load recoding Leifer
from scipy import io
fname = 'C:/Users/kevin/Downloads/WBI_data/heatDataMS_3.mat'
mat = io.loadmat(fname)
behavior = mat['behavior']
Beh = behavior['v'][0][0].squeeze()
R2 = mat['Ratio2']

### preprocss nan
pos = np.argwhere(np.isnan(Beh))
Beh[pos] = np.nanmean(Beh)
Beh = Beh[None,:]

r2_ = R2.reshape(-1)
pos = np.argwhere(np.isnan(r2_))
r2_[pos] = np.nanmean(r2_)
R2 = r2_.reshape(R2.shape[0],R2.shape[1])

# %% neuron-by-neuron
###############################################################################
# %%
def tau_var(signal, full_acf=False):
    ### ACF function
    n = len(signal)
    mean = np.mean(signal)
    centered_signal = signal - mean
    sig_var = np.var(signal)
    autocorr = np.correlate(centered_signal, centered_signal, mode='full') / (sig_var * n)
    acf = autocorr[n-1:]
    lags = np.arange(0, len(acf))
    ### fit exp to ACF
    try:
        popt, _ = curve_fit(exponential_fit, lags, acf)
        tau = popt[1]
    except RuntimeError:
        tau = np.NaN
    if full_acf is True:
        return tau, sig_var, acf
    else:
        return tau, sig_var

def exponential_fit(x, A, tau):
    return A * np.exp(-x / tau)

def AR_solve(x1, x0):
    n,T = x1.shape
    inv_xxt = np.linalg.pinv(x0@x0.T)
    A = inv_xxt @ x0 @ x1.T
    eps = np.mean((x1 - A@x0)**2)**0.5/T
    return A, eps

# %% fit ACF and VAR
dt = 1
T = 200
T_init = 2200 #900, 1100, 1300, 1500
N = R2.shape[0]
VCs = np.zeros((N,T,2))
for tt in range(0,T):
    for nn in range(0,N):
        signal = R2[nn,T_init+tt:T_init+tt+T]
        tau, var = tau_var(signal)
        VCs[nn,tt,:] = np.array([tau, var])
        
# %%
plt.figure()
plt.scatter(VCs[:,:,0].reshape(-1), VCs[:,:,1].reshape(-1), alpha=0.05)
plt.xlabel(r'$\tau$', fontsize=30)
plt.ylabel('$\sigma^2$', fontsize=30)
plt.xlim([0,25])

# %% populational AR
###############################################################################
dt = 1
T = 200
T_init = 2200 #900, 1100, 1300, 1500 ## base, pre, during, post; worm 3: 1100 vs. 1400!, 2200
N = R2.shape[0]
As = np.zeros((N,N,T))
eps = np.zeros(T)
var_t = np.zeros((N,T))
for tt in range(0,T):
    x1,x0 = R2[:,T_init+tt:T_init+tt+T], R2[:,T_init+tt+dt:T_init+tt+T+dt]
    As[:,:,tt], eps[tt] = AR_solve(x1,x0)
    var_t[:,tt] = np.var(x0,1)
    
# %%
eigs = np.zeros((N,T))
critt = np.zeros(T)
for tt in range(0,T):
    A_ = As[:,:,tt]
    uu,vv = np.linalg.eig(A_)
    eigs[:,tt] = uu
    critt[tt] = np.sum(np.real(uu)>.95)/N

# %%
plt.figure()
plt.subplot(121)
plt.plot(critt)
plt.xlabel('time', fontsize=30)
plt.ylabel('fraction of critical modes', fontsize=30)
plt.ylim([0.01, 0.5])
plt.subplot(122)
plt.plot(eps)
plt.xlabel('time', fontsize=30)
plt.ylabel('AR error', fontsize=30)

# %%
plt.figure()
plt.subplot(121)
plt.imshow(np.abs(np.real(eigs)), aspect='auto',vmin=0, vmax=1.1)
plt.colorbar()
plt.xlabel('time', fontsize=30)
plt.ylabel('modes', fontsize=30)
plt.subplot(122)
#plt.imshow(var_t, aspect='auto')
plt.plot(np.mean(var_t,1))
plt.ylabel('var')

# %% checking idea with ground-truth
###############################################################################
# %% Ideas
# idea 1: can do sliding window, then clustering in the mode space
# idea 2: experimentally distinguish crictical and non-critical (spontaneuous and driven?) transitions
# Idea 3: use SSM (switching LDS?) to pull out states, then find states with different stability

# %% I1
### model ground-truth with switching LDS
### test if A and Sigma alters in time
### sliding window of LL?
# %%
T = 1000
N = 50
def make_stable_LDS():
    A = np.random.randn(N,N)
    uu,vv = np.linalg.eig(A)
    s = np.real(uu) / np.max(np.abs(np.real(uu)))*0.98
    s[np.real(s)<0] *= -1
    A = np.real(vv @ np.diag(s) @ np.linalg.inv(vv))
    Q = np.random.randn(N)
    Q = 0.01*(Q.T@Q + np.eye(N))
    return A, Q
A1, Q1 = make_stable_LDS()
A2, Q2 = make_stable_LDS()

xt = np.zeros((N,T))
xt[:,0] = np.random.randn(N)
for tt in range(T-1):
    if tt < T//2:
        xt[:,tt+1] = A1 @ xt[:,tt] + np.random.multivariate_normal(np.zeros(N), Q1)
    else:
        xt[:,tt+1] = A2 @ xt[:,tt] + np.random.multivariate_normal(np.zeros(N), Q2)
        
plt.figure()
plt.plot(xt.T)

# %% analysis
#xt = Xt*1
T_init = 200
T = 200
T_final = 800
dt = 1
deT = T_final-T_init
As = np.zeros((N,N,deT))
Sigs = np.zeros((N,N,deT))
eps = np.zeros(deT)
for tt in range(0,deT):
    x1,x0 = xt[:,T_init+tt:T_init+tt+T], xt[:,T_init+tt+dt:T_init+tt+T+dt]
    A_, eps[tt] = AR_solve(x1,x0)
    As[:,:,tt] = A_
    Sigs[:,:,tt] = (x1 - A_@x0) @ (x1 - A_@x0).T
    
# %%
eigs = np.zeros((N,deT))
eigs_n = np.zeros((N,deT))
critt = np.zeros(deT)
for tt in range(0,deT):
    A_ = As[:,:,tt]
    uu,vv = np.linalg.eig(A_)
    eigs[:,tt] = uu
    critt[tt] = np.sum(np.real(uu)>.95)/N
    S_ = Sigs[:,:,tt]
    uu,vv = np.linalg.eig(S_)
    eigs_n[:,tt] = uu/np.max(np.abs(uu))

# %%
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.imshow((np.real(eigs)), aspect='auto',vmin=0, vmax=1.1)
plt.colorbar()
plt.xlabel('time', fontsize=30)
plt.ylabel('modes', fontsize=30)
plt.subplot(122)
plt.imshow(np.log(np.real(eigs_n)), aspect='auto')#,vmin=0, vmax=1.1)
plt.colorbar()
plt.xlabel('time', fontsize=30)
plt.ylabel('noise', fontsize=30)

plt.figure()
plt.subplot(121)
plt.plot(critt)
plt.xlabel('time', fontsize=30)
plt.ylabel('fraction of critical modes', fontsize=30)
plt.ylim([0.01, 0.5])
plt.subplot(122)
plt.plot(eps)
plt.xlabel('time', fontsize=30)
plt.ylabel('AR error', fontsize=30)

# %% try low-D bifurcation, noise or input-driven
#dx = r + x - x**3
T = 1000
r = .3
D = 5
N = 50
sig = 0.1
dt = 0.1
xt = np.zeros(T)
Xt = np.zeros((N,T))
rt = Xt*1
#M = np.random.randn(N)
v1, v2 = np.random.randn(N), np.random.randn(N)
M = np.random.randn(N,N)*5 + (np.outer(v1,v1) + np.outer(v2,v2))*1
for tt in range(T-1):
    ### 1D latent
#    xt[tt+1] = xt[tt] + dt*(r + xt[tt] - xt[tt]**3) + np.sqrt(dt*D)*np.random.randn()
#    Xt[:,tt] = M* xt[tt] + np.random.randn(N)*sig
    ### attractor network
    Xt[:,tt+1] = Xt[:,tt] + dt*(-Xt[:,tt] + M @ rt[:,tt]) + np.sqrt(dt*D)*np.random.randn(N)
    rt[:,tt+1] = np.tanh(Xt[:,tt+1])#1/(1+np.exp(Xt[:,tt]))
plt.figure()
plt.subplot(121)
plt.plot(v1@Xt)
plt.subplot(122)
plt.imshow(Xt,aspect='auto')
