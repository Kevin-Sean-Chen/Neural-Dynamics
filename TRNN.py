# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 01:14:51 2022

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %% Tensor-RNN setting
N = 200  # number of neurons
k = 10  # temporal kernel
g = 1.5  # scaling strength
M = np.random.randn(N,N,k)*g*1/np.sqrt(N)  # tensor network

# %% construct more stable network
taus = np.random.rand(N**2)
taus = taus*k  # kernel time scale
amps = np.random.rand(N**2)
amps = amps*2-1  # kernel amplitude
ii = 0
temp = np.arange(k)
for i in range(N):
    for j in range(N):
        M[i,j,:] = M[i,j,0]*amps[ii]*np.exp(-temp/taus[ii])

def NL(x):
    nl = np.tanh(x)
#    nl = 1/(1+np.exp(x))
#    nl = np.exp(x)
    return nl

def spk(x):
    p = np.random.rand(x.shape[0])
    s = x*0
    s[x>p] = 1
    return s

W = np.random.randn(N,k)  # readout kernel

# %% dynamic target
T = 500
dt = 0.1
time = np.arange(0,T,dt)
lt = len(time)

###target pattern
amp = 0.7;
freq = 1/60;
rescale = 1.2
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*time*rescale) + \
    (amp/2.0)*np.sin(2.0*np.pi*freq*time*rescale) + \
    (amp/6.0)*np.sin(3.0*np.pi*freq*time*rescale) + \
    1*(amp/3.0)*np.sin(4.0*np.pi*freq*time*rescale)
ft = ft/.7
plt.figure()
plt.plot(time,ft)

# %% FORCE learning
# initialization
learn_every = 1
alpha = 1.
wo = W*1#np.zeros((N,k))
dw = np.zeros((N,k))
xt = np.zeros((N,lt))
x0 = np.random.randn(N,k)
xt[:,:k] = x0
rt = xt*1
zt = np.zeros(lt)
P = (1.0/alpha)*np.eye(N)

# dynamics
for tt in range(k,lt):
#    rt[:,tt] = NL(np.einsum('ijk,jk->i',  M, xt[:,tt-k:tt]))  # kernel, network, and nonlinear
#    xt[:,tt] = xt[:,tt-1] + dt*(-xt[:,tt-1] + rt[:,tt])
    rt[:,tt] = NL(xt[:,tt-k:tt])[:,0] #+ np.random.randn()*0.2  # kernel, network, and nonlinear
    xt[:,tt] = np.einsum('ijk,jk->i',  M, rt[:,tt-k:tt])
    zt[tt] = np.sum(np.einsum('ik,ik->i',  wo, rt[:,tt-k:tt]))

    ### add IRLS here~~
    # learning
    if np.mod(tt, learn_every) == 0:
        kk = (P @ rt[:,tt])[:,None]
        rPr = rt[:,tt][:,None].T @ kk
        c = 1.0/(1.0 + rPr)
        P = P - (kk @ kk.T) * c  #projection matrix
    	
        # update the error for the linear readout
        e = zt[tt] - ft[tt] # error term
	
    	# update the output weights
        dw = -(e*kk*c)#[:,None]
        wo = wo + dw
        
        # update the internal weight matrix ... need to fix this for kernels!!
#        M = M + np.repeat(dw,N,1).T[:,:,None]
        # np.einsum('ij,jk->ijk',  np.tensordot(dw,dw.T,axes=1), dw)
        #np.repeat(dw,N,1).T
    
    
    ### ... low-rank M
    
plt.figure()
plt.imshow(rt, aspect='auto')

# %% testing

# %%
###############################################################################
# target pattern
def sigmoid(x):
    return 1/(1+np.exp(x))
V = 10
H = 5
N = V+H
T = 1000
px = np.random.randn(V,T)
for vv in range(V):
    px[vv,:] = sigmoid(np.convolve(px[vv,:],np.ones(20),'same'))
temp = np.random.rand(V,T)
X = px*0
X[px>temp] = 1

# %%
def NL_spk(x):
    return np.exp(x)
def NL_r(x):
    return 10/(1+np.exp(-x))
N = 30
dt = 0.1
wind = 10
xx = np.arange(wind)
tau_h = 1
h = np.flipud(-.1*np.exp(-xx/tau_h)[:,None])[:,0]
v1,v2 = np.random.randn(N,1),np.random.randn(N,1)
v1[v1>0] = 1
v1[v1<0] = -1
v2[v2>0] = 1
v2[v2<0] = -1
J = np.random.randn(N,N)/np.sqrt(N)*.1 + (v1 @ v1.T)/N*.5 + (v2 @ v2.T)/N*.5 + (v1 @ v2.T)/N*.5
x0 = np.random.rand(N)
yt = np.zeros((N,T))
xt = yt*0
xt[:,0] = x0
for tt in range(wind,T):
    xt[:,tt] = yt[:,tt-wind:tt] @ h*1 + 0.*J @ NL_r(xt[:,tt-1]) + 5.*J@yt[:,tt-1]
    yt[:,tt] = np.random.poisson(NL_r(xt[:,tt])*dt)

plt.figure()
plt.imshow(yt,aspect='auto')
# %%
#def neural_model(x,theta):
#    current = theta.T @ x
#    sig = sigmoid(current)
#    hidden = sig[-H:]
#    ht = hidden*0
#    ht[hidden>np.random.rand(H)] = 1
#    return ht
#eta = 0.1
#kappa = 1
#alpha = 1
#theta_init = np.random.randn(N,N)

