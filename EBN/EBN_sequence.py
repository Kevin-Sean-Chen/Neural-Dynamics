# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:14:00 2020

@author: kevin
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %% functions
def Threshold(u, D, mu, vv):
    """
    Spiking threshold
    """
    output_dim, N = D.shape
    ss = np.zeros(N)
    Di = np.linalg.norm(D,axis=0)**2
    Ti = 0.5*(Di+mu+vv)  #threshold
    ss[np.where(u>Ti)[0]] = 1
    return ss

def Phi(x):
    """
    Sigmoind nonlinearity
    """
    return np.tanh(x)

def Seq(Ct, lt, wi):
    """
    Create sequence target with the output_dim of neurons tiling time lt with bump width
    """
    NN, lt = Ct.shape
    ttb = np.tile(np.arange(0,lt),(NN,1))  #take log for nonlinear time
    dbcenter = lt / (NN+int(lt/800)) # spacing between bumps
    width = wi*dbcenter # width of each bump
    bcenters = 1.*dbcenter + dbcenter*np.arange(0,NN)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)  #raise-cosine function formula
    temp = ttb - np.tile(bcenters,(lt,1)).T
    x = np.array([bfun(xx,width) for xx in temp]) 
    return x  #neuron x time sequence

# %% EBN model
#dimensions
N = 50
T = 150
dt = 0.01
time = np.arange(0,T,dt)
lt = len(time)
input_dim = N
output_dim = N

#stimuli
A = 2  #amplitude
nn = 0.01  #noise
wi = 10  #width of the bump
ct = np.zeros((input_dim,lt))  #command input time series
ct[:,10:] = np.random.randn(1,lt-10)*1.
ct_ = ct.copy()
ct_[:,10:] = 10*np.ones((1,lt-10)) + np.random.randn(1,lt-10)*1.
Xs = A*Seq(ct, lt, wi) + nn*np.random.randn(output_dim,lt)  #target dynamics
Xs_ = np.flipud(Xs)  #reverse sequence
lamb_u = 10 #voltage time scale
lamb_r = 2  #spiking rate time scale
k = 100.  #error coupling
eta = 0.2  #learning rate
mu = 10**-6  #L2 spiking penalty
vv = 10**-5  #L1 spiking penalty
sig = 0.0  #noise strength

#connectivity
Ws = np.random.randn(N,N)  #slow connection to be learned
D = np.random.randn(output_dim,N)  #output connections
Wf = D.T @ D + mu*np.eye(N)  #fast connections
F = D.copy()  #input connections  #F = np.random.randn(input_dim,N)  

#time series
us = np.zeros((N,lt))  #neural activities
us[:,0] = np.random.randn(N)*0.1
err = np.zeros((output_dim,lt))  #error signal
spk = np.zeros((N,lt))  #spiking activity
rs = np.zeros((N,lt))  #firing rate
rec = np.zeros((output_dim,lt))  #store the reconstruction

# %%
#Dynamics
for tt in range(0,lt-1):
    spk[:,tt] = Threshold(us[:,tt], D, mu, vv)  #spiking process
    rec[:,tt] = np.matmul(D,rs[:,tt])
    err[:,tt] = Xs_[:,tt] - rec[:,tt]  #error signal
    #Ws = Ws + eta*dt*np.outer( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
    us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + 2*np.matmul(F.T,ct_[:,tt]) - np.matmul(Wf,spk[:,tt]) \
      + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
    rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate

# %% contextual
#Dynamics
for tt in range(0,lt-1):
    spk[:,tt] = Threshold(us[:,tt], D, mu, vv)  #spiking process
    rec[:,tt] = np.matmul(D,rs[:,tt])
    err[:,tt] = Xs_[:,tt] - rec[:,tt]  #error signal
    Ws = Ws + eta*dt*np.outer( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
    us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct_[:,tt]) - np.matmul(Wf,spk[:,tt]) \
      + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
    rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate
    
# %% plotting
plt.figure()
plt.subplot(211)
plt.imshow(spk,aspect='auto')
plt.ylabel('Neurons')
plt.subplot(212)
plt.plot(time,rec[0,:].T,label='Reconstruct')
plt.plot(time,Xs[0,:].T,'--',label='Target')
plt.xlim([0,time[-1]])
plt.legend()
plt.xlabel('time')

# %%
plt.figure()
plt.subplot(311)
plt.title('spiking pattern')
plt.imshow(spk,aspect='auto')
plt.ylabel('Neurons')
plt.subplot(312)
plt.title('target')
plt.imshow(Xs,aspect='auto')
plt.ylabel('Neurons')
plt.subplot(313)
plt.title('reconstruct')
plt.imshow(rec,aspect='auto')
plt.ylabel('Neurons')
plt.xlabel('time')

# %% post analysis
window = np.ones(100)  #smoothing window
peak_loc = np.zeros(N)
for nn in range(N):
    rate = np.convolve(window,spk[nn,:])
    peak_loc[nn] = np.argmax(rate)
sorted_loc = np.argsort(peak_loc)
plt.figure()
plt.title('sorted spikes')
plt.imshow(spk[sorted_loc,:],aspect='auto')
plt.ylabel('Neurons')

# %% generative
plt.figure()
plt.subplot(221)
plt.plot(ct[0,:])
plt.ylabel('input cue')
plt.subplot(223)
plt.title('sorted spikes')
plt.imshow(spk[sorted_loc,:],aspect='auto')
plt.ylabel('Neurons')

for tt in range(0,lt-1):
    spk[:,tt] = Threshold(us[:,tt], D, mu, vv)  #spiking process
    rec[:,tt] = np.matmul(D,rs[:,tt])
    err[:,tt] = Xs_[:,tt] - rec[:,tt]  #error signal
    us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct_[:,tt]) - np.matmul(Wf,spk[:,tt]) \
      + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
    rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate

window = np.ones(100)  #smoothing window
peak_loc = np.zeros(N)
for nn in range(N):
    rate = np.convolve(window,spk[nn,:])
    peak_loc[nn] = np.argmax(rate)
sorted_loc = np.argsort(peak_loc)

plt.subplot(222)
plt.plot(ct_[0,:].T)
plt.ylabel('input cue')
plt.subplot(224)
plt.title('sorted spikes')
plt.imshow(spk[sorted_loc,:],aspect='auto')
plt.ylabel('Neurons')
plt.xlabel('time')