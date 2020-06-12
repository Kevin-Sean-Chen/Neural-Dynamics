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

def 

# %% EBN model
#dimensions
N = 50
T = 300
dt = 0.01
time = np.arange(0,T,dt)
lt = len(time)
input_dim = 3
output_dim = 3

#stimuli
A = 2  #amplitude
nn = 0.1  #noise
ct = np.zeros((input_dim,lt))  #command input time series
ct[:,1000:3000] = np.random.randn(1,2000)*1.
Xs = A*np.ones((output_dim,lt)) + np.random.randn((output_dim,lt))  #target dynamics

#biophysics
lamb_u = 20 #voltage time scale
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

#Dynamics
for tt in range(0,lt-1):
    spk[:,tt] = Threshold(us[:,tt], D, mu, vv)  #spiking process
    rec[:,tt] = np.matmul(D,rs[:,tt])
    err[:,tt] = Xs[:,tt] - rec[:,tt]  #error signal
    Ws = Ws + eta*dt*np.outer( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
    us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
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