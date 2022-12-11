#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:23:08 2022

@author: kschen
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import pandas as pd

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)

# %% AP circuit
# biophysics
N = 2  # two-neuron system
J = np.array([[2,-2],[-6,6]])*.5  # simple circuit that captures balanced output when both presented
J = np.array([[2,-2],[-3,3]])*.5
tau = 1  # time constant
sig = .5  # noise strength
tauB = 2  # behavioral filtering
def NL(x):
    return 1/(1+np.exp(-x))  # nonlinearity

# time vector
dt = 0.1
T = 500
time = np.arange(0,T,dt)
lt = len(time)

# stimulus
dur = 10  # stimulus duration
A = 2  # stimulus amplitude
dur_i = int(dur/dt)
It = np.zeros(lt)
It[int(lt/2):int(lt/2)+dur_i] = A
inpvec = np.array([1,0])

# %% simulation
# initialization
Vt = np.zeros((N,lt))  # voltage through time
Bt = np.zeros((N,lt))  # behavior through time

# dynamics
for tt in range(1,lt):
    Vt[:,tt] = Vt[:,tt-1] + dt/tau*(-Vt[:,tt-1] + J @ NL(Vt[:,tt-1]) + inpvec*It[tt-1]) + np.random.randn(2)*np.sqrt(sig*dt)  # circuit dynamics
    pos = np.argmax(Vt[:,tt])
    # Bt[pos,tt] = 1  # assigne behavior
    Bpuls = np.zeros(2)
    Bpuls[pos] = 1
    Bt[:,tt] = Bt[:,tt-1] + dt/tauB*(-Bt[:,tt-1] + Bpuls)  # behavioral slow dynamics
    
# %% plotting
plt.figure()
plt.subplot(211)
plt.plot(Vt.T)
plt.plot(It)
plt.subplot(212)
plt.plot(Bt.T)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### Scanning conditions
# %%
vecs = np.array([[0,0],
                 [1,0],
                 [0,1],
                 [1,1]])  # stimulus conditions
conds = vecs.shape[0]
rep = 50  # repeat trials
wind = dur_i*3  # duration to find behavior
nb = 2  # behavioral state definistions (# an art here...)
record = np.zeros((conds, nb, rep))  # recodings: stimulus x behavior x repeats
pause = 2  # pause if one state is not much larger than the other

for rr in range(rep):
    print(rr)
    for cc in range(conds):
        inpvec = vecs[cc,:]  # stimulus condition
        # initialization
        Vt = np.zeros((N,lt))  # voltage through time
        Bt = np.zeros((N,lt))  # behavior through time
        # dynamics
        for tt in range(1,lt):
            Vt[:,tt] = Vt[:,tt-1] + dt/tau*(-Vt[:,tt-1] + J @ NL(Vt[:,tt-1]) + inpvec*It[tt-1]) + np.random.randn(2)*np.sqrt(sig*dt)  # circuit dynamics
            pos = np.argmax(Vt[:,tt])
            # Bt[pos,tt] = 1  # assigne behavior
            Bpuls = np.zeros(2)
            Bpuls[pos] = 1
            Bt[:,tt] = Bt[:,tt-1] + dt/tauB*(-Bt[:,tt-1] + Bpuls)  # behavioral slow dynamics
        
        ### analysis
        Bres = np.sum(Bt[:,int(lt/2):int(lt/2)+wind], 1)
        Bres = np.argmax(Bres)
        record[cc,Bres,rr] = 1  # assigned
        
# %% anbalsis
rev_ratio = np.zeros(conds)  # three conditions
for_ratio = np.zeros(conds)
for cc in range(conds):
    rev_ratio[cc] = np.sum(record[cc,0,:])/rep
    for_ratio[cc] = np.sum(record[cc,1,:])/rep

beh = ['backward','forward']
conditions = ['no stim','head','tail','head+tail']
fig, axs = plt.subplots(1, conds, constrained_layout=True)
for cc in range(conds):
    axs[cc].bar(beh,[rev_ratio[cc],for_ratio[cc]])
    axs[cc].set_title(conditions[cc])