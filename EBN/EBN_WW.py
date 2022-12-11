# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:56:16 2020

@author: kevin
"""

#EBN_WW
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %% setup parameters
a = 270  #I/O parameters
b = 108
d = 0.154
gamma = 0.641  #kinetic coupling
taus = 100  #synaptic time scale
tauA = 2  #AMPA ionic time scale
#J = np.array([[0.2609, -0.0497],[-0.0497, 0.2609]])*1  #coupling
J = np.array([[0.1561, -0.264],[-0.264, 0.1561]])
I0 = 0.3255  #background current
I0 = 0.2346
sig_noise = 0.02  #noise strength

def H(a,b,d,x):
    """
    effective single-cell input– output relation H
    """
    hi = (a*x-b)/(1-np.exp(-d*(a*x-b)))
    return hi

# %% stimuli and initialize
dt = 0.1  #ms
T = 500
time = np.arange(0,T,dt)
lt = len(time)
S = np.zeros((2,lt))  #synaptic current
Is = np.zeros((2,lt))  #stimuli
In = np.zeros((2,lt))  #input noise
x = np.zeros((2,lt))  #total current
rs = np.zeros((2,lt))  #spike rate
S[:,0] = np.random.rand(2)*0.01
x[:,0] = np.random.rand(2)*0.01
# noise
for tt in range(lt-1):
    In[:,tt+1] = In[:,tt] + dt/tauA*(-In[:,tt]) + np.random.randn(2)*np.sqrt(tauA*sig_noise**2)*dt**2/tauA
# stimuli
mu0 = np.zeros(lt)  #time-dependent stimulus presentation
cc = 60  # % of coherence of stimuli
#cc[100:500] = c_
mu0[200:2200] = 30  #Hz
JA = 9.9*10**-4  #input strength
Is = JA*mu0*( 1 + np.vstack((+1*cc, -1*cc))/100 )  #input to two populations

# %% neural dynamics
for tt in range(1,lt):
    x[:,tt-1] = J @ S[:,tt-1] + I0 + In[:,tt-1] + Is[:,tt-1]  #current input
    rs[:,tt] = H(a,b,d,x[:,tt-1])  #spike rate
    S[:,tt] = S[:,tt-1] + dt*( -S[:,tt-1]/taus + (1-S[:,tt-1])*gamma*rs[:,tt-1] )  #synaptic dynamics
    
# %% plotting
plt.figure()
plt.subplot(211)
plt.title('coherence='+str(cc)+'%')
plt.plot(time,rs.T)
plt.xlabel('time (ms)')
plt.ylabel('firing rate (Hz)')
plt.subplot(212)
plt.plot(time,(Is+In).T)
plt.xlabel('time (ms)')
plt.ylabel('input Ii')

# %% iterations
rep = 10
cc = -70  # % of coherence of stimuli
for rr in range(rep):
    S = np.zeros((2,lt))  #synaptic current
    Is = np.zeros((2,lt))  #stimuli
    In = np.zeros((2,lt))  #input noise
    x = np.zeros((2,lt))  #total current
    rs = np.zeros((2,lt))  #spike rate
    S[:,0] = np.random.rand(2)*0.0
    x[:,0] = np.random.rand(2)*0.0
    # noise
    for tt in range(lt-1):
        In[:,tt+1] = In[:,tt] + dt/tauA*(-In[:,tt]) + np.random.randn(2)*np.sqrt(tauA*sig_noise**2)*dt**2/tauA
    # stimuli
    mu0 = np.zeros(lt)  #time-dependent stimulus presentation
    
    #cc[100:500] = c_
    mu0[200:2200] = 30  #Hz
    JA = 5.2*10**-4  #input strength
    Is = JA*mu0*( 1 + np.vstack((+1*cc, -1*cc))/100 )  #input to two populations    
    for tt in range(1,lt):
        x[:,tt-1] = J @ S[:,tt-1] + I0 + In[:,tt-1] + Is[:,tt-1]  #current input
        rs[:,tt] = H(a,b,d,x[:,tt-1])  #spike rate
        S[:,tt] = S[:,tt-1] + dt*( -S[:,tt-1]/taus + (1-S[:,tt-1])*gamma*rs[:,tt-1] )  #synaptic dynamics
    plt.plot(time,rs[0,:],'b-')

# %% store experiments
rep = 10
ccs = np.array([-60,-30,0,+30,+60])
stimuli_ = np.zeros((len(ccs),rep,lt))  #for all coherence level, repetitions, through time
records_ = np.zeros((len(ccs),rep,lt))
for ci,cc in enumerate(ccs):
    for rr in range(rep):
        S = np.zeros((2,lt))  #synaptic current
        Is = np.zeros((2,lt))  #stimuli
        In = np.zeros((2,lt))  #input noise
        x = np.zeros((2,lt))  #total current
        rs = np.zeros((2,lt))  #spike rate
        S[:,0] = np.random.rand(2)*0.0
        x[:,0] = np.random.rand(2)*0.0
        # noise
        for tt in range(lt-1):
            In[:,tt+1] = In[:,tt] + dt/tauA*(-In[:,tt]) + np.random.randn(2)*np.sqrt(tauA*sig_noise**2)*dt**2/tauA
        # stimuli
        mu0 = np.zeros(lt)  #time-dependent stimulus presentation
        
        mu0[200:2200] = 30  #Hz
        JA = 5.2*10**-4  #input strength
        Is = JA*mu0*( 1 + np.vstack((+1*cc, -1*cc))/100 )  #input to two populations    
        for tt in range(1,lt):
            x[:,tt-1] = J @ S[:,tt-1] + I0 + In[:,tt-1] + Is[:,tt-1]  #current input
            rs[:,tt] = H(a,b,d,x[:,tt-1])  #spike rate
            S[:,tt] = S[:,tt-1] + dt*( -S[:,tt-1]/taus + (1-S[:,tt-1])*gamma*rs[:,tt-1] )  #synaptic dynamics
        ### recording
        stimuli_[ci,rr,:] = Is[0,:]  #store input to I1
        records_[ci,rr,:] = rs[0,:]  #store output from I1 
        
# %% for randomize trials
temps = stimuli_.reshape(len(ccs)*rep,lt)
tempr = records_.reshape(len(ccs)*rep,lt)
perm = np.random.permutation(len(ccs)*rep)  #shuffle the simulated trials for continuous training
WW_s = temps[perm,:]
WW_r = tempr[perm,:]

