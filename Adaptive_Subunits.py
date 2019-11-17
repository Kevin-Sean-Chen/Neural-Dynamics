#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:30:59 2019

@author: kschen
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")
#%matplotlib inline

# %% parameters
#time
dt = 0.1  #in units of 10 ms for speed
T = 1000
time = np.arange(0,T,dt)

#time scales
tau_c = 2  #fast time scael in 10 ms
tau_s = 5000  #slow time scale
gamma = 0.1  #overlapping of subunit receptive field
p0 = 100  #input synaptic strength
q = 0.5  #probability to connect subunits and neuron
#q0 = 1  #strength of subunit-neuron synapse

#connectivity (image to subunits)
K = 50  #number of neurons
m_avg = 2  #number of subunits per neurons
M = int(K*m_avg)  #total number of subunits
Q = np.random.rand(K,M)  #subunit-neuron conection
Q[Q>q] = 1
Q[Q<=q] = 0
for kk in range(K):  #check for input unity
    sumsyn = sum(Q[kk,:])
    if sumsyn==0:
        Q[kk,:] = 1/M
    else:
        Q[kk,:] = Q[kk,:]/sumsyn
        
#adding inhibitory connnections
pi = 0.5
signM = np.random.rand(Q.shape[0],Q.shape[1])
signM[signM>pi] = 1
signM[signM<=pi] = -1
Q = Q*signM

m = 5  #subunits per stimulus pool
N = int(M/m)  #possible unique images
P = np.ones((N,M))*gamma  #image-subunit connection
temp = 0
for nn in range(N):
    #P[nn,temp:temp+m] = 1
    P[nn,np.random.randint(0,M,(m))] = p0
    temp = temp + m

# %% stimuli
#marks
fnum = 4  #number of unique frames in a sequence
dur = int(20/dt)  #duration of each frame in ms
L = 10  #repeating the sequence
mark = np.arange(0,fnum,1)
mark2 = np.repeat(mark,dur,axis=0)
marks = np.matlib.repmat(np.expand_dims(mark2,axis=1),L,1).reshape(-1)
### subs
marks_ = np.matlib.repmat(np.expand_dims(mark2,axis=1),5,1).reshape(-1)
marksub = np.array([0,1,4,3])
marksub = np.repeat(marksub,dur,axis=0)
marks = np.concatenate( (np.concatenate((marks_, marksub)),marks_ ) )
###
if len(marks)>len(time):
    marks = marks[:len(time)]
else:
    marks = np.concatenate((marks,-np.ones(len(time)-len(marks))))


# %% dynamics
xs = np.zeros((M,len(time)))  #subunit activity
alphas = np.zeros((M,len(time)))  #subunit adaptation
ys = np.zeros((K,len(time)))  #neurons
xs[:,0] = 0.1
alphas[:,0] = .1
ys[:,0] = 0.1
### w/ inhibitory feedforward
#kk = np.random.randn(subu)

for tt in range(0,len(time)-1):
    I_index = int(marks[tt])  #stimulus index
    if I_index<0:
        xs[:,tt+1] = xs[:,tt] + dt*(1/tau_c)*(-xs[:,tt] + alphas[:,tt]*0)  #subunit
        alphas[:,tt+1] = alphas[:,tt] + dt*(1/tau_s)*((1-alphas[:,tt]) - alphas[:,tt]*0)  #adaptation
    else:
        xs[:,tt+1] = xs[:,tt] + dt*(1/tau_c)*(-xs[:,tt] + alphas[:,tt]*P[I_index,:])  #subunit
        alphas[:,tt+1] = alphas[:,tt] + dt*(1/tau_s)*((1-alphas[:,tt]) - alphas[:,tt]*P[I_index,:])  #adaptation
    ys[:,tt+1] = ys[:,tt] + dt*(1/tau_c)*(-ys[:,tt] + np.matmul(Q,xs[:,tt]))  #neurons

# %% plotting heat
plt.imshow(ys,aspect='auto')
plt.xlabel('time (in 10ms)')
plt.figure()
plt.plot(time,ys.T)

# %% PSTH
window = np.arange(2500,6500)
ID = 23
plt.figure()
plt.subplot(211)
plt.plot(time[window],marks[window])
plt.xticks([], [])
plt.ylabel('frames')
plt.subplot(212)
plt.plot(time[window],ys[ID,window].T)
plt.xlabel('time')
plt.ylabel('dF/F')


