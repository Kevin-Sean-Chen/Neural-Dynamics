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
T = 2000
time = np.arange(0,T,dt)

#time scales
tau_c = 5  #fast time scael in 10 ms
tau_s = 1000  #slow time scale
gamma = 0.05  #overlapping of subunit receptive field
p0 = 100  #input synaptic strength
q = 0.5  #sparsity to connect subunits and neuron
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
pi = 0.1
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

def NL(x):
    """
    Nonlinearity for output neuron
    """
    return np.array( [max(xx,0) for xx in x] )  #ReLu

# %% stimuli
#marks
fnum = 8  #number of unique frames in a sequence
dur = int(20/dt)  #duration of each frame in ms
L = 20  #repeating the sequence
mark = np.arange(0,fnum,1)
mark2 = np.repeat(mark,dur,axis=0)
marks = np.matlib.repmat(np.expand_dims(mark2,axis=1),L,1).reshape(-1)
### subs
marks_ = np.matlib.repmat(np.expand_dims(mark2,axis=1),6,1).reshape(-1)
marksub = np.array([0,1,2,3,8,5,6,7])  #np.array([0,1,2,4])  
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
    ys[:,tt+1] = ys[:,tt] + dt*(1/tau_c)* (-ys[:,tt] +  NL(np.matmul(Q,xs[:,tt])) )  #neurons

# %% plotting heat
plt.imshow(ys,aspect='auto')
plt.xlabel('time (in 10ms)')
plt.figure()
plt.plot(time,ys.T)

# %% PSTH
window = np.arange(0,11000)  #np.arange(7000,13000)
ID = 10
plt.figure()
plt.subplot(211)
#plt.plot(time[window],marks[window])
plt.plot(time[window], marks[window])
plt.xticks([], [])
plt.ylabel('frames')
plt.subplot(212)
#plt.plot(time[window],ys[ID,window].T)
plt.plot(time[window], ys[ID,window].T)
plt.xlabel('time')
plt.ylabel('dF/F')

# %% sustained PSTH
window = np.arange(0,6500)
IDs = np.array([13,25,31])
cs = ['r','g','b']
plt.figure()
plt.subplot(411)
plt.plot(time[window]/100,marks[window])
plt.ylabel('image')
for ii,ids in enumerate(IDs):
    plt.subplot(4,1,ii+2)
    plt.plot(time[window]/100,ys[ids,window],cs[ii])
plt.xlabel('time (s)')

# %% stats-test
plt.figure()
core = 1850+7000  #core response that would be subsituted
subs = 3500+7000  #position that it is subsituted
alld = np.zeros((K,4))  #neurons by delta-backwards
for nn in range(0,K):
    for ss in range(0,alld.shape[1]):
        alld[nn,ss] = ys[nn,subs+ss*dur]/ys[nn,core+ss*dur]
        
for ss in range(0,alld.shape[1]):
    plt.plot(alld[:,ss],label=str(ss)+'-back')
plt.legend()
plt.xlabel('neuron ID')
plt.ylabel('relative response')
plt.hlines(1,0,50,linestyles='dashed')

# %%
plt.plot(alld.T)
plt.xlabel('substitutions delay')
plt.ylabel('relative response')