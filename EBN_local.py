# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:30:26 2020

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

def Target_dynamics(Ct, dt, NL_model):
    """
    Complex nonlinear dynamics as a target signal
    """
    if NL_model=='1D':
        input_dim, lt = Ct.shape
        x = np.zeros((input_dim,lt))
        for tt in range(lt-1):
            x[:,tt+1] = x[:,tt] + dt*(x[:,tt]*(0.5-x[:,tt])*(x[:,tt]+0.5) + Ct[:,tt])  #1D attractor dynamics
    if NL_model=='2D':
        input_dim, lt = Ct.shape
        x = np.zeros((input_dim,lt))
        for tt in range(lt-1):
            x[0,tt+1] = x[0,tt] + dt*(Ct[0,tt]/0.02 + x[1,tt]/0.125)
            x[1,tt+1] = x[1,tt] + dt*(Ct[1,tt]/0.02 + (2*(1-x[0,tt]**2)*x[1,tt]-x[0,tt])/0.125 )
    if NL_model=='3D':
        input_dim, lt = Ct.shape
        x = np.zeros((input_dim,lt))
        for tt in range(lt-1):
            x[0,tt+1] = x[0,tt] + dt*( Ct[0,tt]/0.02 + 10*(x[1,tt]-x[0,tt]) )
            x[1,tt+1] = x[1,tt] + dt*( Ct[1,tt]/0.02 - x[0,tt]*x[2,tt] - x[1,tt] )
            x[2,tt+1] = x[2,tt] + dt*( Ct[2,tt]/0.02 + x[0,tt]*x[1,tt] - 8*(x[2,tt]+28)/3 )
    return x

# %% EBN model
#dimensions
N = 100
T = 300
dt = 0.01
time = np.arange(0,T,dt)
lt = len(time)
input_dim = 3
output_dim = 3

#stimuli
ct = np.zeros((input_dim,lt))  #command input time series
ct[:,1000:3000] = np.random.randn(1,2000)*1.
Xs = Target_dynamics(ct, dt, '3D')*0.1  #target dynamics

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

# %% plot 3D Lorentz
plt.figure()
ax = plt.axes(projection='3d')
ax.plot(rec[0,1000:5000], rec[1,1000:5000], rec[2,1000:5000], '-b', label='Reconstruct')
ax.plot(Xs[0,1000:5000], Xs[1,1000:5000], Xs[2,1000:5000], '--r', alpha = 0.5, label='Target')
ax.legend()
#%%
plt.figure()
plt.subplot(411)
plt.imshow(spk,aspect='auto')
plt.ylabel('Neurons')
for dd in range(3):
    plt.subplot(4,1,dd+2)
    plt.plot(time,rec[dd,:].T,label='Reconstruct')
    plt.plot(time,Xs[dd,:].T,'--',label='Target')
    plt.xlim([0,time[-1]])
plt.legend()
plt.xlabel('time')

# %% Batch learning
trials = 30
#initialization
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
#run batches
for bb in range(trials):
    #Dynamics
    for tt in range(0,lt-1):
        spk[:,tt] = Threshold(us[:,tt], D, mu, vv)  #spiking process
        rec[:,tt] = np.matmul(D,rs[:,tt])
        err[:,tt] = Xs[:,tt] - rec[:,tt]  #error signal
        Ws = Ws + eta*dt*np.outer( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
        us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
          + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
        rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate

# %% Poinsson EBN
###############################################################################
# %% 
def Poisson_spk(uu, D, mu, vv, alpha, FM, Fm):
    """
    Relax the hard spiking threshold to Poinsson nonlinearity
    """
    output_dim, N = D.shape
    Di = np.linalg.norm(D,axis=0)**2
    Ti = 0.5*(Di+mu+vv)  #threshold
    fv = FM/(1+FM*np.exp(-alpha*(uu-Ti))) + Fm  #sigmoind nonlinearity
    pspk = 1-np.exp(-fv*dt)  #calculate probability of spiking
    iisp = np.zeros(N)
    iisp[np.where(pspk>np.random.rand(N))[0]] = 1  #find neurons that spiked
    return iisp

# %%
#Poisson neuron settings
alpha = 1000  #slope of sigmoid
FM = 500
Fm = 0

#stimuli
ct = np.zeros((input_dim,lt))  #command input time series
ct[:,1000:3000] = np.random.randn(1,2000)*1.
Xs = Target_dynamics(ct, dt, '3D')*0.1  #target dynamics

#biophysics
lamb_u = 20  #spiking time scale (0.1)
lamb_r = 2
k = 100.  #error coupling
eta = 0.5  #learning rate
mu = 10**-6  #L2 spiking penalty
vv = 10**-5  #L1 spiking penalty
sig = 0.0  #noise strength

#connectivity
Ws = np.random.randn(N,N)*0.1  #slow connection to be learned
D = np.random.randn(output_dim,N)*0.1  #output connections
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
    spk[:,tt] = Poisson_spk(us[:,tt], D, mu, vv, alpha, FM, Fm)  #Poisson spiking process
    rec[:,tt] = np.matmul(D,rs[:,tt])
    err[:,tt] = Xs[:,tt] - rec[:,tt]  #error signal
    Ws = Ws + eta*dt*np.outer( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
#    Ws = Ws + eta*dt*np.matmul( (rs[:,tt]) , k*(np.matmul(D.T,err[:,tt])).T )  #w/o nonlinearity
    us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
      + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
    rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate

# %%
#plotting
plt.figure()
plt.subplot(211)
plt.imshow(spk,aspect='auto')
plt.ylabel('Neurons')
plt.subplot(212)
plt.plot(time,rec.T,label='Reconstruct')
plt.plot(time,Xs.T,'--',label='Target')
plt.xlim([0,time[-1]])
plt.legend()
plt.xlabel('time')

# %% Batch learning
trials = 30
###initialization
#connectivity
Ws = np.random.randn(N,N)  #slow connection to be learned
D = np.random.randn(output_dim,N)  #output connections
Wf = D.T @ D + mu*np.eye(N)  #fast connections
F = D.copy()  #input connections  #F = np.random.randn(input_dim,N)  
#time series
us = np.zeros((N,lt))  #neural activities
us[:,0] = np.random.randn(N)*0.1
err = np.zeros((output_dim,lt))  #error signal
err_filt = np.zeros((output_dim,lt))  #filtered error signal
spk = np.zeros((N,lt))  #spiking activity
rs = np.zeros((N,lt))  #firing rate
rec = np.zeros((output_dim,lt))  #store the reconstruction
#run batches
for bb in range(trials):
    #Dynamics
    for tt in range(0,lt-1):
        spk[:,tt] = Poisson_spk(us[:,tt], D, mu, vv, alpha, FM, Fm)  #Poisson spiking process
        rec[:,tt] = np.matmul(D,rs[:,tt])
        err[:,tt] = Xs[:,tt] - rec[:,tt]  #error signal
        err_filt[:,tt+1] = err_filt[:,tt] + dt*(-1.*(err_filt[:,tt]) + err[:,tt])  #filtered error signal
        Ws = Ws + eta*dt*np.outer( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
#        Ws = Ws + eta*dt*np.matmul( (rs[:,tt]) , k*(np.matmul(D.T,err_filt[:,tt])).T )  #w/o nonlinearity and filtered error
        us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
          + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
        rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate


# %% Kernel EBN
###############################################################################
# %% Basis function for temporal kernel
def basis_function1(nkbins, nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    #nBases = 3
    #nkbins = 10 #binfun(duration); # number of bins for the basis functions
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.4),(nBases,1))  #take log for nonlinear time
    dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
    width = 5.*dbcenter # width of each bump
    bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)  #raise-cosine function formula
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] 
    #plt.plot(np.array(BBstm).T)
    return np.array(BBstm).T

# %%
#dimensions
N = 30
T = 200
dt = 0.01
time = np.arange(0,T,dt)
lt = len(time)
input_dim = 1
output_dim = 1

#setup for kernels
pad = 100  #length of temporal kernel
h = 5  #number of basis function used
Ks = (np.fliplr(basis_function1(pad, h).T).T).T  #weights x length (h as the weights on basis)

#Poisson neuron settings
alpha = 1000  #slope of sigmoid
FM = 25
Fm = 0

#stimuli
ct = np.zeros((input_dim,lt))  #command input time series
ct[:,1000:3000] = np.random.randn(1,2000)*0.5
Xs = Target_dynamics(ct, dt, '1D')  #target dynamics

#biophysics
lamb_u = 0.1  #spiking time scale
lamb_r = 0.5  #firing rate scale
lamb_w = 0.9  #relaxation of learning
k = 1.  #error coupling
eta = 0.5  #learning rate
mu = 10**-6  #L2 spiking penalty
vv = 10**-5  #L1 spiking penalty
sig = 0.0  #noise strength

#connectivity
Ws = np.random.randn(N,N,h)*0.1  #slow connection to be learned
D = np.random.randn(output_dim,N)*0.1  #output connections
Wf = D.T @ D + mu*np.eye(N)  #fast connections
F = D.copy()  #input connections  #F = np.random.randn(input_dim,N)

#time series
us = np.zeros((N,lt))  #neural activities
us[:,:pad] = np.random.randn(N,pad)*0.1
err = np.zeros((output_dim,lt))  #error signal
spk = np.zeros((N,lt))  #spiking activity
rs = np.zeros((N,lt))  #firing rate
rec = np.zeros((output_dim,lt))  #store the reconstruction

#Dynamics
for tt in range(pad,lt-1):
    spk[:,tt] = Poisson_spk(us[:,tt], D, mu, vv, alpha, FM, Fm)  #Poisson spiking process
    rec[:,tt] = np.matmul(D,rs[:,tt])
    err[:,tt] = Xs[:,tt] - rec[:,tt]  #error signal
#    Ws = Ws + eta*dt*np.matmul( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
#    Ws = Ws + eta*dt*np.matmul( (rs[:,tt]) , k*(np.matmul(D.T,err[:,tt])).T )  #w/o nonlinearity
    ### learning with weights on kernel bases
    Ws = (1-lamb_w)*Ws + eta*dt*np.matmul( np.matmul(Ks,Phi(rs[:,tt-pad:tt]).T)[:,:,None] , (np.matmul(D.T,err[:,tt]))[:,None].T ).T
    us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
      + np.einsum('ijk,jk->i',  np.matmul(Ws,Ks), Phi(rs[:,tt-pad:tt])) \
      + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
    rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate

# %% Batch learning
trials = 50
###initialization
#connectivity
Ws = np.random.randn(N,N,h)*0.1  #slow connection to be learned
D = np.random.randn(output_dim,N)*0.1  #output connections
Wf = D.T @ D + mu*np.eye(N)  #fast connections
F = D.copy()  #input connections  #F = np.random.randn(input_dim,N)
#time series
us = np.zeros((N,lt))  #neural activities
us[:,:pad] = np.random.randn(N,pad)*0.1
err = np.zeros((output_dim,lt))  #error signal
spk = np.zeros((N,lt))  #spiking activity
rs = np.zeros((N,lt))  #firing rate
rec = np.zeros((output_dim,lt))  #store the reconstruction
#run batches
for bb in range(trials):
    #Dynamics
    for tt in range(pad,lt-1):
        spk[:,tt] = Poisson_spk(us[:,tt], D, mu, vv, alpha, FM, Fm)  #Poisson spiking process
        rec[:,tt] = np.matmul(D,rs[:,tt])
        err[:,tt] = Xs[:,tt] - rec[:,tt]  #error signal
        ### learning with weights on kernel bases
        Ws = (1-lamb_w)*Ws + eta*dt*np.matmul( np.matmul(Ks,Phi(rs[:,tt-pad:tt]).T)[:,:,None] , k*(np.matmul(D.T,err[:,tt]))[:,None].T ).T
        us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
          + np.einsum('ijk,jk->i',  np.matmul(Ws,Ks), Phi(rs[:,tt-pad:tt])) \
          + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
        rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate
        
# %%
#plotting
plt.figure()
plt.subplot(211)
plt.imshow(spk,aspect='auto')
plt.ylabel('Neurons')
plt.subplot(212)
plt.plot(time,rec.T,label='Reconstruct')
plt.plot(time,Xs.T,'--',label='Target')
plt.xlim([0,time[-1]])
plt.legend()
plt.xlabel('time')


# %%
###############################################################################
###############################################################################
# %% EXTERNAL examples
# %% iterate through recordings
###for WW model for now
trials = WW_r.shape[0]
Cs = WW_s.copy()
Os = WW_r.copy()
###initialization
#biophysics
lamb = 0.1  #spiking time scale
k = 1.  #error coupling
eta = 0.5  #learning rate
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
#run batches
for bb in range(trials):
    #unfold data
    ct = Cs[bb,:][None,:]  #command input stimuli
    xs = Os[bb,:][None,:]  #target observation
    #Dynamics
    for tt in range(0,lt-1):
        spk[:,tt] = Threshold(us[:,tt], D, mu, vv)  #spiking process
        rec[:,tt] = np.matmul(D,rs[:,tt])
        err[:,tt] = xs[:,tt] - rec[:,tt]  #error signal
        Ws = Ws + eta*dt*np.matmul( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
        us[:,tt+1] = us[:,tt] + dt*(-lamb*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
          + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
        rs[:,tt+1] = rs[:,tt] + dt*(-lamb*rs[:,tt] + spk[:,tt])  #spik rate

# %% learning result as generative process
bb = 15  #pick on replay
#time series
us = np.zeros((N,lt))  #neural activities
us[:,0] = np.random.randn(N)*0.1
err = np.zeros((output_dim,lt))  #error signal
spk = np.zeros((N,lt))  #spiking activity
rs = np.zeros((N,lt))  #firing rate
rec = np.zeros((output_dim,lt))  #store the reconstruction
#unfold data
ct = Cs[bb,:][None,:]  #command input stimuli
xs = Os[bb,:][None,:]  #target observation
#Dynamics
for tt in range(0,lt-1):
    spk[:,tt] = Threshold(us[:,tt], D, mu, vv)  #spiking process
    rec[:,tt] = np.matmul(D,rs[:,tt])
    err[:,tt] = xs[:,tt] - rec[:,tt]  #error signal
#    Ws = Ws + eta*dt*np.matmul( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
    us[:,tt+1] = us[:,tt] + dt*(-lamb*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
      + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
    rs[:,tt+1] = rs[:,tt] + dt*(-lamb*rs[:,tt] + spk[:,tt])  #spik rate

# %%
plt.figure()
plt.subplot(311)
plt.imshow(spk,aspect='auto')
plt.ylabel('Neurons')
plt.subplot(312)
plt.plot(time,rec.T,label='Reconstruct')
plt.plot(time,xs.T,'--',label='Target')
plt.legend()
plt.subplot(313)
plt.plot(time,ct.T)
plt.ylabel('input')
plt.xlim([0,time[-1]])
plt.xlabel('time')

# %% stats
c_level = np.zeros(trials)  #storing input level
r_state = np.zeros(trials)  #store steady-state response
r_truth = np.zeros(trials)
for bb in range(trials):
    #time series
    us = np.zeros((N,lt))  #neural activities
    us[:,0] = np.random.randn(N)*0.1
    err = np.zeros((output_dim,lt))  #error signal
    spk = np.zeros((N,lt))  #spiking activity
    rs = np.zeros((N,lt))  #firing rate
    rec = np.zeros((output_dim,lt))  #store the reconstruction
    #unfold data
    ct = Cs[bb,:][None,:]  #command input stimuli
    xs = Os[bb,:][None,:]  #target observation
    #Dynamics
    for tt in range(0,lt-1):
        spk[:,tt] = Threshold(us[:,tt], D, mu, vv)  #spiking process
        rec[:,tt] = np.matmul(D,rs[:,tt])
        err[:,tt] = xs[:,tt] - rec[:,tt]  #error signal
    #    Ws = Ws + eta*dt*np.matmul( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
        us[:,tt+1] = us[:,tt] + dt*(-lamb*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
          + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
        rs[:,tt+1] = rs[:,tt] + dt*(-lamb*rs[:,tt] + spk[:,tt])  #spik rate
    ### store generaitve I/O results
    c_level[bb] = Cs[bb,1000]  #where the stimulus is representative
    r_state[bb] = rec[0,-10]  #steady-state
    r_truth[bb] = xs[0,-10]  #true value

# %%
plt.figure()
plt.plot(c_level+np.random.randn(trials)*0.0002,r_state,'o',alpha=0.5,markersize=15,label='generative')
plt.plot(c_level+np.random.randn(trials)*0.0002,r_truth,'ko',alpha=0.5,markersize=5,label='true')
plt.xlabel('input strength (maps to coherence)')
plt.ylabel('steady-state response')
plt.legend()