# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:44:23 2020

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

def Environement(x,y,C0,sig,target):
    """
    Given location in 2D x,y and the scale factor C0, width sig, and target location, return environment concentration E
    """
#    E = C0*np.exp(-((x-target[0])**2+(y-target[1])**2)/sig)
    E = C0*np.exp(-((y-target[1])**2)/sig)
    return E

def Sensory(E,K):
    """
    Given envioronment concentration E and the kernel K return sensory activation S
    """
    S = np.dot(E,K)
    return S

def Action(S,N):
    """
    Given the sensory activation S and nonlinearity N return an action A
    """
    P = 1./(1+np.exp(N*S))
    if np.random.rand()<P:
        A = 1
    else:
        A = 0
    return A

def basis_function1(nkbins, nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.4),(nBases,1))  #take log for nonlinear time
    dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
    width = 5.*dbcenter # width of each bump
    bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)  #raise-cosine function formula
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] 
    return np.array(BBstm).T

def chemotaxis_track(Ns, pars, time):
    """
    Simulate chemotaxis trajectories that have behavioral output: 
    angles change in motion and concentration sensing dC (further add Venteral/Dorsal bends)
    """
    C0, sig, target, K, N, thr, v, vr = pars
    dt = time[1]-time[0]
    tl = len(time)
    kl = len(K)
    Et, St, At, temp = np.zeros((Ns,tl)), np.zeros((Ns,tl)), np.zeros((Ns,tl)), np.zeros((Ns,tl))
    xys = np.zeros((Ns,2,tl))
    ###iteration through repeats
    for nn in range(Ns):
        x,y = np.random.randn(2)
        th = np.random.randn()*2*np.pi-np.pi
        ###iteration through time
        for tt in range(kl,tl):
            ### update chemotaxis measurements
            Et[nn,tt] = Environement(x,y,C0,sig,target)
            Et_ = Et[nn,tt-kl:tt]
            St[nn,tt] = Sensory(Et_,K)
            St_ = St[nn,tt]
            temp[nn,tt] = Action(St_, N)
            At_ = temp[nn,tt]
            ### update kinematics
            dth = np.random.randn()*thr + At_*(np.random.rand()*2*np.pi-np.pi)
            th = th + dth
            dd = (v+vr*np.random.randn())*dt
            x = x + dd*np.sin(dth)
            y = y + dd*np.cos(dth)
            # angles as the output
#            dth = np.arctan(np.sin(dth)/np.cos(dth)) + np.pi
            At[nn,tt] = th
            xys[nn,:,tt] = np.array([x,y])
    return Et, St, At, xys  #, Vs, Ds

# %% parameters and simulation
dt = 0.1
T = 100
time = np.arange(0,T,dt)
tl = len(time)
thr = .5  #noise strength on angle change
v = 1.1  #mean velocity
vr = 0.1  #noise strength on velocity
C0 = 10**3  #concentration scaling
sig = 50  #width of concentration profile
target = np.array([0,50])  #traget position
kl = 20  #kenrel length
nb = 5  #number of basis for the kenel
K = np.dot(np.random.randn(nb), (np.fliplr(basis_function1(kl,nb).T).T).T)  #constructing the kernel with basis function
K = -np.abs(K)
N = 100 #scaling of logistic nonlinearity
Ns = 50  #number of repetitionsd
pars = C0, sig, target, K, N, thr, v, vr
Et, St, At, xys = chemotaxis_track(Ns, pars, time)

# %%
plt.figure()
plt.subplot(311)
plt.plot(time,Et.T)
plt.ylabel('Odor concentration')
plt.subplot(312)
plt.plot(time,St.T)
plt.ylabel('Sensory activity')
plt.subplot(313)
plt.plot(time,At.T)
plt.ylabel('head angle')
plt.xlabel('time')

# %% tracks
plt.figure()
x = np.arange(-10,60,1)
xx_grad = C0*np.exp(-((x-50)**2)/1000)
plt.imshow(np.expand_dims(xx_grad,axis=1).T,extent=[-10,60,-10,10],aspect="auto")
for nn in range(Ns):
    plt.plot(xys[nn,1,:], xys[nn,0,:],linewidth=3.5)#,'grey')
    
# %% train EBN
# %% iterate through recordings
### load data and parameter for chemotaxis model
trials = Ns
Cs = St.copy()
Os = At.copy()
lt = tl
output_dim = 1
###initialization
#biophysics
lamb_u = 10  #spiking time scale
lamb_r = .1
k = 10.  #error coupling
eta = 0.1  #learning rate
mu = 10**-6  #L2 spiking penalty
vv = 10**-5  #L1 spiking penalty
sig = 0.0  #noise strength
#connectivity
N = 22
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
        Ws = Ws + eta*dt*np.outer( Phi(rs[:,tt]) , (np.matmul(D.T,err[:,tt])).T )  #synaptic learning rule
        #Ws = Ws * mask  #connectivity constraints
        us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
          + np.matmul(Ws, Phi(rs[:,tt])) + k*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
        rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate

# %% learning result as generative process
bb = 1  #pick on replay
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
    us[:,tt+1] = us[:,tt] + dt*(-lamb_u*us[:,tt] + np.matmul(F.T,ct[:,tt]) - np.matmul(Wf,spk[:,tt]) \
      + np.matmul(Ws, Phi(rs[:,tt])) + 0.*np.matmul(D.T,err[:,tt]) + sig*np.random.randn(N))  #voltage
    rs[:,tt+1] = rs[:,tt] + dt*(-lamb_r*rs[:,tt] + spk[:,tt])  #spik rate

# %%
plt.figure()
plt.subplot(311)
plt.imshow(spk,aspect='auto')
plt.ylabel('Neurons')
plt.subplot(312)
plt.plot(time,rec.T,label='Reconstruct')
plt.plot(time,xs.T,'--',label='Target')
plt.xlim([0,time[-1]])
plt.ylabel('head angle')
plt.legend()
plt.subplot(313)
plt.plot(time,ct.T)
plt.ylabel('input')
plt.xlim([0,time[-1]])
plt.xlabel('time')
