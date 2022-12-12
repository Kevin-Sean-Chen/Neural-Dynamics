# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 00:04:47 2022

@author: kevin
"""
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

#%matplotlib qt5
# %% parameters
N = 200  #number of neurons
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0  #learning initial constant
dt = 0.1
nsecs = 500
learn_every = 2  #effective learning rate

scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M = np.random.randn(N,N)*g*scale
cutoff = 100
uu,ss,vv = np.linalg.svd(M)
ss[cutoff:] = 0
M = uu @ np.diag(ss) @ vv
sparse = np.random.rand(N,N)
#M[sparse>p] = 0
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1
M = M*mask

nRec2Out = N
wo = np.zeros((nRec2Out,1))
dw = np.zeros((nRec2Out,1))
wf = 2.0*(np.random.rand(N,1)-0.5)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

###target pattern
amp = 0.7;
freq = 1/60;
rescale = 2
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
    (amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
    (amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
    1*(amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
#ft[ft<0] = 0
ft = ft/1.5

wo_len = np.zeros((1,simtime_len))    
zt = np.zeros((1,simtime_len))
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,simtime_len))
rt = np.zeros((N,simtime_len))

x = x0
r = np.tanh(x)
z = z0

### hyperparameters
P = (1.0/alpha)*np.eye(nRec2Out)  # inverse covariance
v_alpha = np.zeros(N)  # weight sparsity prior
sig = 1  #noise level
update_every = 50

# %% FORCE learning
plt.figure()
ti = 0
for t in range(len(simtime)-1):
    ti = ti+1
    x = (1.0-dt)*x + M @ (r*dt) #+ wf * (z*dt)
    r = np.tanh(x)
    rt[:,t] = r[:,0]  #xt[:,t] = x[:,0]
    z = wo.T @ r
    
    if np.mod(ti, learn_every) == 0:
        k = P @ r;
        rPr = r.T @ k
        c = 1.0/(1.0 + rPr)
        P = P - k @ (k.T * c)  #projection matrix
        P = P*sig
        P = P - P@np.diag(v_alpha)@P@np.diag(v_alpha)@P  # update with sparse prior
        
        # update the error for the linear readout
        e = z-ft[ti]
        
        # update the output weights
        dw = -e*k*c*sig  # update with noise prior
        wo = wo + dw
        
        # update the internal weight matrix using the output's error
        M = M + np.repeat(dw,N,1).T

    # Store the output of the system.
    zt[0,ti] = np.squeeze(z)
    wo_len[0,ti] = np.sqrt(wo.T @ wo)	
    
    if np.mod(ti, update_every)==0:
    ### update hyperparameters
    ### ... should try online iterative version~~
        v_alpha = (1-v_alpha*np.diag(P))/wo[:,0]**2
        sig = np.linalg.norm(ft[:t]-np.squeeze(wo.T@rt[:,:t])) / (t+np.sum(1-v_alpha*np.diag(P)))

error_avg = sum(abs(zt[0,:]-ft))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.plot(ft)
plt.plot(zt[0,:],'--')

plt.figure()
plt.imshow(rt,aspect='auto')
# %% testing
zpt = np.zeros((1,simtime_len))
ti = 0
x = x0
r = np.tanh(x)
z = z0
for t in range(len(simtime)-1):
    ti = ti+1 
    
    x = (1.0-dt)*x + M @ (r*dt) #+ wf * (z*dt)
    r = np.tanh(x)
    z = wo.T @ r

    zpt[0,ti] = z

zpt = np.squeeze(zpt)
plt.figure()
plt.plot(ft)
plt.plot(zpt)