# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 00:49:49 2022

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import optimize

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %% parameters
N = 6
T = 200
dt = 0.1
tau = 10
eps0 = 0.1
tau_adapt= 1
rho0 = 1000
theta = 0
du = 1
muM = 0.00001
#muQ = 0.00001
tau_G = 1
tau_base = 10

# %% initialization
time = np.arange(0,T,dt)
lt = len(time)
W = np.random.randn(N,N)*0.1
F_hat = np.zeros(lt)
F_bar = np.zeros(lt)
ut = np.zeros((N,lt))
rhot = np.zeros((N,lt))
Xt = np.zeros((N,lt))
H = np.zeros((N,N,lt))
phi = np.zeros((N,lt))
eps = np.zeros((N,lt))

# %% target pattern
sigs = np.sin(time[:,None]/np.array([10,20,30])[None,:])
#target_rho = sigs.T.copy()
#target_rho = np.random.randn(N,lt)

temp = np.transpose([sigs.T] * 2)
target_rho = temp.reshape(lt, N).T*0.75  #arbistrary rate target pattern
target_rho[target_rho<0] = 0.

#temp = np.transpose([sigs.T] * 10)
#target_rho = temp.reshape(lt,30).T*0.75  #arbistrary rate target pattern
#target_rho[target_rho<0] = 0.
#v_list = np.random.choice(np.arange(0,30),20)

# %% dynamics--online simple model
for tt in range(1,lt):
    ### neural dynamics
    phi[:,tt] = phi[:,tt-1] + dt*1/tau*(Xt[:,tt-1] - phi[:,tt-1])  #filtered
    eps[:,tt] = eps[:,tt-1] + dt*1/tau_adapt*(Xt[:,tt-1]*1 -0*eps0*tau_adapt - eps[:,tt-1])  #adaptation
    ut[:,tt] = W @ phi[:,tt] - eps0*eps[:,tt] + target_rho[:,tt]#+ eps  #potential
    rhot[:,tt] = rho0*(1e-3)*np.exp((ut[:,tt]-theta)/du)  #firing rate, in kHz
    
    #spk_t = np.random.rand()
    #pos = np.where(spk_t<rhot[:,tt])[0]  #avoid run-away ##### Bernouli for now
    #Xt[pos,tt] = 1  #spiking process
    
    spike_probs = 1-np.exp(-rhot[:, tt])
    Xt[:, tt] = np.random.binomial(n = 1, p = spike_probs)
    
#    ### free energy calculation
#    F_tau = np.nansum(np.log(rhot[v_list,tt])*Xt[v_list,tt] + rhot[v_list,tt])  #instantaneous free-energy
#    F_hat[tt] = F_hat[tt-1] + dt*1/tau_G*(-F_hat[tt-1] + F_tau)  #short-term average
#    F_bar[tt] = F_bar[tt-1] + dt*1/tau_base*(-F_bar[tt-1] + F_hat[tt])  #long-term average
#    eT = F_hat[tt] - F_bar[tt]  #error signal
#    
#    ### weight update
#    dw = np.zeros((N,N))
#    for ii in range(N):
#        for jj in range(N):
#            ### Hebbian trace
#            H[ii,jj,tt] = H[ii,jj,tt-1] + dt*1/tau_G*(phi[ii]*(Xt[ii,tt] - target_rho[ii,tt])*phi[jj])
#            ### weights
#            if ii in v_list:
#                dw[ii,jj] = muM*H[ii,jj,tt]
#            else:
#                dw[ii,jj] = -eT*muM**H[ii,jj,tt]
#    W = W + dw

# %%
plt.figure()
plt.imshow(Xt,aspect='auto')

# %% direct inference
## LOG LIKELIHOOD FUNCTION
def log_likelihood(curr_weights):

    curr_weights = np.reshape(curr_weights, [N, N])

    total_potential = np.matmul(curr_weights, phi) - eps0 * eps

    curr_rho = rho0 * np.exp((total_potential - theta) / du)
    LL = np.sum(np.multiply(np.log(curr_rho), Xt) - curr_rho) * dt

    return -LL

# %%
x0 = np.random.normal(loc=0, scale=0.1, size=N*N)
res = optimize.minimize(log_likelihood, x0, options={'disp': True}, tol=1e-5, method = "BFGS")
inf_weights = np.reshape(res.x, [N, N])

# %%
plt.figure()
plt.plot(W, inf_weights, 'o')
### this is not comparable for now... but without input we can do the same thing with weight inference!!

# %% dynamics

