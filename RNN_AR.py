# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 22:05:51 2023

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import ssm

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

### studying bistability in RNN model, analyze time series, and fit with ssm
# simulate models with bistability: including noise driven transition or low-rank drifts (true bifurcation)
# analyze network activity by fitting AR model and looking at spectrum
# fit switching LDS and check if the dynamical matrix can pick up some signature

# %% bistable RNN
def NL(x):
    return np.tanh(x)
N = 50
s = 0.
g = 1.
M = np.random.randn(N,N)
uu,ss,vv = np.linalg.svd(M)
v1,v2 = uu[:,1], uu[:,2] #np.random.randn(N), np.random.randn(N)
J = np.random.randn(N,N)/np.sqrt(N) + (np.outer(v1,v1) + np.outer(v2,v2) + np.outer(v1,v2)*.0)
D = 2.
Ds = 0.1
tau_s = 20
T = 100
dt = 0.1
time = np.arange(0,T,dt)
lt = len(time)
xt = np.zeros((N,lt))
xt[:,0] = np.random.randn(N)*.5
eta = np.zeros(lt)
rt = xt*1

for tt in range(0,lt-1):
    xt[:,tt+1] = xt[:,tt] + dt*(-xt[:,tt] + s*rt[:,tt] + g*J@rt[:,tt]) + np.sqrt(dt*D)*np.random.randn(N)
    rt[:,tt+1] = NL(xt[:,tt+1])
    ### noisy overlap
    J = J + np.outer(v1,v2)*eta[tt]
    eta[tt+1] = eta[tt] + dt*(-eta[tt]/tau_s) + np.sqrt(dt*Ds)*np.random.randn()

plt.figure()
plt.imshow(rt, aspect='auto')

# %% checking critical-slow down
xt = rt*1
def AR_solve(x1, x0):
    n,T = x1.shape
    inv_xxt = np.linalg.pinv(x0@x0.T)
    A = inv_xxt @ x0 @ x1.T
    eps = np.mean((x1 - A@x0)**2)**0.5/T
    return A, eps

T_init = 200
T = 60
T_final = 800
dt = 1
deT = T_final-T_init
As = np.zeros((N,N,deT))
Sigs = np.zeros((N,N,deT))
eps = np.zeros(deT)
for tt in range(0,deT):
    x1,x0 = xt[:,T_init+tt:T_init+tt+T], xt[:,T_init+tt+dt:T_init+tt+T+dt]
    A_, eps[tt] = AR_solve(x1,x0)
    As[:,:,tt] = A_
    Sigs[:,:,tt] = (x1 - A_@x0) @ (x1 - A_@x0).T
    
# %%
eigs = np.zeros((N,deT))
eigs_n = np.zeros((N,deT))
critt = np.zeros(deT)
for tt in range(0,deT):
    A_ = As[:,:,tt]
    uu,vv = np.linalg.eig(A_)
    eigs[:,tt] = uu
    critt[tt] = np.sum(np.real(uu)>.9)/N
#    critt[tt] = np.sum(np.abs(np.real(uu)-1)<0.1)/N
    S_ = Sigs[:,:,tt]
    uu,vv = np.linalg.eig(S_)
    eigs_n[:,tt] = uu/np.max(np.abs(uu))

# %%
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.imshow((np.real(eigs)), aspect='auto',vmin=0, vmax=1.1)
plt.colorbar()
plt.xlabel('time', fontsize=30)
plt.ylabel('modes', fontsize=30)
plt.subplot(122)
plt.imshow(np.log(np.real(eigs_n)), aspect='auto')#,vmin=0, vmax=1.1)
plt.colorbar()
plt.xlabel('time', fontsize=30)
plt.ylabel('noise', fontsize=30)

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(critt)
plt.xlabel('time', fontsize=30)
plt.ylabel('fraction of critical modes', fontsize=30)
plt.ylim([0.01, 0.5])
plt.subplot(122)
plt.plot(eps)
plt.xlabel('time', fontsize=30)
plt.ylabel('AR error', fontsize=30)

plt.figure()
plt.imshow(xt[:,T_init:T_final], aspect='auto')

# %% SMM inference
# Set the parameters of the SLDS
data = xt.T
n_disc_states = 3       # number of discrete states
latent_dim = 2       # number of latent dimensions
emissions_dim = N*1      # number of observed dimensions
slds = ssm.SLDS(emissions_dim, n_disc_states, latent_dim, emissions="gaussian_orthog")

# Fit the model using Laplace-EM with a structured variational posterior
q_lem_elbos, q_lem = slds.fit(data, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               num_iters=100, alpha=0.0)

# Get the posterior mean of the continuous states
q_lem_x = q_lem.mean_continuous_states[0]

# Find the permutation that matches the true and inferred states
#slds.permute(find_permutation(states_z, slds.most_likely_states(q_lem_x, data)))
q_lem_z = slds.most_likely_states(q_lem_x, data)

# Smooth the data under the variational posterior
q_lem_y = slds.smooth(q_lem_x, data)

# %%
plt.figure(figsize=(5, 15))
plt.subplot(311)
plt.imshow(q_lem_z[:,None].T, aspect='auto')
plt.subplot(313)
plt.plot(q_lem_y)
plt.subplot(312)
plt.imshow(data.T, aspect='auto')

# %% compute specturm... with latents
eig_ssm = np.zeros((latent_dim, lt))
s_eig = np.zeros((n_disc_states, latent_dim))
for ss in range(latent_dim):
    A_ = slds.dynamics.As[ss,:,:].squeeze()
    uu,ee = np.linalg.eig(A_)
    s_eig[ss,:] = np.real(uu)
for tt in range(lt):
    zt = q_lem_z[tt]
    eig_ssm[:,tt] = s_eig[zt,:]
plt.figure()
plt.plot(eig_ssm.T)
