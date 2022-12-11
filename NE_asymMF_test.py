# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:58:03 2021

@author: kevin
"""

import numpy as np
import scipy as sp
from scipy.optimize import minimize
import itertools
import copy

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

# %% Kinetic Ising model
###############################################################################
# %% Stimuli
T = 10000
D = 3
smooth = 100
noise = np.random.randn(T,D)
X = noise.copy()
for ii in range(D):
    X[:,ii] = np.convolve(noise[:,ii],np.ones(smooth),'same')/smooth *5
    #X[:,ii] = np.convolve(noise[:,1],np.ones(smooth),'same') + np.random.randn(T)*3  #correlated

# %% Network settings
N = 20
Phi = np.random.randn(D,N)
J = np.random.randn(N,N)/N**0.5 /10
#vv = np.random.randn(N)
#J = np.outer(vv,vv)*0.1 + np.random.randn(N,N)*0.05
#J = J_nMF.copy()

def Current(h,J,s):
    theta = h + J @ s
    return theta

def Transition(si,thetai,beta):
    P = np.exp(-si*thetai*beta)/(2*np.cosh(thetai*beta))
    rand = np.random.rand()
    if P>rand:
        s_ = -si.copy()
    elif P<=rand:
        s_ = si.copy()
    return s_

def AsyncTrans(si,thetai):
    s_ = 0
    return s_

# %% Dynamics
def Kinetic_Ising(X,Phi,J,kbT):
    beta = 1/kbT
    N, T = Phi.shape[1], X.shape[0]
    S = np.ones((N,T))
    for tt in range(0,T-1):
        Ht = X[tt,:] @ Phi
        Theta = Current(Ht, J, S[:,tt])
        for ss in range(N):
            S[ss,tt+1] = Transition(S[ss,tt],Theta[ss],beta)
    return S

S = Kinetic_Ising(X,Phi,J,1)

plt.figure()
plt.imshow(S, aspect='auto')


# %% Learning Jij from input
###############################################################################
# %% Create target spike trains
def spiking(X,Phi,kbT):
    beta = 1/kbT
    prob_spk = np.linalg.pinv(Phi) @ X.T  #consistant with Ising model temperature effects
    #NL = np.exp(prob_spk*beta)  #exponential nonlinearity
    NL = 1/(1+np.exp(-beta*prob_spk))
    rand = np.random.poisson(NL,size=prob_spk.shape)
    pos = np.where(rand>0)
    spk = np.zeros_like(NL)-1
    spk[pos] = 1
    return spk
#spks = np.random.randint(2,size=(N,T))
spks = spiking(X,Phi,1)
si = spks.copy()

plt.figure()
plt.imshow(si, aspect='auto')

# %% MF methods
def nMF(si,kbT):
    beta = 1/kbT
    mi = np.mean(si,1)
    dsi = si-mi[:,None]
    #Cij = dsi @ dsi.T / dsi.shape[1] #np.cov(dsi)
    Dij = dsi[:,0:-2] @ dsi[:,1:-1].T / dsi.shape[1]
    Cij = np.cov(si) #dsi @ dsi.T / dsi.shape[1] #np.cov(dsi)
    #Dij = si[:,0:-2] @ si[:,1:-1].T - np.outer(mi[0:-2],mi[1:-1])
    Aij_nMF = beta*np.diag(1-mi**2)
    J_nMF = np.linalg.pinv(Aij_nMF) @ Dij @ np.linalg.pinv(Cij)
#    J_nMF = -np.linalg.inv(Cij)
    return J_nMF

kbT = .5
si = Kinetic_Ising(X, 1*Phi, J, kbT)
J_nMF = nMF(si, kbT)
plt.figure()
plt.imshow(J_nMF,aspect='auto')
plt.figure()
plt.plot(J.reshape(-1), J_nMF.reshape(-1),'.')

# %% decoding
###############################################################################
# run kinetic Ising model as above with J_nMF
g = 0.5
kbT = .1
si = spiking(X,Phi,kbT)
J_nMF = nMF(si,kbT)
#J_rand = np.random.randn(N,N)
# generate spikes and use Phi @ S to reconstruct stimulus
#S_gen = Kinetic_Ising(X, Phi, J_nMF, kbT)
#X_rec = (Phi @ S_gen).T
h_nMF = kbT*np.arctan(si) - J_nMF @ si
X_rec = (Phi @ h_nMF).T

plt.figure()
plt.plot(X,X_rec,'o')

# %% Scanning
# compare reconstruction and scan through data length
eps = 10**-16
def nMF_decoding(kbT, X, Phi, si, cond):
    if cond=="MF":
        JJ = nMF(si,kbT)
        #JJ = -np.linalg.inv(np.cov(si))
    elif cond=="rand":
        JJ = np.random.randn(N,N)
    boundm = si.copy()
    boundm[np.where(boundm>0)] = boundm[np.where(boundm>0)]-eps
    boundm[np.where(boundm<0)] = boundm[np.where(boundm<0)]+eps
    h_nMF = kbT*np.arctanh(boundm) - JJ @ boundm
    h_nMF = 1*np.arctanh(boundm) - JJ @ boundm  #not real mean field!!
    X_rec = (Phi @ h_nMF).T
    cof = np.corrcoef(X.reshape(-1),X_rec.reshape(-1))[0][1]
    return cof

#si = spiking(X,Phi,0.1)
si = Kinetic_Ising(X, Phi, J_nMF, kbT)
Ts = np.arange(0.1,5,0.5)
decs_MF = np.array([nMF_decoding(tt, X, Phi, si, "MF") for tt in Ts])
decs_rand = np.array([nMF_decoding(tt, X, Phi, si, "rand") for tt in Ts])

plt.figure()
plt.plot(1/Ts,np.abs(decs_MF),'-o')
plt.plot(1/Ts,decs_rand,'-o')
plt.figure()
plt.plot(1/Ts,(decs_MF-decs_rand)/decs_MF,'-o')

# %% different smoothness
def stim_gen(par):
    X = np.random.randn(T,D)
    for ii in range(D):
        X[:,ii] = np.convolve(noise[:,ii],np.ones(par),'same')
        #X[:,ii] = np.convolve(noise[:,1],np.ones(100),'same') + np.random.randn(T)*par  #correlated
    return X

ss = np.array([5,10,50,100,150,200,500])*1
decs_ss = np.zeros(len(ss))
for ii in range(len(ss)):
    X = stim_gen(ss[ii])
    si = spiking(X,Phi,g)
    decs_ss[ii] = nMF_decoding(10, X, Phi, si, "MF")

plt.figure()
plt.plot(ss,decs_ss,'-o')

# %% different covariance and stimulus statistics
def stim_SDE(a,tau):
    X = np.zeros((T,D))
    A = np.ones((D,D))*a
    np.fill_diagonal(A,-np.ones(D))
    dt = 1
    for tt in range(T-1):
        X[tt+1,:] = X[tt,:] + dt*(A @ X[tt,:] + np.random.randn(D))
    return X

X_sde = stim_SDE(0.425,50)  #0.4-0.5 is sensitive
plt.figure()
plt.plot(X_sde)

# %% correlation time
aa = np.array([0.2,0.3,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49])
decs_aa = np.zeros(len(aa))
aa_rec = np.zeros(len(aa))
for ii in range(len(aa)):
    X = stim_SDE(aa[ii],100)
    si = spiking(X,Phi,0.5)
    decs_aa[ii] = nMF_decoding(10, X, Phi, si, "MF")
    aa_rec[ii] = np.corrcoef(X[:,0],X[:,1])[0][1]

plt.figure()
plt.plot(aa_rec,decs_aa,'-o')

# %%
# move on to non-stationary problems~~
###############################################################################
# %% repeating trials
def spiking_rep(X,Phi,kbt,rr):
    spks_rep = np.zeros((rr, Phi.shape[1], X.shape[0]))
    for r in range(rr):
        spks_rep[r,:,:] = spiking(X,Phi,kbT)
    return spks_rep

spk_rep = spiking_rep(X,Phi,kbT,50)

# %%
def nMF_NS(si,kbT):
    rep,N,T = si.shape
    mT = np.tile(np.mean(si,0),(rep,1,1))
    dsi = si - mT
    Cij = np.zeros((N,N,T))
    Dij = np.zeros((N,N,T))
    for tt in range(T-1):
        Cij[:,:,tt] = dsi[:,:,tt].T @ dsi[:,:,tt] / T
        Dij[:,:,tt] = dsi[:,:,tt].T @ dsi[:,:,tt+1] / T
    mm = np.mean(si[:,:,1:],0)
    Aij_nMF = 1/kbT*np.array([np.diag(1-mm[:,ii]**2) for ii in range(mm.shape[1])]).T
    #1/kbT*(1-mm**2) #1/kbT*np.diag(1-mm**2)
    B = np.mean((Aij_nMF*Cij[:,:,:-1]),2)  #np.mean(Aij_nMF @ Cij,2)
    D = np.mean(Dij,2)
    J_nMF = D @ np.linalg.pinv(B)
    return J_nMF

# %% testing~
eps = 10**-15
def nMF_dec_rep(kbT, X, Phi, si, cond):
    if cond=="MF":
        JJ = nMF_NS(si,kbT)
        #JJ = -np.linalg.pinv(np.cov(si))
    elif cond=="rand":
        JJ = np.random.randn(N,N)
    mm = np.mean(si,0)
    boundm = mm.copy()
    boundm[np.where(boundm>0)] = boundm[np.where(boundm>0)]-eps
    boundm[np.where(boundm<0)] = boundm[np.where(boundm<0)]+eps
    h_nMF = kbT*np.arctanh(boundm) - JJ @ boundm
    X_rec = (Phi @ h_nMF).T
#    import numpy.ma as ma
#    a=ma.masked_invalid(A)
#    b=ma.masked_invalid(B)
#    msk = (~a.mask & ~b.mask)
#    print(ma.corrcoef(a[msk],b[msk]))
    cof = np.corrcoef(X.reshape(-1),X_rec.reshape(-1))[0][1]
    return cof

rep = 50
si = spiking_rep(X,Phi,.1,rep)
Ts = np.arange(0.1,5,0.5)
decs_MF = np.array([nMF_dec_rep(tt, X, Phi, si, "MF") for tt in Ts])
decs_rand = np.array([nMF_dec_rep(tt, X, Phi, si, "rand") for tt in Ts])

plt.figure()
plt.plot(1/Ts,np.abs(decs_MF),'-o')
plt.plot(1/Ts,decs_rand,'-o')
plt.figure()
plt.plot(1/Ts,(decs_MF-decs_rand)/decs_MF,'-o')

# %%
# run kinetic Ising model as above with J_nMF
kbT = 1.1
si = spiking_rep(X,Phi,kbT,rep)
mm = np.mean(si,0)
J_nMF = nMF_NS(si,kbT)
J_rand = np.random.randn(N,N)
# generate spikes and use Phi @ S to reconstruct stimulus
#S_gen = Kinetic_Ising(X, Phi, J_nMF, kbT)
#X_rec = (Phi @ S_gen).T
h_nMF = kbT*np.arctanh(mm-eps) - J_nMF @ (mm-eps)
X_rec = (Phi @ h_nMF).T

plt.figure()
plt.plot(X,X_rec,'o')

# %% iterative version
###############################################################################
#ground truth
rep = 20
si = np.zeros((rep, N, T))
for r in range(rep):
    si[r,:,:] = Kinetic_Ising(X,Phi,J,kbT)

# %%
#initial condition and setup
eps = 10**-16*1
mm = np.mean(si,0)
mT = np.tile(np.mean(si,0),(rep,1,1))
dsi = si - mT
Cij = np.zeros((N,N,T))
Dij = np.zeros((N,N,T))
for tt in range(T-1):
    Cij[:,:,tt] = dsi[:,:,tt].T @ dsi[:,:,tt] /rep
    Dij[:,:,tt] = dsi[:,:,tt].T @ dsi[:,:,tt+1] /rep
mm = np.mean(si[:,:,1:],0)
Aij_nMF = 1/kbT*np.array([np.diag(1-mm[:,ii]**2) for ii in range(mm.shape[1])]).T
B = np.mean((Aij_nMF*Cij[:,:,:-1]),2)
D = np.mean(Dij,2)
J_nMF = D @ np.linalg.pinv(B)
boundm = mm.copy()
boundm[np.where(boundm>0)] = boundm[np.where(boundm>0)]-eps
boundm[np.where(boundm<0)] = boundm[np.where(boundm<0)]+eps
h_nMF = kbT*np.arctanh(boundm) - J_nMF @ boundm
Ji, Hi, mi = copy.deepcopy(J_nMF), copy.deepcopy(h_nMF), copy.deepcopy(boundm)
#iterations
itt = 100
for ii in range(itt):
    ##
    Ai = 1/kbT*np.array([np.ones((N,N))*(1-mi[:,jj]**2) for jj in range(mi.shape[1])]).T  #diagonal
    
    Bi = np.mean((Ai*Cij[:,:,:-1]),2)
    Ji = D @ np.linalg.pinv(Bi)  #main update
    #boundm = copy.deepcopy(mi)
    boundm[np.where(boundm>0)] = boundm[np.where(boundm>0)]-eps
    boundm[np.where(boundm<0)] = boundm[np.where(boundm<0)]+eps
    Hi = kbT*np.arctanh(boundm) - Ji @ boundm ##########################might be wrong here~~
    uu = Hi + Ji @ boundm
    mi = np.tanh(1/kbT*uu)

# %% test (nMF method from Roudi&Hertz, 2011)
mT = np.tile(np.mean(si,0),(rep,1,1))
dsi = si - mT
Cij = np.zeros((N,N,T))
Dij = np.zeros((N,N,T))
for tt in range(T-1):
    Cij[:,:,tt] = dsi[:,:,tt].T @ dsi[:,:,tt] /rep
    Dij[:,:,tt] = dsi[:,:,tt].T @ dsi[:,:,tt+1] /rep
mi = np.mean(si[:,:,1:],0)
D = np.mean(Dij,2)
boundm = mi.copy()
boundm[np.where(boundm>0)] = boundm[np.where(boundm>0)]-eps
boundm[np.where(boundm<0)] = boundm[np.where(boundm<0)]+eps
#mi = boundm.copy()
Bi_ = np.zeros((N,N,N,T-1))  #ikjt
for nn in range(N):
    ai = 1/kbT*np.array([np.ones((N,N))*(1-mi[nn,jj]**2) for jj in range(mi.shape[1])]).T
    Bi_[nn,:,:,:] = ai*Cij[:,:,:-1]
Bi = np.mean(Bi_,3)
Jij = np.zeros((N,N))
for ii in range(N):
    for jj in range(N):
        B = np.linalg.pinv(Bi[ii,:,:])
        Jij[ii,jj] = np.dot(D[ii,:],B[:,jj])
    
#Jij[nn,:] = D[nn,:] @ np.linalg.pinv(Bi[nn,:,:])#np.sum((D[nn,:] @ np.linalg.pinv(Bi[nn,:,:])),1)

# %%
###############################################################################
###############################################################################
# %%
# move on to higher-order TAP method~~

# %% Coordinate-descending
#optimize output layer
        
#optimize connectivity


# %% MF non-stationary inference
#for ii in range(N):
#    for ti in range(T):
#        Del_ = Del0.copy()
#        while dd < tol:
#            for tj in range(T):
#                Del = Del_.copy()
#                mt = 0
#                at = 0
#            for jj in range(N):
#                for kk in range(N):
#                    K[kk,jj] = np.mean(at @ C[kk,jj,:])
#            Ki = np.linalg.pinv(K)
            
# %% FORCE
###############################################################################
#try with the same algorithm but asymmetric inverse updates


