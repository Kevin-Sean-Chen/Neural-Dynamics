# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:02:48 2021

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
matplotlib.rc('xtick', labelsize=35) 
matplotlib.rc('ytick', labelsize=35) 

import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

# %% Kinetic Ising model
###############################################################################
# %% Stimuli
T = 1000
D = 3
smooth = 10
noise = np.random.randn(T,D)
X = noise.copy()
for ii in range(D):
    X[:,ii] = np.convolve(noise[:,ii],np.ones(smooth),'same')/smooth *1
    #X[:,ii] = np.convolve(noise[:,1],np.ones(smooth),'same') + np.random.randn(T)*3  #correlated

# %% Network settings
N = 50
Phi = np.random.randn(D,N)
J = np.random.randn(N,N)/N**0.5 /1
#J = J-np.diag(J)
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

kbT = .2
S = Kinetic_Ising(X,Phi,J,kbT)

plt.figure()
plt.imshow(S[:,:], aspect='auto')
plt.xlabel('time',fontsize=40)
plt.ylabel('cell',fontsize=40)

# %% Iterations
###############################################################################
# %% initialization
si = copy.deepcopy(S)
Cij = np.cov(si)
J0 = np.linalg.pinv(Cij)

# %% looping
its = 500
gamma = 0.1
Jij = copy.deepcopy(J0)*10  #Jij connectivity
Jij = np.cov(S)
ht = 1*(X @ Phi).T #np.zeros((N,T))  #local field through time
Ls = np.zeros(its)
beta = 1#1/kbT
for ii in range(its):
    dLt = 0
    dLs = 0
    for tt in range(T-1):
        #ht[:,tt] = X[tt,:] @ Phi  #cheating here for now
        ht[:,tt] = ht[:,tt] + (beta)*gamma*(si[:,tt+1] - np.tanh(beta*Current(ht[:,tt],Jij,si[:,tt])))
        dLt = dLt + (beta)*(si[:,tt+1][:,None] - np.tanh(beta*Current(ht[:,tt]*0,Jij,si[:,tt]))[:,None]) @ si[:,tt][:,None].T
        dLs = dLs + ht[:,tt]*si[:,tt+1] - np.log(2*np.cosh(ht[:,tt]))
    dL = dLt/T
    Jij = Jij + gamma*dL
    Ls[ii] = np.sum(dLs)

# try without input: w/o x in current

# %% Jij reconstruction
#Jij = Jij - np.diag(Jij)
m = Jij.shape[0]
idx = (np.arange(1,m+1) + (m+1)*np.arange(m-1)[:,None]).reshape(m,-1)
out = Jij.ravel()[idx]

#out = iter_NE(si,500,gamma)
plt.figure()
plt.plot(J[:],out[:],'k.',Markersize=15)
plt.xlabel(r'$J_{ij}$',fontsize=40)
plt.ylabel('$\hat{J_{ij}}$',fontsize=40)


# %% MF tests
J_nMF = nMF(si,kbT)
plt.figure()
plt.plot(J[:],J_nMF[:],'k.',Markersize=15)
plt.xlabel(r'$J_{ij}$',fontsize=40)
plt.ylabel('$\hat{J_{ij}}_{MF}$',fontsize=40)
#plt.xlim([-0.035,0.035])
#plt.ylim([-0.035,0.035])

# %% X stimuli reconstruction
X_rec = ht.T @ np.linalg.pinv(Phi)
plt.figure()
plt.plot(X[:],X_rec[:],'k.')
plt.xlabel(r'$X$',fontsize=40)
plt.ylabel('$\hat{X}$',fontsize=40)

# %% correlation structure
Corr = np.cov(ht)
J_wod = Jij.copy()
np.fill_diagonal(Corr,np.ones(N)*np.nan)
np.fill_diagonal(J_wod,np.ones(N)*np.nan)
plt.figure()
plt.plot(Corr[:],J_wod,'k.',markersize=15)
plt.xlabel('$Cov(\Phi x)_{ij}$',fontsize=40)
plt.ylabel('$J^*_{ij}$',fontsize=40)

# %% Repeating trials
# %%
###############################################################################
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
spks = spiking(X,Phi,.1)
si = spks.copy()

plt.figure()
plt.imshow(si, aspect='auto')

# %% some analysis
def AS(J):
    S = 0.5*(J+J.T)
    A = 0.5*(J-J.T)
    asym = np.linalg.norm(A)/np.linalg.norm(S)
    return asym

def iter_NE(si,its,gamma):
    Cij = np.cov(si)
    J0 = np.linalg.pinv(Cij)
    Jij = copy.deepcopy(J0)  #Jij connectivity
    ht = (X @ Phi).T #np.zeros((N,T))  #local field through time
    for ii in range(its):
        dLt = 0
        for tt in range(T-1):
            #ht[:,tt] = X[tt,:] @ Phi  #cheating here for now
            ht[:,tt] = ht[:,tt] + (beta)*gamma*(si[:,tt+1] - np.tanh(beta*Current(ht[:,tt],Jij,si[:,tt])))
            dLt = dLt + (beta)*(si[:,tt+1][:,None] - np.tanh(beta*Current(ht[:,tt],Jij,si[:,tt]))[:,None]) @ si[:,tt][:,None].T
        dL = dLt/T
        Jij = Jij + gamma*dL
    return Jij

ees = 10**-5
def iter_decoding(J, kbT, X, Phi, si):
    boundm = si.copy()
    boundm[np.where(boundm>0)] = boundm[np.where(boundm>0)]-ees
    boundm[np.where(boundm<0)] = boundm[np.where(boundm<0)]+ees
#    h_nMF = 1/kbT*np.arctanh(boundm) - J @ boundm
    h_nMF = np.arctanh(boundm) - J @ boundm  #not real mean field!!
    h_nMF[np.isinf(h_nMF)] = 0  #removal here
    X_rec = (Phi @ h_nMF).T
    cof = np.corrcoef(X.reshape(-1),X_rec.reshape(-1))[0][1]
    return cof

def stim_gen(par):
    X = np.random.randn(T,D)
    for ii in range(D):
        X[:,ii] = np.convolve(noise[:,ii],np.ones(par),'same')
        #X[:,ii] = np.convolve(noise[:,1],np.ones(100),'same') + np.random.randn(T)*par  #correlated
    return X

def stim_SDE(a,tau):
    #D = 2  ## for 2D-cross-correlation
    X = np.zeros((T,D))
    A = np.ones((D,D))*a
    np.fill_diagonal(A,-np.ones(D))
    dt = 1
    for tt in range(T-1):
        X[tt+1,:] = X[tt,:] + dt*(A @ X[tt,:] + np.random.randn(D))
    return X

def stim_D(D):
    noise = np.random.randn(T,D)
    X = noise.copy()
    for ii in range(D):
        X[:,ii] = np.convolve(noise[:,ii],np.ones(smooth),'same')/smooth *1
    return X

def EP(J,S):
    jj = J - J.T
    dd = S[:,0:-2] @ S[:,1:-1].T / S.shape[1]
    ep = np.sum(jj @ dd)
    return ep

# %%
def stim_bistable(par):
    X = np.random.randn(T,D)
    for ii in range(D):
        X[:,ii] = np.convolve(noise[:,ii],np.ones(100),'same')
    state = np.sin(np.arange(0,T)/50)
    pos_h = np.where(state>0.5)[0]
    pos_l = np.where(state<=0.5)[0]
    X[pos_h,:] = X[pos_h,:]+par
    X[pos_l,:] = X[pos_l,:]-par
    
    return X

X = stim_bistable(20)
plt.figure()
plt.subplot(121)
plt.plot(X)
aa,bb=np.histogram(sp.stats.zscore(X).reshape(-1),bins=100)
plt.subplot(122)
plt.plot(bb[:-1],aa)
p_x = aa/sum(aa)
dU = -np.log(p_x[np.argmin(np.abs(bb-0))])
print(dU)

# %%
plt.figure()
aa,bb=np.histogram(sp.stats.zscore(X).reshape(-1),bins=100)
plt.plot(bb[:-1],aa/np.sum(aa))
plt.ylabel('P(x)',fontsize=40)

# %% test with bistability
bb = np.array([0,10,20,30,35])
px0 = np.zeros(len(bb))
asyms = np.zeros(len(bb))
cofs = np.zeros(len(bb))
control = np.zeros(len(bb))
eps = np.zeros(len(bb))
for ii in range(len(bb)):
    X = stim_bistable(bb[ii])
    si = spiking(X, Phi, 0.5)  #different targeted spiks
    Jij = iter_NE(si,its,gamma)  #NE inference
    asyms[ii] = AS(Jij)  #record asymmetry
    cofs[ii] = iter_decoding(Jij, 0.5, X, Phi, si)
    control[ii] = iter_decoding(np.random.randn(N,N)/N**0.5, 0.5, X, Phi, si)
    eps[ii] = EP(Jij, si)
    
    aa,binn=np.histogram(sp.stats.zscore(X).reshape(-1),bins=100)
    p_x = aa/sum(aa)
    neglogdU = p_x[np.argmin(np.abs(binn-0))]
    px0[ii] = neglogdU

#bar...0.2,5,0.7...
# plt.plot(px0, eps) 
# %% temperature
X = stim_gen(100)
Ts = np.arange(0.1,1.2,0.15)
asyms = np.zeros(len(Ts))
cofs = np.zeros(len(Ts))
control = np.zeros(len(Ts))
eps = np.zeros(len(Ts))
for tt in range(len(Ts)):
    si = spiking(X, Phi, Ts[tt])  #different targeted spiks
    beta = 1/Ts[tt]  ###testing
    Jij = iter_NE(si,its,gamma)  #NE inference
    asyms[tt] = AS(Jij)  #record asymmetry
    cofs[tt] = iter_decoding(Jij, beta, X, Phi, si)
    control[tt] = iter_decoding(np.random.randn(N,N), beta, X, Phi, si)
    eps[tt] = EP(Jij, si)
    
# %% correlation
beta = 1/0.5
ss = np.array([5,10,50,100,150,200,450])*1
decs_ss = np.zeros(len(ss))
asyms = np.zeros(len(ss))
cofs = np.zeros(len(ss))
control = np.zeros(len(ss))
eps = np.zeros(len(ss))
for ii in range(len(ss)):
    X = stim_gen(ss[ii])
    si = spiking(X, Phi, 0.5)  #different targeted spiks
    Jij = iter_NE(si,its,gamma)  #NE inference
    asyms[ii] = AS(Jij)  #record asymmetry
    cofs[ii] = iter_decoding(Jij, 0.5, X, Phi, si)
    control[ii] = iter_decoding(np.random.randn(N,N), 0.5, X, Phi, si)
    eps[ii] = EP(Jij, si)

# %% Xcorr
#aa = np.array([0.2,0.3,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49])
aa = np.array([0.2,0.3,0.4, 0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48])
decs_ss = np.zeros(len(aa))
asyms = np.zeros(len(aa))
cofs = np.zeros(len(aa))
control = np.zeros(len(aa))
eps = np.zeros(len(aa))
for ii in range(len(aa)):
#    Phi_2d = np.random.randn(2,N)
    X = stim_SDE(aa[ii],100)
    si = spiking(X, Phi, 0.5)  #different targeted spiks
    Jij = iter_NE(si,its,gamma)  #NE inference
    asyms[ii] = AS(Jij)  #record asymmetry
    cofs[ii] = iter_decoding(Jij, 0.5, X, Phi, si)
    control[ii] = iter_decoding(np.random.randn(N,N), 0.5, X, Phi, si)
    eps[ii] = EP(Jij, si)
    
    
# %% dimensionality
nn = np.arange(1,10,1)
asyms = np.zeros(len(nn))
cofs = np.zeros(len(nn))
control = np.zeros(len(nn))
eps = np.zeros(len(nn))
for ii in range(len(nn)):
    phi = np.random.randn(nn[ii],N)
    xx = stim_D(nn[ii])
    ### way to generate higher-D stimuli here
    si = spiking(xx, phi, .5)  #different targeted spiks
    Jij = iter_NE(si,its,gamma)  #NE inference
    asyms[ii] = AS(Jij)  #record asymmetry
    cofs[ii] = iter_decoding(Jij, .5, xx, phi, si)
    control[ii] = iter_decoding(np.random.randn(N,N)/N**0.5, .5, xx, phi, si)
    eps[ii] = EP(Jij, si)

# %% scaling
dd = np.arange(1,10,1)
asyms = np.zeros(len(dd))
cofs = np.zeros(len(dd))
control = np.zeros(len(dd))
eps = np.zeros(len(dd))

Phi = np.random.randn(D,N)
X = stim_D(D)
si = spiking(X, Phi, 0.5)  #different targeted spiks
Jij = iter_NE(si,its,gamma)  #NE inference

for ii in range(len(dd)):
    si_ = si[:ii,:]
    Jij_ = Jij[:ii,:ii]
    Phi_ = Phi[:,:ii]
    asyms[ii] = AS(Jij_)  #record asymmetry
    cofs[ii] = iter_decoding(Jij_, 0.5, X, Phi_, si_)
    control[ii] = iter_decoding(np.random.randn(ii,ii)/ii**0.5, 0.5, X, Phi_, si_)
    eps[ii] = EP(Jij_, si_)

# %%
param = nn.copy()
plt.figure()
plt.plot(param,asyms,'-o',markersize=15)
plt.xlabel(r'$\sigma$',fontsize=40)
plt.ylabel(r'$\eta$',fontsize=40)
plt.figure()
plt.plot(param,np.abs(eps),'-o',markersize=15)
plt.xlabel(r'$\sigma$',fontsize=40)
plt.ylabel('$EP$',fontsize=40)

# %%
plt.figure()
plt.plot(param,cofs/control,'-o',markersize=15)
plt.xlabel(r'$\sigma$',fontsize=40)
plt.ylabel('$D^*/D_{ind}$',fontsize=40)

# %%
plt.figure()
plt.plot(param, cofs*1,'-o',markersize=15)
plt.xlabel(r'$\sigma$',fontsize=40)
plt.ylabel('$D$',fontsize=40)
