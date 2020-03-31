# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:22:13 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dotmap as DotMap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5

# %% functions
eps = 10**-15
    
def NL(x):
    """
    Poinsson emission with logistic nonlinearity
    """
    nl = np.random.poisson(10/(1+np.exp(-x)))
#    nl = np.random.poisson(np.log(1+np.exp(x)+eps))
#    nl = np.array([max(min(100,xx),0) for xx in x])  #ReLu
    return nl

def neglog(theta, Y, X, pad, nb):
    """
    negative log-likelihood to optimize theta (parameters for kernel of all neurons)
    with neural responses time length T x N neurons, and the padding window size
    return the neg-ll value to be minimized
    """
#    T,N = Y.shape
#    dc = theta[:N]  #the DC compontent (baseline fiting) that's not dynamic-dependent
#    wo_dc = theta[N:]  #parameters to construct kernels
#    pars_per_cell = int(len(wo_dc)/(N))  #the number of parameters per neurons (assuming they're the same)
#    kernels_per_cell = int(pars_per_cell/nb)  #the number of kernels per neuron (assuming they're the same)
#    theta_each = np.reshape(wo_dc, (N, kernels_per_cell, nb))  #N x kernels x nb
#    k = np.array([kernel(theta_each[nn,kk,:], pad) for nn in range(0,N) for kk in range(0,kernels_per_cell)])  #build kernel for each neuron
#    k = k.reshape((N, kernels_per_cell*pad))  # N x (kernel_per_cell*pad)
#    k =  np.concatenate((dc[:,None], k),axis=1).T # adding back the DC baseline
    
    k = kernel(theta,pad)
    v = NL(X @ k)  #nonlinear function
    nl_each = -(np.matmul(Y.T, np.log(v+eps)) - np.sum(v))  #Poisson negative log-likelihood
    #nl_each = -( np.matmul(Y.T, np.log(v+eps)) - np.matmul( (1-Y).T, np.log(1-v+eps)) )  #Bernouli process of binary spikes
    nl = nl_each#.sum()
    return nl


def flipkernel(k):
    """
    flipping kernel to resolve temporal direction
    """
    return np.squeeze(np.fliplr(k[None,:])) ###important for temporal causality!!!??
    
def kernel(theta, pad):
    """
    Given theta weights and the time window for padding,
    return the kernel contructed with basis function
    """
    nb = len(theta)
    basis = basis_function1(pad, nb)  #construct basises
    k = np.dot(theta, basis.T)  #construct kernels with parameter-weighted sum
    return flipkernel(k)

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

def basis_function2(n, k, tl):
    """
    More biophysical delayed function, given a width parameter n, location of kernel k,
    and the time window tl (n=5-10 is a normal choice)
    """
    beta = np.exp(n)
    fkt = beta*(tl/k)**n*np.exp(-n*(tl/k))
    return fkt

def build_matrix(stimulus, spikes, pad, couple):
    """
    Given time series stimulus (T time x N neurons) and spikes of the same dimension and pad length,
    build and return the design matrix with stimulus history, spike history od itself and other neurons
    """
    T, N = spikes.shape  #neurons and time
    SN = stimulus.shape[0]  #if neurons have different input (ignore this for now)
    
    # Extend Stim with a padding of zeros
    Stimpad = np.concatenate((stimulus,np.zeros((pad,1))),axis=0)
    # Broadcast a sampling matrix to sample Stim
    S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
    X = np.squeeze(Stimpad[S])
    if couple==0:
        X = X.copy()
        X = np.concatenate((np.ones((T,1)), X),axis=1)
    elif couple==1:
        X_stim = np.concatenate((np.ones((T,1)), X),axis=1)  #for DC component that models baseline firing
    #    h = np.arange(1, 6)
    #    padding = np.zeros(h.shape[0] - 1, h.dtype)
    #    first_col = np.r_[h, padding]
    #    first_row = np.r_[h[0], padding]
    #    H = linalg.toeplitz(first_col, first_row)
        
        # Spiking history and coupling
        spkpad = np.concatenate((spikes,np.zeros((pad,N))),axis=0)
        # Broadcast a sampling matrix to sample Stim
        S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
        X_h = [np.squeeze(spkpad[S,[i]]) for i in range(0,N)]
        # Concatenate the neuron's history with old design matrix
        X_s_h = X_stim.copy()
        for hh in range(0,N):
            X_s_h = np.concatenate((X_s_h,X_h[hh]),axis=1)
        X = X_s_h.copy()
#        #print(hh)
    
    return X

def build_convolved_matrix(stimulus, spikes, Ks, couple):
    """
    Given stimulus and spikes, construct design matrix with features being the value projected onto kernels in Ks
    stimulus: Tx1
    spikes: TxN
    Ks: kxpad (k kernels with time window pad)
    couple: binary option with (1) or without (0) coupling
    """
    T, N = spikes.shape
    k, pad = Ks.shape
    
    Stimpad = np.concatenate((stimulus,np.zeros((pad,1))),axis=0)
    S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
    Xstim = np.squeeze(Stimpad[S])
    Xstim_proj = np.array([Xstim @ Ks[kk,:] for kk in range(k)]).T
    
    if couple==0:
        X = np.concatenate((np.ones((T,1)), Xstim_proj),axis=1)
    elif couple==1:
        spkpad = np.concatenate((spikes,np.zeros((pad,N))),axis=0)
        Xhist = [np.squeeze(spkpad[S,[i]]) for i in range(0,N)]
        Xhist_proj = [np.array([Xhist[nn] @ Ks[kk,:] for kk in range (k)]).T for nn in range(N)]
        
        X = Xstim_proj.copy()
        X = np.concatenate((np.ones((T,1)), X),axis=1)
        for hh in range(0,N):
            X = np.concatenate((X,Xhist_proj[hh]),axis=1)
    return X

# %% ground truth GLM
T = 1000
stim = np.random.randn(T,1)*1
#stim[stim>0]=1
#stim[stim<=0] = 0
nb = 5
ks = np.random.randn(nb)
pad = 30
K = kernel(ks,pad)[:,None]
spk = NL(np.convolve(stim[:,0],K[:,0],'same'))[:,None]
X = build_matrix(stim,spk,pad,0)
spk = NL(X @ K)

plt.figure()
plt.subplot(121)
plt.plot(K)
plt.subplot(122)
plt.plot(spk)

# %% inference
X = build_matrix(stim,spk,pad,0)
npars = nb
theta0 = 1*np.random.randn(npars)  #np.ones(npars) #ks + 
res = sp.optimize.minimize( neglog, theta0, args=(spk, X, pad, nb), method='Nelder-Mead',options={'disp':True,'maxiter':1000})#, method="L-BFGS-B", tol=1e-10)#, options={'disp':True,'gtol':1e-2})
theta_infer = res.x

# %%
plt.figure()
plt.plot(K)
plt.plot(kernel(theta_infer,pad),'-o')
plt.plot(kernel(theta0,pad),'--')

# %%
###############################################################################
###############################################################################
# %% test pyGLM
from pyglmnet import GLMCV

### This is super-easy if we rely on built-in GLM fitting code
glm = GLMCV(distr="binomial", tol=1e-3,
            score_metric="pseudo_R2",
            alpha=1.0, learning_rate=3, max_iter=100, cv=3, verbose=True)

glm.fit(X, np.squeeze(spk))

# %% 
plt.figure()
pyglm_infer = glm.beta_
plt.plot(pyglm_infer/np.linalg.norm(pyglm_infer))
plt.plot(K/np.linalg.norm(K),'--')

# %% Two-neuron circuit with pyGLMnet
###############################################################################
###############################################################################
# %% combine basis function here
###convolve with basis first??
N = 2
dt = 0.1  #ms
T = 3000
time = np.arange(0,T,dt)
lt = len(time)

x = np.zeros((N,lt))  #voltage
spk = np.zeros_like(x)  #spikes
syn = np.zeros_like(x)  #synaptic efficacy
syn[:,0] = np.random.rand(N)
rate = np.zeros_like(x)  #spike rate
x[:,0] = np.random.randn(N)*1
#J = np.random.randn(N,N)
J = np.array([[1.5, -.5],\
              [-1.5, 1.]])
#J = np.array([[1, -0],\
#          [-0, 1.]])
J = J.T*5
noise = 0.
stim = np.random.randn(lt)*20  #np.random.randn(N,lt)*20.
taum = 2  #5 ms
taus = 50  #50 ms
E = 1

eps = 10**-15
def LN(x):
    """
    nonlinearity
    """
    ln = 1/(1+np.exp(-x*1.+eps))   #logsitic
#    ln = np.log(1+np.exp(x)+eps)
#    ln = np.array([max(min(100,xx),0) for xx in x])  #ReLu
    return ln  #np.random.poisson(ln) #ln  #Poinsson emission

def spiking(ll,dt):
    """
    Given Poisson rate (spk per second) and time steps dt return binary process with the probability
    """
    N = len(ll)
    spike = np.random.rand(N) < ll*dt  #for Bernouli process
    return spike

###iterations for neural dynamics
for tt in range(0,lt-1):
    x[:,tt+1] = x[:,tt] + dt/taum*( -x[:,tt] + (np.matmul(J,LN(syn[:,tt]*x[:,tt]))) + stim[tt]*np.array([1,1]) + noise*np.random.randn(N)*np.sqrt(dt))
    spk[:,tt+1] = spiking(LN(x[:,tt+1]),dt)
    rate[:,tt+1] = LN(x[:,tt+1])
    syn[:,tt+1] = 1#syn[:,tt] + dt*( (1-syn[:,tt])/taus - syn[:,tt]*E*spk[:,tt] )
    
plt.figure()
plt.subplot(411)
plt.imshow(spk,aspect='auto');
plt.subplot(412)
plt.imshow(rate,aspect='auto');
plt.subplot(413)
plt.plot(time,x.T);
plt.xlim([0,time[-1]])
plt.subplot(414)
plt.plot(time,stim.T);
plt.xlim([0,time[-1]])

# %% raw points
pad = 50
X = build_matrix(stim[:,None], spk.T, pad, 1)
glm = GLMCV(distr="binomial", tol=1e-5,
            score_metric="pseudo_R2",
            alpha=0., learning_rate=0.1, max_iter=100, cv=3, verbose=True)
glm.fit(X, np.squeeze(spk[0,:]))

# %%
yhat = simulate_glm('binomial', glm.beta0_, glm.beta_, X)
plt.plot(yhat)

# %%
#reg_lambda = np.logspace(np.log(1e-6), np.log(1e-8), 100, base=np.exp(1))
#glm_poissonexp = GLMCV(distr='poisson', verbose=False, alpha=0.05,
#            max_iter=100, learning_rate=2e-1, score_metric='pseudo_R2',
#            reg_lambda=reg_lambda, eta=4.0)
#glm_poissonexp.fit(X, np.squeeze(spk[0,:]))

# %%
# %%
# %% test with basis function
pad = 100  #window for kernel
nbasis = 7  #number of basis
couple = 1
Y = np.squeeze(spk[0,:])  #spike train of interest
#Ks = basis_function1(pad, nbasis).T  #basis function for projection
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
stimulus = stim[:,None]
X = build_convolved_matrix(stimulus, spk.T, Ks, couple)
glm = GLMCV(distr="binomial", tol=1e-8,
            score_metric="pseudo_R2",
            alpha=0., learning_rate=0.001, max_iter=100, cv=3, verbose=True)
glm.fit(X, Y)


# %% direct simulation
yhat = simulate_glm('binomial', glm.beta0_, glm.beta_, X)
plt.figure()
plt.subplot(211)
plt.plot(yhat)
plt.subplot(212)
plt.plot(spiking(LN(yhat),dt))
plt.plot(Y*0.5,'--')

# %% unpacking basis
theta = glm.beta_
dc_ = theta[0]
theta_ = theta[1:]
if couple == 1:
    theta_ = theta_.reshape(nbasis,N+1)  #nbasis times (stimulus + N neurons)
    allKs = np.array([theta_[:,kk] @ Ks for kk in range(N+1)])
elif couple == 0:
    allKs = Ks.T @ theta_

plt.figure()
plt.plot(allKs.T)

# %% reconstruction
rec_Ks = allKs.reshape(-1)+1
X_rec = build_matrix(stimulus, spk.T, pad, couple)[:,1:]
conv_ = X_rec @ rec_Ks
spk_rec = spiking(LN(conv_+dc_-0),dt)

plt.figure()
plt.plot(Y)
plt.plot(spk_rec-1,'--')

# %%
### GLM network simulation
def GLM_net(allK, dcs, S):
    """
    Simulate a GLM network given all response and coupling kernels and the stimulus
    """
    N, K, h = allK.shape  #N neurons x K kernels x pad window
    _,T = S.shape  #all the same stimulus for now
    us = np.zeros((N,T))  #all activity through time
    spks = np.zeros((N,T))  #for spiking process
    K_stim = allK[:,0,:]  # N x pad response kernels
    K_couple = allK[:,1:,:]  # N x N xpad coupling filter between neurons (and itself)
    #K_couple.transpose(1, 0, 2).strides
    
    for tt in range(h,T):
        ut = np.einsum('ij,ij->i', S[:,tt-h:tt], (K_stim)) + \
             np.einsum('ijk,ik->i',  (K_couple), spks[:,tt-h:tt])  #neat way for linear dynamics
        ut = LN(ut + dcs)  #share the same nonlinearity for now
        #ut = spiking(LN(us[:,tt]+dcs),dt)
        us[:,tt] = ut #np.random.poisson(ut)
        spks[:,tt] = spiking(us[:,tt],dt)
    return us, spks

# %%
#from pyglmnet import GLM, simulate_glm
#
#n_samples, n_features = 1000, 100
#distr = 'poisson'
#
## random sparse regressors
#beta0 = np.random.rand()
#beta = sp.sparse.random(1, n_features, density=0.2).toarray()[0]
## simulate data
#Xtrain = np.random.normal(0.0, 1.0, [n_samples, n_features])
#ytrain = simulate_glm('poisson', beta0, beta, Xtrain)
#Xtest = np.random.normal(0.0, 1.0, [n_samples, n_features])
#ytest = simulate_glm('poisson', beta0, beta, Xtest)
#
## create an instance of the GLM class
#glm = GLM(distr='probit')
#
## fit the model on the training data
#glm.fit(X, spk[0,:])
#
## predict using fitted model on the test data
#yhat = glm.predict(X)
