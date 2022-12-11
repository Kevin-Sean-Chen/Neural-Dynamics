#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:22:19 2022

@author: kschen
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy.sparse import spdiags
from scipy import sparse
from scipy.optimize import minimize

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)

# %% Goals:
    # 1. code up LDS with spiking output
    # 2. do inference to recover latent and parameters
    # 3. add learning rule to learn the recurrent signal that captures latent dynamics
    # 4. ... if possible, generalize or compare different latents! (continuous GP or discrete switching)
    
# %% LDS setup
# dimensions
nz = 2  # latent dimension
ny = 10  # observation dimension
ni = 1  # input dimension
lt = 1000  # time length
# construct A matrix with SVD
gamma = 0.02  # ratio
Sa = np.sort(np.random.rand(nz)*gamma+(1-gamma))[::-1]  # eigenvalues
Ua,ss,vv = np.linalg.svd(np.random.randn(nz,nz))  # ortho vectors
A = Ua @ np.diag(Sa) @ Ua.T
# construct noise covariance
Q = np.random.randn(nz)
Q = 0.01*(np.outer(Q,Q) + np.eye(nz))  # symmetric covariance matrix
def sample_noise():
    return np.random.multivariate_normal(np.zeros(nz), Q)  # multivariate Gaussian noise
# observation matrix
C = 0.5*np.random.randn(ny,nz)
def logistic(x):
    return 1/(1+np.exp(-x))  # logistic for Bernoulli process
def gen_spk(x):
    return np.random.rand(ny)<logistic(x)  # spiking with logistic probability

# input part
B = np.random.randn(nz,ni)
ss = 0.5*np.random.randn(ni,lt)  # external input
muz = B @ ss  # additive intput to latents

# initialization
zz = np.zeros((nz, lt))  # latent
xx = np.zeros((ny, lt))  # projection of latent
yy = np.zeros((ny, lt))  # observation spikes

zz[:,0] = muz[:,0] + sample_noise()
xx[:,0] = C @ zz[:,0] #+ muy[:,0]
yy[:,0] = gen_spk(xx[:,0]) #np.random.rand(ny)<logistic(xx[:,0])


# %% LDS iterations
for tt in range(1,lt):
    zz[:,tt] = A @ zz[:,tt-1] + sample_noise() + muz[:,tt]
    xx[:,tt] = C @ zz[:,tt-1] #+ muy[:,tt]  # without input in the projection for now
    yy[:,tt] = gen_spk(xx[:,tt])

# %% plotting latent and spikes
plt.figure()
plt.subplot(211)
plt.plot(zz.T)
plt.subplot(212)
plt.imshow(yy, aspect='auto')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% Inference!!
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%
class LDSBern_inference():
    def __init__(self, LDS_params, data):
        self.A = LDS_params[0]
        self.B = LDS_params[1]
        self.C = LDS_params[2]
        self.Q = LDS_params[3]
        self.nz = C.shape[1]
        yy,ss = data
        self.yy,self.ss = yy,ss
        self.nT = yy.shape[1]
        self.ny = C.shape[0]
        self.ni = ss.shape[0]
        self.logEvidence = 0
        
    def z_MAP_compute(self, prs=None, zz0=None):
        """
        E-step: MAP estimation for the latent z
        This is done through block-matrix tricks and minimizing the negative log-posterior
        
        When prs is given (not None), this is used for evidence optimization in the M-step
        This is done through
        """
        ### for MAP estimate of the latent
        if zz0 is None:
            zz0 = np.zeros((self.nz,self.nT))
        
        ### for evidence optimization for parameters
        if prs is not None:
            A,C = self.unvecLDSprs(prs, self.ny)
        else:
            A,C = self.A, self.C
            
        # Compute contributions from inputs
        muz = self.B @ self.ss  #additive intput to latents (as column vectors)
        ### sparse block matrix computation
        Am = sparse.kron(spdiags(np.ones((self.nT-1)),-1, self.nT, self.nT),-A) + sparse.eye(self.nz*self.nT)
        Cm = sparse.kron(sparse.eye(self.nT), C)
        Qinv = sparse.kron(sparse.eye(self.nT), np.linalg.inv(self.Q))
        Qinv_tile = Am.T @ (Qinv @ Am)
        
        ### sparse Hessian log-likelihood (interesting method from Pillow)
        nn = np.repeat(np.arange(self.nz)[:,None],self.nz,1).T  # indecies
        ii = nn.T.reshape(-1)[:,None] + np.arange(0,self.nz*self.nT,nz)[None,:]  # roiw indices for Hessian
        jj = nn.reshape(-1)[:,None] + np.arange(0,self.nz*self.nT,nz)[None,:]  # column indices for Hessian
                
        ### setup optimization for posterior
        postargs = [Cm, self.yy.reshape(-1), Qinv_tile, C, ii.reshape(-1), jj.reshape(-1), muz.reshape(-1)]
        result = minimize(self.neg_log_handle, zz0.reshape(-1), args=(postargs))
        zmap = result.x
        z_map = zmap.reshape(zz0.shape)
        
        ### compute the log-evidence
        neglogpost, _, zzHess = self.neg_log_posterior(zmap, postargs)
        
        ### compute log-evidence
        logEv = -neglogpost + 0.5*np.linalg.logdet(Qinv) - 0.5*np.linalg.logdet(zzHess)
            
        return z_map, logEv
    
    def neg_log_handle(self,zz,args):
        """
        Handle to return nll for optimization of expected latent
        """
        nll,_,_ = self.neg_log_posterior(zz,args)
        return nll
    
    def neg_log_posterior(self,zz,args):
        """
        Negative log posterior for the latent z
        This is used for optimization in the MAP estimation function
        """
        # loading
        Cm, yy, Qinv, C, ii, jj, muz = args
        
        # Compute projection of inputs onto GLM weights for each class
        xproj = Cm @ zz # + muy  # "logit" of input to latent
        zzctr = zz - muz  # zero-mean latent
        
        # compute nLL
        f,_,_ = self.softplus(xproj)
        nll = -np.dot(yy,xproj) + np.sum(f)  # neg-log-likelihood
        nll = nll + 0.5*np.sum(zzctr*(Qinv@zzctr))  # neg-log-posterioi
        
        # compute gradient
        f,df,ddf = self.softplus(xproj)  # log normed and derivative
        grad = Cm.T @ (df-yy) + Qinv @ zzctr  # gradient
        
        # compute Hessian
        Cddf = C.T[:,:,None] * np.reshape(ddf,(1,self.ny,-1))
        CddfC = np.array([ci.T @ C for ci in Cddf.T]).T  # weird python way to do pagemtimes...
        Hess = sp.sparse(ii,jj,CddfC.reshape(-1))
        Hess = Hess + Qinv
        return nll, grad, Hess
    
    def run_Mstep_LapEvd(self):
        """
        M-step with Laplace approximation of the evidence
        Update parameters by maximizing the evidence
        """
        # prepare arguments
        prs0 = [self.A.reshape(-1), self.C.reshape(-1)]  # initial params
        # postargs [self.yy, self.ny, self.Q]
        
        # compute MAP estimate
        result = minimize(self.neg_log_evd_handle, prs0)
        prs_hat = result.x
        
        ### parameter update
        Ahat, Chat = self.unvecLDSprs(prs_hat)
        self.A = Ahat
        self.C = Chat
        self.logEvidence = -self.neg_log_evd_handle(prs_hat)
        return None
    
    def neg_log_evd_handle(self, prs0):
        """
        Handle to return evidence for parameter optimization
        """
        zzmap, logevd = self.z_MAP_compute(prs0)
        neglogEvd = -logevd
        return neglogEvd
    
    def unvecLDSprs(self, prs):
        """
        Return matrices from the parameter vectors
        """
        nAprs = self.nz**2
        nACprs = nAprs + self.nz*self.ny  # parameters in A and C matrix
        A = np.reshape(prs[:nAprs], self.nz, self.nz)
        C = np.reshape(prs[nAprs:nACprs], self.ny, self.nz)
        return A,C
    
    def run_LEM(self, iters, tol):
        """
        Main part running EM!
        """
        logEvTrace = np.zeros(iters)
        dlogp, dlogtol, dlogp_prev = np.inf, tol, -np.inf
        jj = 0
        
        while jj<iters and dlogp>dlogtol:
            jj = jj+1
            
            # E-step: optimizing for latent z
            zzmap, logp = self.z_MAP_compute()
            logEvTrace[jj] = logp
            
            # M-step: optimizaing for parameters
            self.run_Mstep_LapEvd()
            
            # update log-likelihood
            dlogp = dlogp - dlogp_prev
            dlogp_prev = dlogp
            
            # display progrss
            print('EM step '+jj+' logP= '+logp)
                
        return None
    
    def softplus(self,x):
        """
        soft-plus function, its derivitive, and checking for overflow
        """
        f = np.log(1+np.exp(x))
        df = np.exp(x)/(1+np.exp(x))
        ddf = np.exp(x)/(1+np.exp(x))**2
        if sum(x.reshape(-1)<-20)>0:  # check small values
            iix = (x.reshape(-1)<-20)
            f[iix] = np.exp(x[iix])
            df[iix] = f[iix]
            ddf[iix] = f[iix]
        if sum(x.reshape(-1)>500)>0:  # check large values
            iix = (x.reshape(-1)>500)
            f[iix] = x[iix]
            df[iix] = 1
            ddf[iix] = 0
        return f, df, ddf

# %% Test inference
LDS_params = [A,B,C,Q]
data = [yy,ss]

lds_inf = LDSBern_inference(LDS_params, data)
lds_inf.run_LEM(10, 0.001)
