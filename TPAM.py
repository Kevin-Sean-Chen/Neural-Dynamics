#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:00:29 2019

@author: kschen
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from pylab import *
import time

#from brian2 import *
#from brian2.units.allunits import henry

#%matplotlib inline

plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'text.usetex': True})
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']
plt.rcParams.update({'font.family': 'serif', 
                     'font.serif':['Computer Modern']})

# %% functions for phasor networks
def crvec(N, D=1):
    """
    Create D x N random phasors drawn from complex plane
    """
    rphase = 2*np.pi * np.random.rand(D, N)
    return np.cos(rphase) + 1.0j * np.sin(rphase)

def norm_range(v):
    """
    Normalization by the complitude
    """
    return (v-v.min())/(v.max()-v.min())

def activation_thresh(x, sigma=0.0, c_thresh=None):
    """
    Nonlinear activation function given a threshold in the phase angle space
    """
    
    if c_thresh is None:
        N = x.shape[0]
        c_thresh = 2.0 / N**0.5
        
    xn = np.abs(x)
    
    a = (x ) / (np.abs(x) + 1e-12)
    a[xn < c_thresh] = 0
    
    return a

def cviz_im(cvec):
    """
    Comvert phasor values to pretty colorful 2D plots
    """
    ss = int(len(cvec)**0.5)
    
    ss_idx = ss**2
    
    im_cvec = np.zeros((ss, ss,3))
#     im_cvec[:,:,3]=1
    c=0
    for i in range(ss):
        for j in range(ss):
            if np.abs(cvec[c]) > 0.05:
                im_cvec[i,j,:] = matplotlib.colors.hsv_to_rgb([(np.angle(cvec[c])/2/pi + 1) % 1, 1, 1])
                
            c+=1
                
    return im_cvec
    
def phase2spikes(cv, freq=5.0):
    st = np.angle(cv) / (2*pi*freq)
    return st

# %% dynamics (from the demo code)
N = 25**2  #number of neurons
D = 201  #number of patterns
Ks = 25  #number of non-zero patterns for sparsity

letter_vectors_c = crvec(N, D)  #complex random phasors

for d in range(D):
    ip = np.random.choice(N, size=(N-Ks), replace=False)  #zero-out activities
    letter_vectors_c[d, ip] = 0

J_c = np.dot(letter_vectors_c.T, np.conj(letter_vectors_c))  #outer-product learning rule
np.fill_diagonal(J_c, 0)

max_steps = 20  #iteration steps
hop_hist = np.zeros((D, max_steps+1))
hop_s_hist = np.zeros((N, max_steps+1), 'complex')

target_idx = 100

hop_state_i = letter_vectors_c[target_idx,:].copy()
hop_state_i[:300] = 0  #hiding half

noise_state = 0.0 * np.random.randn(N) * np.squeeze(crvec(N,1)) #2*(np.random.randn(N) < 0) - 1
hop_state = hop_state_i + noise_state

hop_state /= norm(hop_state)

cols = get_cmap('copper', max_steps)

c_thresh = 0.6/Ks**0.5
for i in range(max_steps):
    hop_hist[:,i] = np.real(np.dot(np.conj(letter_vectors_c), hop_state))    
    hop_s_hist[:,i] = hop_state
    
    hop_u = np.dot(J_c, hop_state) / Ks
    hop_state = activation_thresh(hop_u, c_thresh=c_thresh )
    hop_state /= norm(hop_state)

    plot(hop_hist[:,i], c=cols(i))

hop_hist[:,i+1] =  np.real(np.dot(np.conj(letter_vectors_c), hop_state))
plot(hop_hist[:,i], c=cols(i))


figure(figsize=(6,3))

subplot(131)
imshow(cviz_im(hop_state_i))
title('Initial')
subplot(132)
imshow(cviz_im(hop_s_hist[:, -2]))
title('Converged')
subplot(133)
imshow(cviz_im(letter_vectors_c[target_idx,:]))
title('Target')

# %% capacity scanning
def TPAM(kk, nn, Ph):
    """
    TPAM with kk patterns nn neurons and Ph percentage of sparsity
    Output the training pattern and recalled pattern to compute similarity
    """
    Ks = int(nn*Ph)  #percentage to number of cells with zero-value
    letter_vectors_c = crvec(nn, kk)  #complex random phasors (neuron by patterns)

    for k in range(kk):
        ip = np.random.choice(N, size=(nn-Ks), replace=False)  #zero-out activities
        letter_vectors_c[k, ip] = 0
    
    J_c = np.dot(letter_vectors_c.T, np.conj(letter_vectors_c))  #outer-product learning rule
    np.fill_diagonal(J_c, 0)
    
    max_steps = 30  #iteration steps
    hop_s_hist = np.zeros((N, max_steps+1), 'complex')  #to store complex phasors through iterations
    
    target_idx = np.random.choice(kk,1)[0]  #picking one pattern as the target one
    hop_state_i = letter_vectors_c[target_idx,:].copy()
    hop_state_i[:int(nn/2)] = 0  #hiding half
    
    noise_state = 0.0 * np.random.randn(N) * np.squeeze(crvec(N,1)) #2*(np.random.randn(N) < 0) - 1
    hop_state = hop_state_i + noise_state
    hop_state /= norm(hop_state)  #initilize with noisy patterns
    
    c_thresh = 0.6/Ks**0.5  #threshold of the phasor network
    for i in range(max_steps):
        hop_s_hist[:,i] = hop_state
        hop_u = np.dot(J_c, hop_state) / Ks  #dendritic sum
        hop_state = activation_thresh(hop_u, c_thresh=c_thresh )  #transfer function
        hop_state /= norm(hop_state)  #preserving phase angle
    
    recall = hop_s_hist[:, -2]
    pattern = letter_vectors_c[target_idx,:]
    
    return pattern, recall

def complex_similarity(xx,yy):
    """
    Compute cosine similarity between two complex vectors
    """
    similarity = np.dot(np.conj(xx), yy) / (np.dot(np.conj(xx), xx)*np.dot(np.conj(yy), yy))**0.5  #transfer function 
#    Rs = np.real(xx)-np.real(yy)
#    Is = np.imag(xx)-np.imag(yy)
#    normm = 
#    similarity = np.sqrt(np.sum(Rs**2 + Is**2)) / normm
    return similarity

### fix N scan K
N = 100  #number of neurons
R = 15  #repititions
ks = 50  #scan to the number of patterns stored
Ph = 0.5  #sparsity in ratio
mss = np.zeros((ks,R))

for kk in range(0,mss.shape[0]):  #scan through number of patterns
    for rr in range(0,mss.shape[1]):  #repeat the simulation
        patterns, recall = TPAM(kk+1, N, Ph)  ##Hopfield(kk+1,N)
        mss[kk,rr] = complex_similarity(patterns,recall)
    
#    figure(figsize=(6,3))
#    subplot(211)
#    imshow(cviz_im(patterns))
#    title('Converged')
#    subplot(212)
#    imshow(cviz_im(recall))
#    title('Target')

plt.figure()
plt.plot(np.arange(1,ks+1)/N,np.nanmean(mss,axis=1),'-o');
plt.xlabel('alpha (K/N)',fontsize=25)
plt.ylabel('Similarity (cosine angle)',fontsize=25)

# %% scansparsity
Phs = np.array([0.1,0.2,0.3,0.5])
plt.figure()
for pp in Phs:
    N = 100  #number of neurons
    R = 15  #repititions
    ks = 50  #scan to the number of patterns stored
    Ph = pp  #sparsity in ratio
    mss = np.zeros((ks,R))
    
    for kk in range(0,mss.shape[0]):  #scan through number of patterns
        for rr in range(0,mss.shape[1]):  #repeat the simulation
            patterns, recall = TPAM(kk+1, N, Ph)  ##Hopfield(kk+1,N)
            mss[kk,rr] = complex_similarity(patterns,recall)
    
    plt.plot(np.arange(1,ks+1)/N,np.nanmean(mss,axis=1),'-o',label=str(Ph));
    plt.xlabel('alpha (K/N)',fontsize=25)
    plt.ylabel('Similarity (cosine angle)',fontsize=25)
    plt.legend()


# %%
###############################################################################
# %% Hopfield network
def ms(patterns,recall):
    """
    return the max of normalized dot product between trained and recalled pattern
    """
    dots = np.zeros(patterns.shape[0])  #number of patterns stored
    nn = patterns.shape[1]  #number of neurons in the model
    for m in range(0,patterns.shape[0]):
        dots[m] = np.abs(np.sum(patterns[m,:]*recall))
    return np.max(dots)/nn #(patterns.shape[0]*patterns.shape[1])

def Hopfield(K,N):
    """
    Hopfield network with N neurons and trained with K random spin patterns,
    The output is one recall pattern starting from a random initial condition and the actualy patterns
    """
    ###training  #with N neurons and K patterns
    theta = 0
    ww = 1 #normalization constant?
    Wm = np.zeros((N,N))  #trained weights
    patterns = np.zeros((K,N))  #stored patterns
    for k in range(0,K):
        eps = np.random.choice((-1,1),N)
        patterns[k,:] = eps
        for i in range(0,len(eps)):
            for j in range(0,len(eps)):
                if i == j:
                    Wm[i,j] = 0
                else:
                    Wm[i,j] = Wm[i,j]+ww*eps[i]*eps[j]
                    #Wm[j,i] = Wm[i,j]
    ###recall
    recall = np.random.choice((-1,1),N)
    rep = 1000  #repeat simulation of random updates
    for r in range(0,rep):
        p = np.random.randint(0,N)
        h = np.sign(np.dot(Wm[p,:],recall)-theta)
        recall[p] = h

    return patterns, recall

#fix N scan K
N = 100
R = 15
ks = 50
mss = np.zeros((ks,R))
for kk in range(0,mss.shape[0]):  #scan through number of patterns
    for rr in range(0,mss.shape[1]):  #repeat the simulation
        patterns, recall = Hopfield(kk+1,N)
        mss[kk,rr] = ms(patterns,recall)

plt.plot(np.arange(1,ks+1)/N,np.mean(mss,axis=1),'-o');
plt.xlabel('alpha (K/N)',fontsize=25)
plt.ylabel('Similarity (dot product)',fontsize=25)


