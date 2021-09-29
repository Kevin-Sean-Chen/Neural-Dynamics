# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 21:03:11 2021

@author: kevin
"""

import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
#from torch.utils.data import DataLoader
#from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

# %% Simple rate RNN generative model
dt, T, N, tau, s = 0.01, 100, 20, 1, .5
v1, v2 = np.random.randn(N), np.random.randn(N)
Jij = s*np.sqrt(N)*np.random.randn(N,N) #+ np.outer(v1,v1) + np.outer(v2,v2) +np.outer(v１,v２)
### Mask for sparsity
sparsity = 0.6
M = np.random.rand(N,N)
M[M<sparsity] = 0
M[M > 0] = 1
Jij = Jij*M  #imposing no connection for ij

params = Jij, dt, T, N, s, tau
def sigmoid(x):
    return 1/(1+np.exp(-x))
def rateRNN(params):
    Jij, dt, T, N, s, tau = params
    lt = int(T/dt)
    II = np.random.randn(N,lt)*.1
    xs = np.zeros((N,lt))
    for tt in range(lt-1):
        xs[:,tt+1] = xs[:,tt] + dt*1/tau*(-xs[:,tt] + Jij @ sigmoid(xs[:,tt]) + II[:,tt])
    return xs, II

xs, II = rateRNN(params)
plt.figure()
plt.imshow(xs,aspect='auto')
N, lt = xs.shape
plt.figure()
plt.plot(xs.T)

# %% Model-based decoder
class MyRNN(torch.nn.Module):
    def __init__(self, N, lt, M):  #response and input
        super().__init__()
        self.N, self.lt = N, lt
        self.M = torch.Tensor(M)
        Jij = torch.zeros(N, N)
        self.Jij = torch.nn.Parameter(Jij)  #connectivity
        #######################################################################
        self.Jij = torch.nn.Parameter(self.Jij * self.M)
#        self.wfx.weight = torch.nn.parameter.Parameter((self.wfx.weight.data * self.mask_use))
        
    def forward(self, II):
        x_ = torch.zeros(self.N,self.lt)
        II = torch.Tensor(II)
        for tt in range(self.lt-1):
            ### network rate dynamics
            x_[:,tt+1] = x_[:,tt] + dt*1/tau*(-x_[:,tt] + self.Jij @ self.sigmoid(x_[:,tt]) + II[:,tt])
        return x_
    
    @staticmethod
    def sigmoid(z):
        with torch.no_grad():
            t = 1/(1+np.exp(-z))
        return t
    
class BasicModel(torch.nn.Module):
    def __init__(self, N, lt, M):
        super().__init__()
#        self.xs = torch.Tensor(xs)  #time series
#        self.II = torch.Tensor(II)  #input
        self.rnn = MyRNN(N, lt, M)
    
    def forward(self, II):
        x_ = self.rnn(II)
#        reconstruction_loss = torch.nn.cross_entropy_loss(
#            x_, 
#            self.xs, 
#            reduction='sum'
#        ) 
        return x_
    
    def generate(self,II):
        with torch.no_grad():
            return self.rnn(II)
        
# %%
model = BasicModel(N,lt, M)
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
for _ in range(10):
    xs, II = rateRNN(params)
    x_ = model( II )
    # get loss for the predicted output
    loss = criterion(x_, torch.Tensor(xs))
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
# %%
plt.figure()
plt.plot(x_.detach().numpy().T)

for param in model.parameters():
    pp = param.view(-1)
J_ = pp.detach().numpy()
J_ = J_.reshape(N,N)
plt.figure()
plt.plot(Jij,J_,'.')

        
        
# %%
###############################################################################
###############################################################################
# %%
class VAE(torch.nn.Module):
    
    def __init__(self, input_size, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, latent_size*2),
            torch.nn.Linear(latent_size*2, latent_size*2),

            torch.nn.ReLU(),
        )
        self.encoder_mu = torch.nn.Linear(latent_size*2, latent_size)
        self.encoder_log_var = torch.nn.Linear(latent_size*2, latent_size)
        self.decoder = MyRNN(latent_size, input_size)
#        torch.nn.Sequential(
#            torch.nn.Linear(latent_size, latent_size*2),
#            torch.nn.ReLU(),
#            torch.nn.Linear(latent_size*2, input_size),
#            torch.nn.Sigmoid(),
#        )

    def forward(self, input):
        activations = self.encoder(input)
        mu = self.encoder_mu(activations)
        log_var = self.encoder_log_var(activations)
        std = torch.exp(0.5 * log_var)
        latent_state = mu + torch.randn_like(std) * std
        reconstruction = self.decoder(latent_state)
        reconstruction_loss = binary_cross_entropy(
            reconstruction, 
            input, 
            reduction='sum'
        )    
        kld_loss = -torch.sum(1 + log_var - mu**2 - torch.exp(log_var))/2
        return reconstruction_loss + kld_loss

    def generate(self, n):
        with torch.no_grad():
            sampled_latent_state = torch.randn(n, self.latent_size)
            return self.decoder(sampled_latent_state)


# %%
dataset = MNIST('.', train=True, transform=ToTensor(), download=True)
loader = DataLoader(dataset=dataset, batch_size=128, drop_last=True)
vae = VAE(input_size=784, latent_size=400).to('cuda')
optimizer = torch.optim.Adam(params=vae.parameters(), lr=1e-3)
for _ in range(100):
    for batch, _ in loader:
        loss = vae(batch.view(128, 784).to('cuda'))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

generated = vae.generate(n=25).cpu()
grid = make_grid(generated.view(25, 1, 28, 28), 5, 5).numpy()
plt.imshow(np.transpose(grid, (1, 2, 0)))


# %%
###############################################################################
###############################################################################
# %%
class generative(object):
    def __init__(self,s,e,W_ss,sigs,W_pp,Sigp,W_sl,W_se):
        self.W_ss = W_ss  #recurrent
        self.sigs = sigs
        self.W_pp = W_pp  #latent
        self.Sigp = Sigp
        self.W_sl = W_sl  #latent to network
        self.W_se = W_se  #stimulus to network
        self.T = len(s)
        self.e = torch.tensor(e)
        self.s = torch.tensor(s)
        self.l = torch.zeros(len(s))
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def RNN(self):
        u = np.zeros_like(self.s)
        for tt in range(1,self.T):
            self.l[tt] = self.W_ll @ self.l[tt-1]
            self.s[tt] = self.sigmoid(u[tt])
            u[:,tt] = self.W_se @ self.e[tt-1] +self.W_ss @ self.ss[tt] + self.W_sl @ self.l[tt]
        
      