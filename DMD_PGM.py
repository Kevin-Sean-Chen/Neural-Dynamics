# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 19:38:16 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from dotmap import DotMap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %% Neural dynamics
###############################################################################
def Sigmoid(v,beta,Vth):
    """
    Sigmoid nonlinearity for synaptic activation
    """
    return 1/(1+np.exp(-beta*(v-Vth)))

def Gap_current(Gg,v):
    """
    Given gap junction network and voltage, return the current
    """
    Ig = np.array([ Gg[vi,:] * (vv-v) for vi,vv in enumerate(v) ]).sum(1)
    return Ig

def Syn_current(Gs,v,s):
    """
    Given the synaptic network, voltage, and synaptic strength, return the current
    """
    Is = np.array([ Gs[vi,:] * ((vv-v) * s) for vi,vv in enumerate(v) ]).sum(1)
    return Is

#setup
N = 10
T = 300
dt = 0.1
time = np.arange(0,T,dt)
tl = len(time)
Vt = np.zeros((N,tl))  #voltage time series
St = np.zeros((N,tl))  #synaptic time series

#biophysical parameters
noise = 0.1
C = 1
Gc = 1
Ecell = -35
beta = 0.125
Vth = -30
tau_r = 1
tau_d = 5

#network connectivity
psg = 0.5  #probability of connection (the larger the denser connectivity)
pss = 0.6
Gg = np.random.rand(N,N)*.1  #gap connection
Gg[np.random.rand(N,N)>psg] = 0  #sparse
#Gg = 0.5*(Gg+Gg.T)
pos = np.triu_indices(N, k=0)
Gg[pos] = Gg.T[pos]  #symmetric
Gs = np.random.randn(N,N)*3  #synaptic connection
#follow Dale's rule~

Gs[np.random.rand(N,N)>pss] = 0  #sparse

#Stimulli
Iext = np.zeros((N,tl))
Iext[5,1000:1500] = np.ones(500)*100
#Iext = np.random.randn(N,tl)*10.

#neural dynamics
def NeuralNetwork(Iext):
    """
    Given all parameters assigned outside of the function, we take a specific input pattern and simulate neural response
    Return NxT matrix of voltage time series and synaptic dynamics
    """
    Vt[:,0] = Ecell + np.random.randn(N)
    St[:,0] = np.random.randn(N)
    for tt in range(tl-1):
        Ig = Gap_current(Gg,Vt[:,tt])
        Is = Syn_current(Gs,Vt[:,tt],St[:,tt])
        Vt[:,tt+1] = Vt[:,tt] + dt*(-Gc*(Vt[:,tt]-Ecell) - Ig - Is + Iext[:,tt]) + np.sqrt(dt)*np.random.randn(N)*noise
        St[:,tt+1] = St[:,tt] + dt*(tau_r*Sigmoid(Vt[:,tt],beta,Vth)*(1-St[:,tt]) - tau_d*St[:,tt])
    return Vt, St

Vt, St = NeuralNetwork(Iext)
plt.figure()
plt.imshow(Vt,aspect='auto',extent=[0,max(time),0,N])
plt.xlabel('time')
plt.ylabel('neuron ID')
plt.figure()
plt.plot(time,Vt.T)
plt.xlabel('time')
plt.ylabel('voltage')

# %% Dynamic mode decomposition and mode dependencies
###############################################################################
def DMD_depend(X,ii):
    """
    Given response pattern X (NxT) and the stimulated neuron ii, return the mode probability computed from DMD
    """
    u,s,v = np.linalg.svd(X,full_matrices=False)  #SVD of the snapshot
    mode = np.array([np.abs(u[:,si])*ss/np.sum(s)for si,ss in enumerate(s)]).sum(0)  #weights sum of the modes
    prob = mode/mode[ii]  #normalized by the stimulated neuron
    return prob

#def Build_depend():
#building the dependency matrix by probing across neurons
P = np.zeros((N,N))  #pair-wise dependency matrix
for nn in range(N):
    Iext[nn,1000:1500] = np.ones(500)*10  #an arbitary probing stimuli
    Vt,St = NeuralNetwork(Iext)
    X = Vt[:,1000:]  #take snapshot after stimuli
    prob = DMD_depend(X,nn)
    P[nn,:] = prob
#    return P
#P = Build_depend()
plt.figure()
plt.imshow(P,aspect='auto',extent=[0,N,0,N])
plt.xlabel('neuron ID')
plt.ylabel('neuron ID')

# %% Probability graph model
###############################################################################
from anytree import Node, RenderTree
from anytree.dotexport import RenderTreeGraph
from graphviz import Digraph
# graphviz needs to be installed for the next line!
import operator
# %%
#cpt = P.copy()  #functional
cpt = np.abs(Gg)/np.sum(np.abs(Gg),axis=1)  #structural
neurons_name = np.arange(0,N)
def extend_tree(new_parent,pindex,plist,count):  
    if count>1: #line 4
        count = 0
        return
    if new_parent==None:  #line 7
        return
    elif pindex not in plist:  #line 9
        plist.append(pindex)
        child_list = {}
        count = count+1
        for i in range(0,N):  #line 12
            prob = cpt[pindex,i]
            if prob>=thresh and i not in plist:
                child_list[i] = prob  #line 14
#            if pindex in inter:
#                if (prob>=0.1 and i not in plist and i in motor):
#                    child_list[i]=prob

        if child_list==None:  #additional check?
            return
        
        else:
            sorted_child = sorted(child_list.items(), key=operator.itemgetter(1))
            num = len(sorted_child)
            for j in range(0,min(num,num)):
                c = sorted_child[num-j-1][0]
                new_child = Node(neurons_name[c], parent=new_parent)
                extend_tree(new_child, c, plist, count)
# %%           
## example: PLML as the input node
name = '_structural'
fname = 'PLML'+name
count=0
active=[]
Input = Node("Input")
thresh = 0.01

PLML = Node("5",parent=Input)
extend_tree(PLML, 5, active, count)


for pre, fill, node in RenderTree(Input):
    print("%s%s" % (pre, node.name))

RenderTreeGraph(Input).to_dotfile(fname)

# %% save image
from graphviz import render
render('dot', 'png', fname)

