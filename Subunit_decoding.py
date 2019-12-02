# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:38:30 2019

@author: kevin
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.svm import SVC
#import Adaptive_Subunits

sns.set_style("white")
sns.set_context("talk")

# %% initialize model structure and parameters
###These should not be changed/re-run when evaluating across stimuli
#time
dt = 0.1  #in units of 10 ms for speed
T = 2000
time = np.arange(0,T,dt)
#time scales
tau_c = 2  #fast time scael in 10 ms
tau_s = 5000  #slow time scale
gamma = 0.05  #overlapping of subunit receptive field
p0 = 100  #input synaptic strength
q = 0.5  #probability to connect subunits and neuron
#q0 = 1  #strength of subunit-neuron synapse
#connectivity (image to subunits)
K = 50  #number of neurons
m_avg = 2  #number of subunits per neurons
M = int(K*m_avg)  #total number of subunits
Q = np.random.rand(K,M)  #subunit-neuron conection
Q[Q>q] = 1
Q[Q<=q] = 0
for kk in range(K):  #check for input unity
    sumsyn = sum(Q[kk,:])
    if sumsyn==0:
        Q[kk,:] = 1/M
    else:
        Q[kk,:] = Q[kk,:]/sumsyn      
#adding inhibitory connnections
pi = 0.9
signM = np.random.rand(Q.shape[0],Q.shape[1])
signM[signM>pi] = 1
signM[signM<=pi] = -1
Q = Q*signM
m = 5  #subunits per stimulus pool
N = int(M/m)  #possible unique images
P = np.ones((N,M))*gamma  #image-subunit connection
temp = 0
for nn in range(N):
    #P[nn,temp:temp+m] = 1
    P[nn,np.random.randint(0,M,(m))] = p0
    temp = temp + m
    
# %% dynamics
def stimuli():
    """
    Making marks for repeating sequences
    """
    #marks
    fnum = 4  #number of unique frames in a sequence
    dur = int(20/dt)  #duration of each frame in ms
    L = 20  #repeating the sequence
    mark = np.random.choice(N, fnum, replace=False)  #np.arange(0,fnum,1)
    seq_trans = mark[-1]
    mark2 = np.repeat(mark,dur,axis=0)
    marks = np.matlib.repmat(np.expand_dims(mark2,axis=1),L,1).reshape(-1)
    ###
    if len(marks)>len(time):
        marks = marks[:len(time)]
    else:
        marks = np.concatenate((marks,-np.ones(len(time)-len(marks))))
    seq_mark = np.zeros_like(marks)
    seq_mark[np.where(marks==seq_trans)[0]] = 1
    return marks, seq_mark  #marks for each frame and seq_mark for each sequence

def subunit_model(marks):
    """
    Input parameters for the subunit model and return time trace
    with a random set of stumulus set
    """
    #dynamics
    xs = np.zeros((M,len(time)))  #subunit activity
    alphas = np.zeros((M,len(time)))  #subunit adaptation
    ys = np.zeros((K,len(time)))  #neurons
    xs[:,0] = 0.1
    alphas[:,0] = .1
    ys[:,0] = 0.1
    ### w/ inhibitory feedforward
    #kk = np.random.randn(subu)
    for tt in range(0,len(time)-1):
        I_index = int(marks[tt])  #stimulus index
        if I_index<0:
            xs[:,tt+1] = xs[:,tt] + dt*(1/tau_c)*(-xs[:,tt] + alphas[:,tt]*0)  #subunit
            alphas[:,tt+1] = alphas[:,tt] + dt*(1/tau_s)*((1-alphas[:,tt]) - alphas[:,tt]*0)  #adaptation
        else:
            xs[:,tt+1] = xs[:,tt] + dt*(1/tau_c)*(-xs[:,tt] + alphas[:,tt]*P[I_index,:])  #subunit
            alphas[:,tt+1] = alphas[:,tt] + dt*(1/tau_s)*((1-alphas[:,tt]) - alphas[:,tt]*P[I_index,:])  #adaptation
        ys[:,tt+1] = ys[:,tt] + dt*(1/tau_c)*(-ys[:,tt] + np.matmul(Q,xs[:,tt]))  #neurons
    return ys

# %% analysis
def Measure_rij(rep):
    """
    Compute neurons x sequences response profile for decoding
    """
    #max_frameID = np.max(np.unique(marks))-2
    marks, seq_mark = stimuli()
    transitions = np.where(np.diff(seq_mark)>0)[0] #np.where(np.diff(marks)<0)[0]  #should be otherwize
    trans = transitions[:2]  #transient sequence
    sust = transitions[-2:]  #sustain sequence
    yss = []
    for jj in range(0,rep):
        marks, seq_mark = stimuli()
        ys = subunit_model(marks)
        yss.append(np.array([ys[:,trans[0]:trans[1]],ys[:,sust[0]:sust[1]]]))
    return yss  #2 x K x T (0 for transient and 1 for sustain measurements)
    

def MLE_decording(r_ij):
    """
    Neural response r from neuron i under sequence j and estimates the 
    optimal linear weight and returns predicted j'
    """
    #n_neurons = r_ij.shape[0]
    n_sequences = r_ij.shape[1]
    ###following the method in preprint for un-optimized decoding weight
    w_ = r_ij - np.repeat(np.mean(r_ij,axis=1),n_sequences).reshape(r_ij.shape)
    j_est = np.argmax(w_.T @ r_ij, axis=0)
    return j_est

# %% experimental trials
trials = 30
yss = Measure_rij(trials)
trials_t = []
trials_s = []
for k in range(0,trials):
    trials_t.append(np.mean(yss[k][0],axis=1))
    trials_s.append(np.mean(yss[k][1],axis=1))
trials_t = np.array(trials_t)
trials_s = np.array(trials_s)

# %% clustering sequential response
kmeans_t = KMeans(n_clusters=trials).fit(trials_t)

# %% SVM classification
clf = SVC(gamma='auto')
clf.fit(trials_t, np.arange(0,trials)) 
print(clf.score(trials_s, np.arange(0,trials)) )
clf.fit(trials_s, np.arange(0,trials)) 
print(clf.score(trials_t, np.arange(0,trials)) )

# %% scaling of population coding
clf = SVC(gamma='auto')
yy = np.arange(0,trials)
performance_t = []
performance_s = []
for kk in range(2,K,5):
    temp_rt = trials_t[:,:kk]
    temp_rs = trials_s[:,:kk]
    clf.fit(temp_rt, np.arange(0,trials)) 
    performance_t.append(clf.score(temp_rs,yy))
    clf.fit(temp_rs, np.arange(0,trials)) 
    performance_s.append(clf.score(temp_rt,yy))
 
plt.title('population decoding')
plt.plot(performance_t,'-o',label='transient')
plt.plot(performance_s,'-o',label='sustain')
plt.xlabel('number of cells')
plt.ylabel('performance')
plt.legend()