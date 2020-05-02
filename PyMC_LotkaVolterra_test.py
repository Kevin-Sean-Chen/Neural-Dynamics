# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 00:57:13 2020

@author: kevin
"""

from scipy.integrate import odeint
import numpy as np
import theano
from theano import *
import matplotlib.pyplot as plt
import pymc3 as pm

THEANO_FLAGS='optimizer=fast_compile'

# %% ODE model
class FitzhughNagumoModel(object):
    def __init__(self, times, y0=None):
            self._y0 = np.array([-1, 1], dtype=np.float64)
            self._times = times

    def _simulate(self, parameters, times):
        a, b, c = [float(x) for x in parameters]

        def rhs(y, t, p):
#            if t>3 and t<6:
#                I = 1
#            else:
#                I = 0
            V, R = y
            dV_dt = (V - V**3 / 3 + R) * c# + I
            dR_dt = (V - a + b * R) / -c
            return dV_dt, dR_dt
        values = odeint(rhs, self._y0, times, (parameters,),rtol=1e-6,atol=1e-6)
        return values

    def simulate(self, x):
        return self._simulate(x, self._times)

# %% samples
n_states = 2
n_times = 200
true_params = [0.2,0.2,3.]
noise_sigma = 0.5
FN_solver_times = np.linspace(0, 20, n_times)
ode_model = FitzhughNagumoModel(FN_solver_times)
sim_data = ode_model.simulate(true_params)
np.random.seed(42)
Y_sim = sim_data + np.random.randn(n_times,n_states)*noise_sigma
plt.figure(figsize=(15, 7.5))
plt.plot(FN_solver_times, sim_data[:,0], color='darkblue', lw=4, label=r'$V(t)$')
plt.plot(FN_solver_times, sim_data[:,1], color='darkgreen', lw=4, label=r'$R(t)$')
plt.plot(FN_solver_times, Y_sim[:,0], 'o', color='darkblue', ms=4.5, label='Noisy traces')
plt.plot(FN_solver_times, Y_sim[:,1], 'o', color='darkgreen', ms=4.5)
plt.legend(fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Values',fontsize=15)
plt.title('Fitzhugh-Nagumo Action Potential Model', fontsize=25);

# %% Define a non-differentiable black-box op
import theano.tensor as tt
from theano.compile.ops import as_op

@as_op(itypes=[tt.dscalar,tt.dscalar,tt.dscalar], otypes=[tt.dmatrix])
def th_forward_model(param1,param2,param3):

    param = [param1,param2,param3]
    th_states = ode_model.simulate(param)

    return th_states

# %% Generative model
draws = 100
with pm.Model() as FN_model:

    a = pm.Gamma('a', alpha=2, beta=1)
    b = pm.Normal('b', mu=0, sd=1)
    c = pm.Uniform('c', lower=0.1, upper=10)

    sigma = pm.HalfNormal('sigma', sd=1)

    forward = th_forward_model(a,b,c)

    cov=np.eye(2)*sigma**2


    Y_obs = pm.MvNormal('Y_obs', mu=forward, cov=cov, observed=Y_sim)

    startsmc =  {v.name:np.random.uniform(1e-3,2, size=draws) for v in FN_model.free_RVs}

    trace_FN = pm.sample_smc(draws, start=startsmc)

# %%
# %%
pm.plot_posterior(trace_FN, kind='hist', bins=30, color='seagreen');

# %% inference summary
import pandas as pd
results=[pm.summary(trace_FN, ['a']),pm.summary(trace_FN, ['b']),pm.summary(trace_FN, ['c'])\
        ,pm.summary(trace_FN, ['sigma'])]
results=pd.concat(results)
true_params.append(noise_sigma)
results['True values'] = pd.Series(np.array(true_params), index=results.index)
true_params.pop();
results
# %% phase portrait reconstruction

params=np.array([trace_FN.get_values('a'),trace_FN.get_values('b'),trace_FN.get_values('c')]).T
params.shape
new_values = []
for ind in range(len(params)):
    ppc_sol= ode_model.simulate(params[ind])
    new_values.append(ppc_sol)
new_values = np.array(new_values)
mean_values = np.mean(new_values, axis=0)
plt.figure(figsize=(15, 7.5))

plt.plot(mean_values[:,0], mean_values[:,1], color='black', lw=4, label='Inferred (mean of sampled) phase portrait')
plt.plot(sim_data[:,0], sim_data[:,1], '--', color='#ff7f0e', lw=4, ms=6, label='True phase portrait')
plt.legend(fontsize=15)
plt.xlabel(r'$V(t)$',fontsize=15)
plt.ylabel(r'$R(t)$',fontsize=15);

