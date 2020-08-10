r"""
MFNets: Multi-fidelity networks
-------------------------------
This tutorial describes how to implement and deploy multi-fidelity networks to construct a surrogate of the output of a high-fidelity model using a set of lower-fidelity models of lower accuracy and cost.

Multi-index collocation :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_index_collocation.py` takes adavantage of a specific type of relationship between models. In many practical applications this structure may not exist.

MFNets provide a means to encode problem specific relationships between information sources.

In the following we will approximate each information source with a linear subspace model. Specifically given a basis (features) :math:`\phi_\alpha=\{\phi_{\alpha k}\}_{k=1}^K` for each information source :math:`\alpha=`1\ldots,M` we seek approximations of the form 

.. math:: Y_\alpha = h(Z_\alpha)\theta_\alpha = \sum_{k=1}^K \phi_k(Z_\alpha)\theta_{\alpha k}

Given data for each model :math:`y_\alpha=[(y_\alpha^{(1)})^T,(y_\alpha^{(N)})^T]^T` where :math:`y_\alpha^{(i)}=h(\rv_\alpha^{(i)})\theta_\alpha+\epsilon_\alpha^{(i)}\in\reals^{1\times Q}`

We can compute the coefficients of each subspace approximation using maximum likelihood estimation which equates to using least-squares regression in this setting.
"""

def set_polynomial_ensemble_coef_from_flattened(polys,coefs):
    idx1,idx2=0,0
    for ii in range(nmodels):
        idx2 += nparams[ii]
        polys[ii].set_coefficients(coefs[idx1:idx2])
        idx1=idx2
    return polys

import numpy as np
from scipy import stats
import scipy
import pyapprox as pya
from pyapprox.gaussian_network import *
import matplotlib.pyplot as plt
import copy

np.random.seed(2)

f1 = lambda x: np.cos(3*np.pi*x[0,:])[:,np.newaxis]
f2 = lambda x: np.exp(-(x-.5)**2/0.5).T
f3 = lambda x: f2(x)+np.cos(3*np.pi*x).T
functions = [f1,f2,f3]
nmodels = 3

ensemble_univariate_variables = [[stats.uniform(0,1)]]*nmodels

degrees=[3]*nmodels
polys,nparams = get_total_degree_polynomials(ensemble_univariate_variables,degrees)

nsamples = [20,20,degrees[2]+1]
samples_train = [pya.generate_independent_random_samples(p.var_trans.variable,n)
           for p,n in zip(polys,nsamples)]
noise_std=[0.1]*nmodels
noise = [noise_std[ii]*np.random.normal(
    0,noise_std[ii],(samples_train[ii].shape[1],1)) for ii in range(nmodels)]

values_train = [f(s)+n for s,f,n in zip(samples_train,functions,noise)]

basis_matrices = [p.basis_matrix(s) for p,s in zip(polys,samples_train)]
basis_mat = scipy.linalg.block_diag(*basis_matrices)
values = np.vstack(values_train)

coefs = np.linalg.lstsq(basis_mat,values,rcond=None)[0]
polys = set_polynomial_ensemble_coef_from_flattened(polys,coefs)

sf_coef = np.linalg.lstsq(basis_matrices[2],values_train[2],rcond=None)[0]
sf_poly = copy.deepcopy(polys[2])
sf_poly.set_coefficients(sf_coef)

noise_cov_inv = scipy.linalg.block_diag(
    *[1/noise_std[ii]**2*np.eye(samples_train[ii].shape[1])
      for ii in range(nmodels)])
prior_mean = np.zeros((nparams.sum(),1))

assert degrees[0]==degrees[1]==degrees[2]
I1 = np.eye(degrees[0]+1)
I2 = np.eye(degrees[1]+1)
I3 = np.eye(degrees[2]+1)

v1,v2,v3=[1]*nmodels
a31,a32=[0]*(nmodels-1)#check if using [0] gives the high-fidelity only case
a31,a32=[0.1]*(nmodels-1)
rows = [np.hstack([v1*I1,0*I1,a31*v1*I1]),np.hstack([0*I2,v2*I2,a32*v2*I2]),np.hstack([a31*v1*I3,a32*v2*I3,v3*I3])]
#print(rows)
prior_cov=np.vstack(rows)
#print(prior_cov)
#plt.spy(prior_cov,cmap='coolwarm')
#plt.show()

from pyapprox.bayesian_inference.laplace import \
    laplace_posterior_approximation_for_linear_models
post_mean,post_cov = laplace_posterior_approximation_for_linear_models(
    basis_mat,prior_mean,np.linalg.inv(prior_cov),noise_cov_inv,values)
print(np.linalg.cond(prior_cov))
print(prior_cov.min())
print(post_cov.min())

# solve using network
cpd_scales=[a31,a32]
prior_covs=[v1,v2,v3]
basis_matrix_funcs = [p.basis_matrix for p in polys]
network = build_peer_polynomial_network(
    prior_covs,cpd_scales,basis_matrix_funcs,nparams)
labels = [l[1] for l in network.graph.nodes.data('label')]
network.add_data_to_network(samples_train,np.array(noise_std)**2)
#convert_to_compact_factors must be after add_data when doing inference
network.convert_to_compact_factors()
evidence, evidence_ids = network.assemble_evidence(values_train)
factor_post = cond_prob_variable_elimination(
    network,labels,evidence_ids=evidence_ids,evidence=evidence)    
post_mean,post_cov = convert_gaussian_from_canonical_form(
    factor_post.precision_matrix,factor_post.shift)
print(post_cov.min())
#assert False


hf_prior=(prior_mean[nparams[:-1].sum():],
          prior_cov[nparams[:-1].sum():,nparams[:-1].sum():])
hf_posterior=(post_mean[nparams[:-1].sum():],
              post_cov[nparams[:-1].sum():,nparams[:-1].sum():])
xx=np.linspace(0,1,101)
fig,axs=plt.subplots(1,1,figsize=(8,6))
plot_1d_lvn_approx(xx,nmodels,polys[2].basis_matrix,hf_posterior,hf_prior,
                   axs,samples_train,values_train,None,[0,1])
axs.plot(xx,functions[2](xx[np.newaxis,:]),'r',label=r'$f_3$')
plt.show()

polys = set_polynomial_ensemble_coef_from_flattened(polys,post_mean)
#assert False

xx=np.linspace(0,1,1101)
plt.plot(xx,polys[2](xx[np.newaxis,:]),'k--',label=r'$\mathrm{MFNet}$')
plt.plot(xx,functions[2](xx[np.newaxis,:]),'r',label=r'$f_3$')
plt.plot(samples_train[2][0,:],values_train[2][:,0],'o')
plt.plot(xx,sf_poly(xx[np.newaxis,:]),'b:',label=r'$\mathrm{SF}$')
plt.xlabel(r'$z$')
plt.ylabel(r'$f(z)$')
plt.legend()
plt.show()
