r"""
MFNets: Multi-fidelity networks
===============================
This tutorial describes how to implement and deploy multi-fidelity networks to construct a surrogate of the output of a high-fidelity model using a set of lower-fidelity models of lower accuracy and cost [GJGEIJUQ2020]_.

In the :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_index_collocation.py` tutorial we showed how multi-index collocation takes adavantage of a specific type of relationship between models to build a surrogate. In some applications this structure may not exist and so methods that can exlpoit other types of structures are needed. MFNets provide a means to encode problem specific relationships between information sources.

In the following we will approximate each information source with a linear subspace model. Specifically given a basis (features) :math:`\phi_\alpha=\{\phi_{\alpha p}\}_{p=1}^P` for each information source :math:`\alpha=1\ldots,M` we seek approximations of the form 

.. math:: Y_\alpha = h(Z_\alpha)\theta_\alpha = \sum_{p=1}^{P_\alpha} \phi_p(Z_\alpha)\theta_{\alpha p}

Given data for each model :math:`y_\alpha=[(y_\alpha^{(1)})^T,(y_\alpha^{(N_\alpha)})^T]^T` where :math:`y_\alpha^{(i)}=h(\rv_\alpha^{(i)})\theta_\alpha+\epsilon_\alpha^{(i)}\in\reals^{1\times Q}`

MFNets provides a framework to encode correlation between information sources to with the goal of producing high-fidelity approximations which are more accurate than single-fidelity approximations using ony high-fidelity data. The first part of this tutorial will show how to apply MFNets to a simple problem using standard results for the posterior of Gaussian linear models. The second part of this tutorial will show how to generalize the MFNets procedure using Bayesian Networks.

Linear-Gaussian models
----------------------
As our first example we will consider the following ensemble of three univariate information sources which are parameterized by the same variable :math:`\rv_1`

.. math::

   f_0(\rv) &= \cos\left(3\pi\rv_1+0.1\right), \\
   f_1(\rv) &= \exp\left(-0.5(x-0.5)^2\right),\\
   f_2(\rv) &= f_1(\rv)+\cos(3\pi\rv_1)

Let's first import the necessary functions and modules and set the seed for reproducibility
"""
import numpy as np
from scipy import stats
import scipy
import pyapprox as pya
from pyapprox.gaussian_network import *
import matplotlib.pyplot as plt
import copy
from pyapprox.bayesian_inference.laplace import \
    laplace_posterior_approximation_for_linear_models
np.random.seed(2)
from pyapprox.configure_plots import *

#%%
#Now define the information sources and their inputs
nmodels = 3
f1 = lambda x: np.cos(3*np.pi*x+0.1).T
f2 = lambda x: np.exp(-(x-.5)**2/0.5).T
f3 = lambda x: f2(x)+np.cos(3*np.pi*x).T
functions = [f1,f2,f3]
ensemble_univariate_variables = [[stats.uniform(0,1)]]*nmodels

#%%
#Now setup the polynomial approximations of each information source
degrees=[5]*nmodels
polys,nparams = get_total_degree_polynomials(ensemble_univariate_variables,degrees)

#%%
#Generate the training data. Here we will set the noise to be independent Gaussian with mean zero and variance :math:`0.01^2`.
nsamples = [20,20,3]
samples_train = [pya.generate_independent_random_samples(p.var_trans.variable,n)
           for p,n in zip(polys,nsamples)]
noise_std=[0.01]*nmodels
noise = [noise_std[ii]*np.random.normal(
    0,noise_std[ii],(samples_train[ii].shape[1],1)) for ii in range(nmodels)]
values_train = [f(s)+n for s,f,n in zip(samples_train,functions,noise)]

#%%
#In the following we will assume a Gaussian prior on the coefficients of each approximation. Because the noise is also Gaussian and we are using linear subspace models the posterior of the approximation coefficients will also be Gaussian.
#
#First let's setup our linear model involving all information sources
#
#.. math::
#
#   y=\Phi\theta+\epsilon 
#
#Specifically let :math:`\Phi_\alpha\in\reals^{N_i\times P_i}` be Vandermonde-like matrices with entries :math:`\phi_{\alpha p}(\rv_\alpha^{(n)})` in the nth row and pth column. Now define :math:`\Phi=\mathrm{blockdiag}\left(\Phi_1,\ldots,\Phi_M\right)` and :math:`\Sigma_\epsilon=\mathrm{blockdiag}\left(\Sigma_{\epsilon 1},\ldots,\Sigma_{\epsilon M}\right)`, :math:`y=\left[y_1^T,\ldots,y_M^T\right]^T`,  :math:`\theta=\left[\theta_1^T,\ldots,\theta_M^T\right]^T`, and :math:`\epsilon=\left[\epsilon_1^T,\ldots,\epsilon_M^T\right]^T`.
#
#For our three model example we have
#
#.. math::
#
#   \Phi=\begin{bmatrix}\Phi_1 & 0_{M_1P_2} & 0_{M_1P_3}\\ 0_{M_2P_1} & \Phi_2 & 0_{M_2P_3}\\0_{M_3P_1} & 0_{M_3P_1} & \Phi_3 \end{bmatrix} \qquad \Sigma_\epsilon=\begin{bmatrix}\Sigma_{\epsilon 1} & 0_{M_1M_2} & 0_{M_1M_3}\\ 0_{M_2M_1} & \Sigma_{\epsilon 2} & 0_{M_2M_3}\\0_{M_3M_1} & 0_{M_3M_1} & \Sigma_{\epsilon 3} \end{bmatrix}\qquad y= \begin{bmatrix}y_1\\y_2\\y_3 \end{bmatrix}
#
#where the :math:`0_{mp}\in\reals^{m\times p}` is a matrix of zeros.

basis_matrices = [p.basis_matrix(s) for p,s in zip(polys,samples_train)]
basis_mat = scipy.linalg.block_diag(*basis_matrices)
noise_matrices = [noise_std[ii]**2*np.eye(samples_train[ii].shape[1])
                  for ii in range(nmodels)]
noise_cov = scipy.linalg.block_diag(*noise_matrices)
values = np.vstack(values_train)

#%%
#Now let the prior on the coefficients of :math:`Y_\alpha` be Gaussian with mean :math:`\mu_\alpha` and covariance :math:`\Sigma_{\alpha\alpha}`, and the covariance between the coefficients of different information sources :math:`Y_\alpha` and :math:`Y_\beta` be :math:`\Sigma_{\alpha\beta}`, such that the joint density of the coefficients of all information sources is Gaussian with mean and covariance given by
#
# .. math::  \mu=\left[\mu_1^T,\ldots,\mu_M^T\right]^T` \qquad \Sigma=\begin{bmatrix}\Sigma_{11} &\Sigma_{12} &\ldots &\Sigma_{1M} \\ \Sigma_{21} &\Sigma_{22} &\ldots &\Sigma_{2M}\\\vdots &\vdots & \ddots &\vdots \\ \Sigma_{M1} &\Sigma_{M2} &\ldots &\Sigma_{MM}\end{bmatrix}
#
#In the following we will set the prior mean to zero for all coefficients and first try setting all the coefficients to be independent
prior_mean = np.zeros((nparams.sum(),1))
prior_cov = np.eye(nparams.sum())

#%%
#With these definition the posterior distribution of the coefficients is (see :ref:`sphx_glr_auto_tutorials_foundations_plot_bayesian_inference.py`)
#
#.. math:: \Sigma^\mathrm{post}=\left(\Sigma^{-1}+\Phi^T\Sigma_\epsilon^{-1}\Phi\right)^{-1}, \qquad  \mu^\mathrm{post}=\Sigma^\mathrm{post}\left(\Phi^T\Sigma_\epsilon^{-1}y+\Sigma^{-1}\mu\right),
#
#We can find these using
post_mean,post_cov = laplace_posterior_approximation_for_linear_models(
    basis_mat,prior_mean,np.linalg.inv(prior_cov),np.linalg.inv(noise_cov),
    values)

#%%
#Now let's plot the resulting approximation of the high-fidelity data.
hf_prior=(prior_mean[nparams[:-1].sum():],
          prior_cov[nparams[:-1].sum():,nparams[:-1].sum():])
hf_posterior=(post_mean[nparams[:-1].sum():],
              post_cov[nparams[:-1].sum():,nparams[:-1].sum():])
xx=np.linspace(0,1,101)
fig,axs=plt.subplots(1,1,figsize=(8,6))
training_labels = [r'$f_1(z_1^{(i)})$',r'$f_2(z_2^{(i)})$',r'$f_2(z_2^{(i)})$']
plot_1d_lvn_approx(xx,nmodels,polys[2].basis_matrix,hf_posterior,hf_prior,
                   axs,samples_train,values_train,training_labels,[0,1])
axs.set_xlabel(r'$z$')
axs.set_ylabel(r'$f(z)$')
plt.plot(xx,f3(xx),'k',label=r'$f_3$')
plt.show()

#%%
#Unfortunately by assuming that the coefficients of each information source are independent the lower fidelity data is not informing the estimation of the coefficients of the high-fidelity approximation. This statement can be verified by computing an approximation with only the high-fidelity data

single_hf_posterior = laplace_posterior_approximation_for_linear_models(
    basis_matrices[-1],hf_prior[0],np.linalg.inv(hf_prior[1]),
    np.linalg.inv(noise_matrices[-1]),values_train[-1])
assert np.allclose(single_hf_posterior[0],hf_posterior[0])
assert np.allclose(single_hf_posterior[1],hf_posterior[1])

#%%
#We can improve the high-fidelity approximation by encoding a correlation between the coefficients of each information source. In the following we will assume that the cofficients of an information source is linearly related to the coefficients of the other information sources. Specifically we will assume that
#
#.. math:: \theta_\alpha = \sum_{\beta\in\mathrm{pa}(\alpha)} A_{\alpha\beta}\theta_\beta + b_\alpha + v_\alpha,
#
#where :math:`b_\alpha\in\reals^{P_\alpha}` is a deterministic shift, :math:`v_\alpha` is a Gaussian noise with mean zero and covariance :math:`\Sigma_{v_\alpha}\in\reals^{P_\alpha\times P_\alpha}`, and  :math:`\mathrm{pa}(\alpha)\subset \{\beta : \beta=1,\ldots,M, \beta\neq\alpha\}` is a possibly empty subset of indices indexing the information sources upon which the :math:`\alpha` information source is dependent. Here :math:`A_{\alpha\beta}\in\reals^{P_\alpha\times P_\beta}` are matrices which, along with :math:`\Sigma_{v\alpha}`, define the strength of the relationship between the coefficients of each information source. When these matrices are dense each coefficient of :math:`Y_\alpha` is a function of all coefficients in :math:`Y_\beta`. It is often more appropriate, however to impose a sparse structure. For example if :math:`A` is diagonal this implies that the coefficient of a certain basis in the representation of one information source is only related to the coefficient of the same basis in the other information sources.
#
#Note this notation comes from the literature on Bayesian networks which we will use to generalize the procedure described in this tutorial to large ensembles of information sources with complex dependencies.
#
#The variable :math:`v_\alpha` is a random variable that controls the correlation between the coefficients of the information sources. The MFNets framework in [GJGEIJUQ2020]_ assumes that this variable is Gaussian with mean zero and covariance given by :math:`\Sigma_{v\alpha}`. In this example we will set
#
#.. math:: \theta_3=A_{31}\theta_1+A_{32}\theta_2+b_3+v_3
#
#and assume that :math:`\covar{\theta_\alpha}{v_3}=0\: \forall \alpha` and  :math:`\covar{\theta_1}{\theta_2}=0`. Note the later relationship does not mean data from the information from :math:`Y_1` cannot be used to inform the coefficients of :math:`Y_2`.
#
#Given the defined relationship between the coefficients of each information source we can compute the prior over the joint distribiution of the coefficients of all information sources. Without loss of generality we assume the variables have zero mean and:math:`b_3=0` so that
#
#.. math:: \covar{\theta_1}{\theta_3}=\mean{\theta_1\theta_3^T}=\mean{\theta_1\left(A_{31}\theta_1+A_{32}\theta_2+v_\alpha\right)^T}=\covar{\theta_1}{\theta_1}A_{31}^T
#
#similarly :math:`\covar{\theta_2}{\theta_3}=\covar{\theta_2}{\theta_2}A_{32}^T`. We also have
#
#.. math:: \covar{\theta_3}{\theta_3}&=\mean{\theta_1\theta_1^T}=\mean{\left(A_{13}\theta_1+A_{12}\theta_2+v_\alpha\right)\left(A_{31}\theta_1+A_{32}\theta_2+v_3\right)^T}\\&=A_{31}\covar{\theta_1}{\theta_1}A_{31}^T+A_{32}\covar{\theta_2}{\theta_2}A_{32}^T+\covar{v_3}{v_3}
#
#In this tutorial we will set :math:`A_{31}=a_{31} I`, :math:`A_{32}=a_{32} I`, :math:`\Sigma_{11}=s_{11} I`, :math:`\Sigma_{22}=s_{22} I` and  :math:`\Sigma_{v3}=v_{3} I` to be diagonal matrices with the same value for all entries on the diagonal which gives
#
#.. math:: \Sigma=\begin{bmatrix}\Sigma_{11} & 0 & a_{31}\Sigma_{11}\\ 0 & \Sigma_{22} & a_{32}\Sigma_{22}\\ a_{31}\Sigma_{11} & a_{32}\Sigma_{22} & a_{31}^2\Sigma_{11}+a_{32}^2\Sigma_{22}+\Sigma_{v3}\end{bmatrix}
#
#In the following we want to set the prior covariance of each individual information source to be the same, i.e. we set :math:`s_{11}=s_{22}` and :math:`v_3=s_{33}-(a_{31}^2\Sigma_{11}+a_{32}^2\Sigma_{22})`
    
I1,I2,I3 = np.eye(degrees[0]+1),np.eye(degrees[1]+1),np.eye(degrees[2]+1)

s11,s22,s33=[1]*nmodels
a31,a32=[0.7]*(nmodels-1)
#if a31==32 and s11=s22=s33=1 then a31<=1/np.sqrt(2))
assert (s33-a31**2*s11-a32**2*s22)>0
rows = [np.hstack([s11*I1,0*I1,a31*s11*I1]),np.hstack([0*I2,s22*I2,a32*s22*I2]),
        np.hstack([a31*s11*I3,a32*s22*I3,s33*I3])]
prior_cov=np.vstack(rows)

#%%
# Plot the structure of the prior covariance
fig, axs = plt.subplots(1,1,figsize=(8,6))
plt.spy(prior_cov,cmap='coolwarm')
plt.show()

#%% Now lets compute the posterior distribution and plot the resulting approximations
post_mean,post_cov = laplace_posterior_approximation_for_linear_models(
    basis_mat,prior_mean,np.linalg.inv(prior_cov),np.linalg.inv(noise_cov),
    values)

#%%
#Now let's plot the resulting approximation of the high-fidelity data.
hf_prior=(prior_mean[nparams[:-1].sum():],
         prior_cov[nparams[:-1].sum():,nparams[:-1].sum():])
hf_posterior=(post_mean[nparams[:-1].sum():],
             post_cov[nparams[:-1].sum():,nparams[:-1].sum():])
xx=np.linspace(0,1,101)
fig,axs=plt.subplots(1,1,figsize=(8,6))
training_labels = [r'$f_1(z_1^{(i)})$',r'$f_2(z_2^{(i)})$',r'$f_2(z_2^{(i)})$']
plot_1d_lvn_approx(xx,nmodels,polys[2].basis_matrix,hf_posterior,hf_prior,
                   axs,samples_train,values_train,training_labels,[0,1])
axs.set_xlabel(r'$z$')
axs.set_ylabel(r'$f(z)$')
plt.plot(xx,f3(xx),'k',label=r'$f_3$')
plt.show()

#%%
#Depsite using only a very small number of samples of the high-fidelity information source, the multi-fidelity approximation has smaller variance and the mean more closely approximates the true high-fidelity information source, when compared to the single fidelity strategy.
#

# #For this network we have :math:`\mathrm{pa}(1)=\emptyset,\;\mathrm{pa}(1)=\emptyset,\;\mathrm{pa}(2)=\{1,2\}` and the graph has one CPDs which for this example is given by
# #
# #.. math:: \mathbb{P}(\theta_3\mid \theta_1,\theta_2) \sim \mathcal{N}\left(A_{31}\theta_1+A_{32}\theta_2+b_3,\Sigma_{v3}\right),
# #
# #with :math:`b_3=0`.
# #
# #We refer to a Gaussian network based upon this DAG as a peer network. Consider the case where a high fidelity simulation model incorporates two sets of active physics, and the two low-fidelity peer models each contain one of these components. If the high-fidelity model is given, then the low-fidelity models are no longer independent of one another. In other words, information about the parameters used in one set of physics will inform the other set of physics because they are coupled together in a known way through the high-fidelity model.
# #
# #The joint density of the network is given by
# #
# #.. math:: \mathbb{P}(\theta_1,\theta_2,\theta_3)=\mathbb{P}(\theta_3\mid \theta_1,\theta_2)\mathbb{P}(\theta_1)\mathbb{P}(\theta_2)

# #convert_to_compact_factors must be after add_data when doing inference
# network.convert_to_compact_factors()
# evidence, evidence_ids = network.assemble_evidence(values_train)
# factor_post = cond_prob_variable_elimination(
#     network,labels,evidence_ids=evidence_ids,evidence=evidence)    
# #post_mean,post_cov = convert_gaussian_from_canonical_form(
# #    factor_post.precision_matrix,factor_post.shift)
# print(prior_cov)
# print(post_cov,'post_cov')

# #assert False


# hf_prior=(prior_mean[nparams[:-1].sum():],
#           prior_cov[nparams[:-1].sum():,nparams[:-1].sum():])
# hf_posterior=(post_mean[nparams[:-1].sum():],
#               post_cov[nparams[:-1].sum():,nparams[:-1].sum():])
# xx=np.linspace(0,1,101)
# fig,axs=plt.subplots(1,1,figsize=(8,6))
# plot_1d_lvn_approx(xx,nmodels,polys[2].basis_matrix,hf_posterior,hf_prior,
#                    axs,samples_train,values_train,None,[0,1])
# axs.plot(xx,functions[2](xx[np.newaxis,:]),'r',label=r'$f_3$')
# plt.show()

# polys = set_polynomial_ensemble_coef_from_flattened(polys,post_mean)
# #assert False

# xx=np.linspace(0,1,1101)
# plt.plot(xx,polys[2](xx[np.newaxis,:]),'k--',label=r'$\mathrm{MFNet}$')
# plt.plot(xx,functions[2](xx[np.newaxis,:]),'r',label=r'$f_3$')
# plt.plot(samples_train[2][0,:],values_train[2][:,0],'o')
# plt.plot(xx,sf_poly(xx[np.newaxis,:]),'b:',label=r'$\mathrm{SF}$')
# plt.xlabel(r'$z$')
# plt.ylabel(r'$f(z)$')
# plt.legend()
# plt.show()

# #Before computing the multi-fidelity approximation, let's first contruct an approximation using only the high fidelity data.

# #
# #In the following we will use use Gaussian networks to fuse information from a modification of the information enembles used in the previous section. Specifically consider the enemble
# #
# #.. math::
# #
# #   f_0(\rv) &= \cos\left(3\pi\rv_1+0.1\rv_2\right), \\
# #   f_1(\rv) &= \exp\left(-0.5(x-0.5)^2\right),\\
# #   f_2(\rv) &= f_1(\rv)+\cos(3\pi\rv_1)
# #

# nmodels=3
# f1 = lambda x: np.cos(3*np.pi*x[0,:]+0.1*x[1,:])
# f2 = lambda x: np.exp(-(x-.5)**2/0.5)
# f3 = lambda x: f2(x)+np.cos(3*np.pi*x)
# functions = [f1,f2,f3]

# ensemble_univariate_variables=[[stats.uniform(0,1)]*2]+[[stats.uniform(0,1)]]*2

# #%%
# #The difference between this example and the previous is that one of the low-fidelity information sources has two inputs in contrast to the other sources (functions) which have one. These types of sources CANNOT be fused by other multi-fidelity methods. Fusion is possible with MFNets because it relates information sources through correlation between the coefficients of the approximations of each information source. In the context of Bayesian networks the coefficients are called latent variables.
# #
# #Again assume that the coefficients of one source are only related to the coefficient of the corresponding basis function in the parent sources. Note that unlike before the :math:`A_{ij}` matrices will not be diagonal. The polynomials have different numbers of terms and so the :math:`A_{ij}` matrices will be rectangular. They are essentially a diagonal matrix concatenated with a matrix of zeros. Let :math:`A^\mathrm{nz}_{31}=a_{31}I\in\reals^{P_1\times P_1}` be a diagonal matrix relating the coefficients of all the shared terms in :math:`Y_1,Y_3`. Then :math:`A^\mathrm{nz}_{31}=[A^\mathrm{nz}_{31} \: 0_{P_3\times(P_1-P_3)}]\in\reals^{P_1\times P_2}`.
# #
# #Use the following to setup a Gaussian network for our example
# #degrees = [3,5,5]
# degrees = [0,0,0]
# polys,nparams = get_total_degree_polynomials(ensemble_univariate_variables,degrees)
# basis_matrix_funcs = [p.basis_matrix for p in polys]

# s11,s22,s33=[1]*nmodels
# a31,a32=[0.7]*(nmodels-1)
# cpd_scales=[a31,a32]
# prior_covs=[s11,s22,s33]
# network = build_peer_polynomial_network(
#     prior_covs,cpd_scales,basis_matrix_funcs,nparams)

# #%%
# #We can compute the prior from this network using by instantiating the factors used to represent the joint density of the coefficients and then multiplying them together using the conditional probability variable elimination algorithm. We will describe this algorithm in more detail when infering the posterior distribution of the coefficients from data using the graph. When computing the prior this algorithm simply amounts to multiplying the factors of the graph together.
# network.convert_to_compact_factors()
# labels = [l[1] for l in network.graph.nodes.data('label')]
# factor_prior = cond_prob_variable_elimination(
#     network,labels)
# prior_mean,prior_cov = convert_gaussian_from_canonical_form(
#     factor_prior.precision_matrix,factor_prior.shift)
# print(prior_cov)

# #To infer the uncertain coefficients we must add training data to the network.
# nsamples = [10,10,2]
# samples_train = [pya.generate_independent_random_samples(p.var_trans.variable,n)
#            for p,n in zip(polys,nsamples)]
# noise_std=[0.01]*nmodels
# noise = [noise_std[ii]*np.random.normal(
#     0,noise_std[ii],(samples_train[ii].shape[1],1)) for ii in range(nmodels)]
# values_train = [f(s)+n for s,f,n in zip(samples_train,functions,noise)]
# network.add_data_to_network(samples_train,np.array(noise_std)**2)
# fig,ax = plt.subplots(1,1,figsize=(8,5))
# plot_peer_network_with_data(network.graph,ax)
# plt.show()

# #%%
# #Using this graph we can infer the posterior distribution of the information source coefficients using the conditional probability variable elimination algorithm. The algorithm begins by conditioning the each factor of the graph with any data associated with that factor. The graph above will have 3 factors involving data associated with the CPDs :math:`\mathbb{P}(\theta_1\mid y_1),\mathbb{P}(\theta_2\mid y_2),\mathbb{P}(\theta_3\mid y_3)`. 


#%%
#References
#^^^^^^^^^^
#.. [GJGEIJUQ2020] `MFNets: Multi-fidelity data-driven networks for bayesian learning and prediction, International Journal for Uncertainty Quantification, 2020. <https://www.alexgorodetsky.com/static/papers/gorodetsky_jakeman_geraci_eldred_mfnets_2020.pdf>`_
#
#.. [GJGJCP2020] `MFNets: Learning network representations for multifidelity surrogate modeling, 2020. <https://res.arxiv.org/abs/2008.02672>`_
#
#Appendix
#^^^^^^^^
#There is a strong connection between the mean of the Bayes posterior distribution of linear-Gaussian models with least squares regression. Specifically the mean of the posterior is equivalent to linear least-squares regression with a regulrization that penalizes deviations from the prior estimate of the parameters. Let the least squares objective function be
#
#.. math:: f(\theta)=\frac{1}{2}(y-A\theta)^T\Sigma_\epsilon^{-1}(y-A\theta)+\frac{1}{2}(\mu_\theta-\theta)^T\Sigma_\theta^{-1}(\mu_\theta-\theta),
#
#where the first term on the right hand side is the usual least squares objective and the second is the regularization term. This regularized objective is minimized by setting its gradient to zero, i.e.
#
#.. math::
#
#   \nabla_\theta f(\theta)=A^T\Sigma_\epsilon^{-1}(y-A\theta)+\Sigma_\theta^{-1}(\mu_\theta-\theta)=0,
#
#thus
#
#.. math::
#
#   A^T\Sigma_\epsilon^{-1}A\theta+\Sigma_\theta^{-1}\theta=A^T\Sigma_\epsilon^{-1}y+\Sigma_\theta^{-1}\mu_\theta
#
#and so
#
#.. math::
#
#   \theta=\left(A^T\Sigma_\epsilon^{-1}A\theta+\Sigma_\theta^{-1}\right)^{-1}\left(A^T\Sigma_\epsilon^{-1}y+\Sigma_\theta^{-1}\mu_\theta\right).
#
#Noting that :math:`\left(A^T\Sigma_\epsilon^{-1}A\theta+\Sigma_\theta^{-1}\right)^{-1}` is the posterior covariance we obtain the usual expression for the posterior mean
#
#.. math:: \mu^\mathrm{post}=\Sigma^\mathrm{post}\left(A^T\Sigma_\epsilon^{-1}y+\Sigma_\theta^{-1}\mu_\theta\right)



