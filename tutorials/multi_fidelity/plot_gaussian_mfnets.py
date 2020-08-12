r"""
MFNets: Multi-fidelity networks
===============================
This tutorial describes how to implement and deploy multi-fidelity networks to construct a surrogate of the output of a high-fidelity model using a set of lower-fidelity models of lower accuracy and cost [GJGEIJUQ2020]_, [GJGJCP2020]_.

Multi-index collocation :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_index_collocation.py` takes adavantage of a specific type of relationship between models. In many practical applications this structure may not exist.

MFNets provide a means to encode problem specific relationships between information sources.

In the following we will approximate each information source with a linear subspace model. Specifically given a basis (features) :math:`\phi_\alpha=\{\phi_{\alpha p}\}_{p=1}^P` for each information source :math:`\alpha=1\ldots,M` we seek approximations of the form 

.. math:: Y_\alpha = h(Z_\alpha)\theta_\alpha = \sum_{p=1}^{P_\alpha} \phi_p(Z_\alpha)\theta_{\alpha p}

Given data for each model :math:`y_\alpha=[(y_\alpha^{(1)})^T,(y_\alpha^{(N_\alpha)})^T]^T` where :math:`y_\alpha^{(i)}=h(\rv_\alpha^{(i)})\theta_\alpha+\epsilon_\alpha^{(i)}\in\reals^{1\times Q}`

We will show how we can use Byesian inference to use data from each model to produce an accurate approximation of the highest fidelity information source. Indeed the approach we discuss also has the advantage of using all data to inform all approximations.

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
#.. math:: \theta_3=A_{31}\theta_1+A_{32}\theta_2+v_3
#
#and assume that :math:`\covar{\theta_\alpha}{v_3}=0\: \forall \alpha` and  :math:`\covar{\theta_1}{\theta_2}=0`. Note the later relationship does not mean data from the information from :math:`Y_1` cannot be used to inform the coefficients of :math:`Y_2`.
#
#Given the defined relationship between the coefficients of each information source we can compute the prior over the joint distribiution of the coefficients of all information sources. Without loss of generality we assume the variables have zero mean so that
#
#.. math:: \covar{\theta_1}{\theta_3}=\mean{\theta_1\theta_3}=\mean{\theta_1\left(A_{13}\theta_1+A_{12}\theta_2+v_\alpha\right)}=A_{31}\covar{\theta_1}{\theta_1}
#
#similarly :math:`\covar{\theta_2}{\theta_3}=A_{31}\covar{\theta_1}{\theta_1}`. We also have
#
#.. math:: \covar{\theta_1}{\theta_1}&=\mean{\theta_1\theta_1}=\mean{\left(A_{13}\theta_1+A_{12}\theta_2+v_\alpha\right)^T\left(A_{13}\theta_1+A_{12}\theta_2+v_\alpha\right)}\\&=A_{31}^TA_{31}\covar{\theta_1}{\theta_1}+A_{32}^TA_{32}\covar{\theta_2}{\theta_2}+\covar{v_1}{v_1}
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
#Gaussian Networks
#-----------------
#The procedure described in this first part of this tutorial can be viewed as imposing conditional independence relationships between information sources. Let :math:`A,B,C` denote three random variables. :math:`A` is conditionally independent of :math:`B` given :math:`C` in distribution :math:`\mathbb{P}`, denoted by :math:`A \mathrel{\perp\mspace{-10mu}\perp} B \mid C`, if 
#
#.. math:: \mathbb{P}(A = a, B = b \mid C = c)  = \mathbb{P}(A = a\mid C=c) \mathbb{P}(B = b \mid C = c).
#
#We use Bayesian networks to provide an abstraction built on graph theory to describe the conditional independence relationships among groups of random variables. The parameter relationships used in the previous section can be represented as a Bayesian Network with Gaussian conditional independence relationships. Such a network is often referred to as a Gaussian Network.
#
#A Bayesian network (BN) structure is a directed acyclic graphs (DAG) whose nodes represent random variables and whose edges represent conditional probability distributions (CPDs). A graph can be represented by a tuple of vertices (or nodes) and edges :math:`\mathcal{G} = \left( \mathbf{V}, \mathbf{E} \right)`. A node :math:`\theta_{j} \in \mathbf{V}` is a parent of a random variable :math:`\theta_{i} \in \mathbf{V}` if there is a directed edge :math:`[\theta_{j} \to \theta_{i}] \in \mathbf{E}`. The set of parents of random variable :math:`\theta_{i}` is denoted by :math:`\theta_{\mathrm{pa}(i)}`. The set of all CPDs is given by :math:`\{\mathbb{P}(\theta_{i} \mid \theta_{\mathrm{pa}(i)}) : \theta_{i} \in \mathbf{V}\}` and can be used to define a factorization of the graph.
#
#The joint distribution of the variables can be expressed as a product of the set of conditional probability distributions
#
#.. math:: \mathbb{P}(\theta_{1}, \ldots \theta_{M}) = \prod_{i=1}^M \mathbb{P}(\theta_{i} \mid \theta_{\mathrm{pa}(i)}).
#
#For Gaussian networks athese CPDs take the form
#
#.. math:: \mathbb{P}(\theta_{\alpha} \mid \theta_{\mathrm{pa}(\alpha)}) \sim \mathcal{N}\left(\sum_{\beta\in\mathrm{pa}(\alpha)} A_{\alpha\beta}\theta_\beta + b_\alpha,\Sigma_{v\alpha}\right)
#
#which means that :math:`\theta_{\alpha}\sim\mathcal{N}\left(\mu_\alpha,\Sigma_{\alpha\alpha}\right)` where
#
#.. math:: \mu_\alpha=b_\alpha+A_{\alpha\beta}\mu_\beta, \qquad \Sigma_{\alpha\alpha}=\Sigma_{v\alpha}+A_{\alpha\beta}\Sigma_{\beta\beta}A_{\alpha\beta}^T
#
#The parameter relationships in the previous section can be represented by the following network.
fig,ax=plt.subplots(1,1,figsize=(8,3))
plot_peer_network(nmodels,ax)
fig.tight_layout()

#%%
#For this network we have :math:`\mathrm{pa}(1)=\emptyset,\;\mathrm{pa}(1)=\emptyset,\;\mathrm{pa}(2)=\{1,2\}` and the graph has one CPDs which for this example is given by
#
#.. math:: \mathbb{P}(\theta_3\mid \theta_1,\theta_2) \sim \mathcal{N}\left(A_{31}\theta_1+A_{32}\theta_2+b_3,\Sigma_{v3}\right),
#
#with :math:`b_3=0`.
#
#We refer to a Gaussian network based upon this DAG as a peer network. Consider the case where a high fidelity simulation model incorporates two sets of active physics, and the two low-fidelity peer models each contain one of these components. If the high-fidelity model is given, then the low-fidelity models are no longer independent of one another. In other words, information about the parameters used in one set of physics will inform the other set of physics because they are coupled together in a known way through the high-fidelity model.
#
#The joint density of the network is given by
#
#.. math:: \mathbb{P}(\theta_1,\theta_2,\theta_3)=\mathbb{P}(\theta_3\mid \theta_1,\theta_2)\mathbb{P}(\theta_1)\mathbb{P}(\theta_2)
#
#Unless a variable :math:`\theta_i` is a root node of a network, i.e. :math:`\mathrm{pa}(i)=\emptyset` the CPD is in general not Gaussian; for root nodes the CPD is just the density of :math:`\theta_i`. However we can represent the CPDs of no-root nodes and the Gaussian density of root nodes with a consistent reprsentation. Specifically we will use the canonical form
#
#.. math:: \mathcal{C}(x;K,h,g) = \exp\left(g+h^T x-\frac{1}{2} x^T K x\right)
#
#The canonical form of a normal distribution with mean :math:`m` and covariance :math:`C` has the parameters :math:`K=C^{-1}`, :math:`h = K m` and
#
#.. math:: g = -\frac{1}{2} m^T h -\frac{1}{2} n\log(2\pi) +\frac{1}{2} \log |K|.
#The derivation of these expressions is simple and we leave to the reader. The derivation of the canonical form of a linear-Gaussian CPD is more involved however and we derive it here. 
#
#First we assume that the joint density of two Gaussian variables :math:`\theta_i,\theta_j` can be represented as the product of two factors. Specifically let :math:`\phi_{\theta_i}(\theta_i,\theta_j)` be the factor representing the CPD :math:`\mathbb{P}(\theta_i\mid\theta_j)`, let :math:`\phi_{\theta_j}` be the canonical form of the Gaussian :math:`\theta_j`, and assume
#
#..  math:: \mathbb{P}(\theta_i,\theta_j)=\mathbb{P}(\theta_i\mid\theta_j)\mathbb{P}(\theta_j)=\phi_{\theta_i}\phi_{\theta_j}
#
#Given the linear relationship of the CPD :math:`\theta_i=A_ij\theta_j+v_i` the inverse of the covariance of the Gaussian joint density :math:`\mathbb{P}(\theta_i,\theta_j)` is
#
#.. math::
#
#    K&=\begin{bmatrix}\Sigma_{jj} & \Sigma_{jj}A_{ij}^T\\ A_{ij}\Sigma_{jj} & A_{ij}\Sigma_{jj}A_{ij}^T + \Sigma_{vi}\end{bmatrix}^{-1}\\
#     &=\begin{bmatrix}\Sigma_{jj}^{-1}+ A_{ij}^T\Sigma_{vi}^{-1}A_{ij} & -A_{ij}^T\Sigma_{vi}^{-1}\\ -\Sigma_{vi}^{-1}A_{ij} & \Sigma_{vi}^{-1}\end{bmatrix}
#
#where the second equality is derived using the matrix inversion lemma. Using the definition of the canonical form we for :math:`\phi_j` that :math:`K_j=\Sigma_{jj}^{-1}\in\reals^{P_j\times P_j}`. However this is a different size than :math:`K_{j}\in\reals^{(P_i+P_j)\times(P_i+P_j)}`. In the Bayesian network literature these factors are said to have different scope, i.e. they are dependent on different variables. Two multiply two canonical forms with different scopes we simply need to extend the scope of each form so that the resulting scopes match. Now with this in mind, we know that the matrix :math:`K` corresponding to the product of the two factors is the sum of the two matrices :math:`K_i,K_j` of the factors :math:`\phi_i,\phi_j`, thus we have
#
#.. math::
#
#   \begin{bmatrix}\Sigma_{jj}^{-1} & 0 \\ 0 & 0\end{bmatrix}\begin{bmatrix}K_{i11} & K_{i12} \\ K_{i21} & K_{i22}\end{bmatrix}=\begin{bmatrix}\Sigma_{jj}^{-1}+ A_{ij}^T\Sigma_{vi}^{-1}A_{ij} & -A_{ij}^T\Sigma_{vi}^{-1}\\ -\Sigma_{vi}^{-1}A_{ij} & \Sigma_{vi}^{-1}\end{bmatrix}
#
#where we have extended the scope of :math:`\phi_j` and thus added zeros to :math:`K_j`. Equating terms in the above equation yields :math:`K_{i11}=A_{ij}^T\Sigma_{vi}^{-1}A_{ij}`, :math:`K_{i12}=K_{i21}^T=-A_{ij}^T\Sigma_{vi}^{-1}` and :math:`K_{i22}=\Sigma_{vi}^{-1}`.
#
#A similar procedure can be used to find :math:`h=[(A_{ij}^T\Sigma_{vi}^{-1})^T,(\Sigma_{vi}^{-1}b)^T]^T` of the factor product which can then be used to compute the normalization :math:`g` to ensure the resulting factor can be transformed into a density which integrates to 1.
#
#For Gaussian networks the mean and covariance of this Gaussian joint density can easily be computed by utilizing the canonical form of the Gaussian PDF.
# 

#To compute the canonical form of the CPD :math:`\mathbb{P}(Y\mid X)` of two variables :math:`X,Y` (in our example  :math:`X=[\theta_1^T,\theta_2^T]^T,Y=\theta_3`)
#

# # solve using network
# cpd_scales=[a31,a32]
# prior_covs=[s11,s22,s33]
# basis_matrix_funcs = [p.basis_matrix for p in polys]
# network = build_peer_polynomial_network(
#     prior_covs,cpd_scales,basis_matrix_funcs,nparams)

# # plt.show()

# labels = [l[1] for l in network.graph.nodes.data('label')]
# network.add_data_to_network(samples_train,np.array(noise_std)**2)
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

#%%
#References
#^^^^^^^^^^
#.. [GJGEIJUQ2020] `MFNets: Multi-fidelity data-driven networks for bayesian learning and prediction, International Journal for Uncertainty Quantification, 2020. <https://www.alexgorodetsky.com/static/papers/gorodetsky_jakeman_geraci_eldred_mfnets_2020.pdf>`_
#
#.. [GJGJCP2020] `MFNets: Learning network representations for multifidelity surrogate modeling, 2020. <https://res.arxiv.org/abs/2008.02672>`_
