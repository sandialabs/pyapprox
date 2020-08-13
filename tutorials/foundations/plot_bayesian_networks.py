r"""Gaussian Networks
==================
This tutorial describes how to perform efficient inference on a network of Gaussian-linear models using Bayesian networks, often referred to as Gaussian networks. Bayesian networks can be constructed to provide a compact representation of joint distribution, which can be used to marginalize out variables and condition on observational data without the need to construct a full representation of the joint density over all variables.

A Bayesian network (BN) structure is a directed acyclic graphs (DAG) whose nodes represent random variables and whose edges represent probabilistic relationships between them. The graph can be represented by a tuple of vertices (or nodes) and edges :math:`\mathcal{G} = \left( \mathbf{V}, \mathbf{E} \right)`. A node :math:`\theta_{j} \in \mathbf{V}` is a parent of a random variable :math:`\theta_{i} \in \mathbf{V}` if there is a directed edge :math:`[\theta_{j} \to \theta_{i}] \in \mathbf{E}`. The set of parents of random variable :math:`\theta_{i}` is denoted by :math:`\theta_{\mathrm{pa}(i)}`.

Lets import some necessary modules and then construct a DAG consisting of 3 groups of variables.
"""
import numpy as np
from scipy import stats
import pyapprox as pya
from pyapprox.gaussian_network import *
import copy
from pyapprox.configure_plots import *

nmodels=3
fig,ax=plt.subplots(1,1,figsize=(8,3))
_=plot_hierarchical_network(nmodels,ax)
fig.tight_layout()

#%%
#For this network we have :math:`\mathrm{pa}(\theta_1)=\emptyset,\;\mathrm{pa}(\theta_2)=\{\theta_1\},\;\mathrm{pa}(\theta_3)=\{\theta_2\}`.
#
#Bayesian networks use onditional probability distributions (CPDs) to encode the relationships between variables of the graph. Let :math:`A,B,C` denote three random variables. :math:`A` is conditionally independent of :math:`B` given :math:`C` in distribution :math:`\mathbb{P}`, denoted by :math:`A \mathrel{\perp\mspace{-10mu}\perp} B \mid C`, if
#
# .. math:: \mathbb{P}(A = a, B = b \mid C = c)  = \mathbb{P}(A = a\mid C=c) \mathbb{P}(B = b \mid C = c).
#
#The above graph encodes that :math:`\theta_1\mathrel{\perp\mspace{-10mu}\perp} \theta_3 \mid \theta_2`. CPDs, such as this, can be used to form a compact representation of the joint density between all variables in the graph. Specifically, the joint distribution of the variables can be expressed as a product of the set of conditional probability distributions
#
# .. math:: \mathbb{P}(\theta_{1}, \ldots \theta_{M}) = \prod_{i=1}^M \mathbb{P}(\theta_{i} \mid \theta_{\mathrm{pa}(i)}).
#
#For our example we have
#
#.. math:: \mathbb{P}(\theta_1,\theta_2,\theta_3)= \mathbb{P}(\theta_1\mid\mathrm{pa}(\theta_1))\mathbb{P}(\theta_2\mid\mathrm{pa}(\theta_2))\mathbb{P}(\theta_3\mid\mathrm{pa}(\theta_3))=\mathbb{P}(\theta_1)\mathbb{P}(\theta_2\mid \theta_1)\mathbb{P}(\theta_3\mid\theta_2)
#
#Any Bayesian networks can be made up of three types of structures. The hierarchical (or serial) structure we just plotted and the diverging and peer (or V-) structure.
#
#For the peer network plotted using the code below we have :math:`\mathrm{pa}(\theta_1)=\emptyset,\;\mathrm{pa}(\theta_2)=\emptyset,\;\mathrm{pa}(\theta_3)=\{\theta_1,\theta_2\}` and
#
#.. math:: \mathbb{P}(\theta_1,\theta_2,\theta_3)= \mathbb{P}(\theta_1\mid\mathrm{pa}(\theta_1))\mathbb{P}(\theta_2\mid\mathrm{pa}(\theta_2))\mathbb{P}(\theta_3\mid\mathrm{pa}(\theta_3))=\mathbb{P}(\theta_1)\mathbb{P}(\theta_2)\mathbb{P}(\theta_3\mid\theta_1,\theta_2)

nmodels=3
fig,ax=plt.subplots(1,1,figsize=(8,3))
_=plot_peer_network(nmodels,ax)
fig.tight_layout()

#%%
#For the diverging network plotted using the code below we have :math:`\mathrm{pa}(\theta_1)=\{\theta_3\},\;\mathrm{pa}(\theta_2)=\{\theta_3\},\;\mathrm{pa}(\theta_3)=\emptyset` and 
#
#.. math:: \mathbb{P}(\theta_1,\theta_2,\theta_3)= \mathbb{P}(\theta_1\mid\mathrm{pa}(\theta_1))\mathbb{P}(\theta_2\mid\mathrm{pa}(\theta_2))\mathbb{P}(\theta_3\mid\mathrm{pa}(\theta_3))=\mathbb{P}(\theta_3)\mathbb{P}(\theta_1\mid \theta_3)\mathbb{P}(\theta_2\mid\theta_3)

nmodels=3
fig,ax=plt.subplots(1,1,figsize=(8,3))
_=plot_diverging_network(nmodels,ax)
fig.tight_layout()

#%%
# For Gaussian networks we assume that the parameters of each node are related by
#
#.. math:: \theta_i = \sum_{j\in\mathrm{pa}(i)} A_{ij}\theta_j + b_i + v_i,
#
#where :math:`b_i\in\reals^{P_i}` is a deterministic shift, :math:`v_i` is a Gaussian noise with mean zero and covariance :math:`\Sigma_{v_\alpha}\in\reals^{P_\alpha\times P_\alpha}. Consequently, the CPDs take the form
#
# .. math:: \mathbb{P}(\theta_{i} \mid \theta_{\mathrm{pa}(i)}) \sim \mathcal{N}\left(\sum_{j\in\mathrm{pa}(i)} A_{ij}\theta_j + b_i,\Sigma_{vi}\right)
#
# which means that :math:`\theta_{i}\sim\mathcal{N}\left(\mu_i,\Sigma_{ii}\right)` where
#
# .. math:: \mu_i=b_i+A_{ij}\mu_j, \qquad \Sigma_{ii}=\Sigma_{vi}+A_{ij}\Sigma_{jj}A_{ij}^T
#
#Unless a variable :math:`\theta_i` is a root node of a network, i.e. :math:`\mathrm{pa}(\theta_i)=\emptyset` the CPD is in general not Gaussian; for root nodes the CPD is just the density of :math:`\theta_i`. However we can represent the CPDs of no-root nodes and the Gaussian density of root nodes with a consistent reprsentation. Specifically we will use the canonical form of a set of variables :math:`X`
#
#.. math:: \mathcal{\phi}(x;K,h,g) = \exp\left(g+h^T x-\frac{1}{2} x^T K x\right)
#
#This canonical form can be used to represent each CPD in a graph such that
#
#.. math:: \mathbb{P}(\theta_1,\ldots,\theta_M) = \prod_{i=1}^M \phi_i 
#
#For example, the hierarchical structure can be represented by three canonical factors :math:`\phi_1(\theta_1),\phi_2(\theta_1,\theta_2),\phi_3(\theta_2,\theta_3)` such that
#
#.. math:: \mathbb{P}(\theta_1,\theta_2,\theta_3) = \phi_1(\theta_1)\phi_2(\theta_1,\theta_2)\phi_3(\theta_2,\theta_3)
#
#where we have dropped the dependence on :math:`K,h,g` for convenience. These three factors respectively correspond to the three CPDs :math:`\mathbb{P}(\theta_1),\mathbb{P}(\theta_2\mid\theta_1),\mathbb{P}(\theta_3\mid\theta_2)`
#
#To form the joint density (which we want to avoid in practice) and perform inference and marginalization of a Gaussian graph we must first understand the notion of **scope** and three basic operations on canonical factors: **multiplication**, **marginalization** and **conditioning** also knows as reduction.
#
#The **scope** of a canonical form :math:`\phi(x)` is the set of variables :math:`X` and is denoted :math:`\mathrm{Scope}[\phi]`. Consider the hierarchical structure, just discussed the scope of the 3 factors are
#
#.. math:: \mathrm{Scope}[\phi_1]=\{\theta_1\},\mathrm{Scope}[\phi_2]=\{\theta_1,\theta_2\},\mathrm{Scope}[\phi_3]=\{\theta_2,\theta_3\}.
#
#The multiplication of canonical factors with the same scope is simple :math:`X` is simple
#
#.. math:: \phi_1(X,K_1,h_1,g_1)\phi_2(X,K_2,h_2,g_2)=\phi_1(X,K_1+K_2,h_1+h_2,g_1+g_2).
#
#To multiply two canonical forms with different scopes we must extend the scopes to match and then apply the previous formula. This can be done by adding zeros in :math:`K` and :math:`h` of the canonical form. For example consider the two canonical factors
#
#..math:: \phi_2(\theta_1,\theta_2,K_2,h_2,g_2),  \phi_3(\theta_2,\theta_3,K_2,h_3)
#
#Extending the scope and multiplying proceeds as follows
#
#..math:: \phi_2(\theta_1,\theta_2,\begin{bmatrix}K_2 & 0\\ 0 & 0\end{bmatrix},\begin{bmatrix}h_2\\ 0 \end{bmatrix},g_2)\phi_2(\theta_1,\theta_2,\begin{bmatrix}0 & 0\\ 0 & K_3\end{bmatrix},\begin{bmatrix}0\\ h_3 \end{bmatrix},g_3)
#
#of the 3 node hierarchial structure
#
#The canonical form of a normal distribution with mean :math:`m` and covariance :math:`C` has the parameters :math:`K=C^{-1}`, :math:`h = K m` and
#
#.. math:: g = -\frac{1}{2} m^T h -\frac{1}{2} n\log(2\pi) +\frac{1}{2} \log |K|.
#
#The derivation of these expressions is simple and we leave to the reader. The derivation of the canonical form of a linear-Gaussian CPD is more involved however and we derive it here. 
#
#First we assume that the joint density of two Gaussian variables :math:`\theta_i,\theta_j` can be represented as the product of two canonical forms referred to as factors. Specifically let :math:`\phi_{\theta_i}(\theta_i,\theta_j)` be the factor representing the CPD :math:`\mathbb{P}(\theta_i\mid\theta_j)`, let :math:`\phi_{\theta_j}` be the canonical form of the Gaussian :math:`\theta_j`, and assume
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
#A similar procedure can be used to find :math:`h=[(A_{ij}^T\Sigma_{vi}^{-1}b)^T,(\Sigma_{vi}^{-1}b)^T]^T` of the factor product which can then be used to compute the normalization :math:`g` to ensure the resulting factor can be transformed into a density which integrates to 1.


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
# #Using this graph we can infer the posterior distribution of the information source coefficients using the conditional probability variable elimination algorithm. The algorithm begins by conditioning the each factor of the graph with any data associated with that factor. The graph above will have 3 factors involving data associated with the CPDs :math:`\mathbb{P}(\theta_1\mid y_1),\mathbb{P}(\theta_2\mid y_2),\mathbb{P}(\theta_3\mid y_3)`. Using the canonical form of these factors we can easily condition them on available data. Given a canonical form over two variables :math:`X,Y` with
# #
# #.. math:: K=\begin{bmatrix} K_{XX} & K_{XY}\\ K_{YX} & K_{YY}\end{bmatrix}, \qquad h=\begin{bmatrix}h_{X} \\h_{Y}\end{bmatrix}
# #
# #and given data :math:`y` (also called evidence in the literature) the paramterization of the canonical form of the factor conditioned on the data is simply
# #
# #.. math:: K^\prime=K_{XX}, \quad h^\prime=h_X-K_{XY}y, \quad g^\prime=g+h_Y^Ty-\frac{1}{2}y^TK_{yy}y
# #
# #Thus the canonical form of a CPD :math:`\mathbb{P}(\theta_i\mid Y_i=y_i)` has the parameters
# #
# #.. math:: K^\prime=\Phi_i\Sigma_{\epsilon_i}^{-1}\Phi_i^T, \qquad h^\prime=\Phi_i^T\Sigma_{\epsilon_i}^{-1}b,
# #We then combine this conditioned CPD factor with its parent factor (associated with the prior distribution of the parameters :math:`\theta_i` by multiplying these two factors together after eliminating the data variables from the scope of the CPD. This is called the sum-product eliminate variable algorithm. The combined factor has parameters
# #
# #.. math:: K=\Phi_i\Sigma_{\epsilon_i}^{-1}\Phi_i^T+\Sigma_{ii}^{-1}, \qquad h=\Sigma_{ii}^{-1}\mu_i+\Phi_i^T\Sigma_{\epsilon_i}^{-1}y
# #
# #which represents a Gaussian with mean and covariance given by
# #
# #.. math:: \Sigma^\mathrm{post}=\left(\Phi_i\Sigma_{\epsilon_i}^{-1}\Phi_i^T+\Sigma_{ii}^{-1}\right)^{-1}, \qquad \mu^\mathrm{post} = \Sigma^\mathrm{post}\left(\Sigma_{ii}^{-1}\mu_i+\Phi_i^T\Sigma_{\epsilon_i}^{-1}y\right)
# #
# #which is just the usual expression for the posterior of a gaussian linear model using only the linear model, noise and prior associated with a single node. Here we used the relationship between the canonical factors and the covariance :math:`C` and mean :math:`m` of the equivalent Gaussian distribution
# #
# #.. math:: C=K^{-1}, \qquad m=Ch
# #
# #After conditioning all nodes on the available data we have three factors remaining to obtain the joint posterior distribution over the variables of the remaining nodes we simply need to multiply the factors together again one at a time. Starting with one root node we collapse the CPDs one at a time.

#The set of all CPDs is given by :math:`\{\mathbb{P}(\theta_{i} \mid \theta_{\mathrm{pa}(i)}) : \theta_{i} \in \mathbf{V}\}` and can be used to define a factorization of the graph.
