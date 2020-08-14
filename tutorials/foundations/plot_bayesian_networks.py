r"""Gaussian Networks
==================
This tutorial describes how to perform efficient inference on a network of Gaussian-linear models using Bayesian networks, often referred to as Gaussian networks. Bayesian networks can be constructed to provide a compact representation of joint distribution, which can be used to marginalize out variables and condition on observational data without the need to construct a full representation of the joint density over all variables.

Directed Acyclic Graphs
-----------------------
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
#Conditional probability distributions
#-------------------------------------
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
# For Gaussian networks we assume that the parameters of each node :math:`\theta_i=[\theta_{i,1},\ldots,\theta_{i,P_i}]^T` are related by
#
#.. math:: \theta_i = \sum_{j\in\mathrm{pa}(i)} A_{ij}\theta_j + b_i + v_i,
#
#where :math:`b_i\in\reals^{P_i}` is a deterministic shift, :math:`v_i` is a Gaussian noise with mean zero and covariance :math:`\Sigma_{v_i}\in\reals^{P_i\times P_i}`. Consequently, the CPDs take the form
#
# .. math:: \mathbb{P}(\theta_{i} \mid \theta_{\mathrm{pa}(i)}) \sim \mathcal{N}\left(\sum_{j\in\mathrm{pa}(i)} A_{ij}\theta_j + b_i,\Sigma_{vi}\right)
#
# which means that :math:`\theta_{i}\sim\mathcal{N}\left(\mu_i,\Sigma_{ii}\right)` where
#
# .. math:: \mu_i=b_i+A_{ij}\mu_j, \qquad \Sigma_{ii}=\Sigma_{vi}+A_{ij}\Sigma_{jj}A_{ij}^T
#
#The Canonical Form
#------------------
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
#Multiplying Canonical Forms
#---------------------------
#The **scope** of a canonical form :math:`\phi(x)` is the set of variables :math:`X` and is denoted :math:`\mathrm{Scope}[\phi]`. Consider the hierarchical structure, just discussed the scope of the 3 factors are
#
#.. math:: \mathrm{Scope}[\phi_1]=\{\theta_1\},\quad\mathrm{Scope}[\phi_2]=\{\theta_1,\theta_2\},\quad\mathrm{Scope}[\phi_3]=\{\theta_2,\theta_3\}.
#
#The multiplication of canonical factors with the same scope is simple :math:`X` is simple
#
#.. math:: \phi_1(X;K_1,h_1,g_1)\phi_2(X;K_2,h_2,g_2)=\phi_1(X;K_1+K_2,h_1+h_2,g_1+g_2).
#
#To multiply two canonical forms with different scopes we must extend the scopes to match and then apply the previous formula. This can be done by adding zeros in :math:`K` and :math:`h` of the canonical form. For example consider the two canonical factors of the :math:`\theta_1,\theta_2` with :math:`P_1=1,P_2=1`
#
#.. math:: \phi_2\left(\theta_1,\theta_2;\begin{bmatrix}K_{2,11} & K_{2,12}\\ K_{2,21} & K_{2,22}\end{bmatrix},\begin{bmatrix}h_{2,1}\\ h_{2,2} \end{bmatrix},g_2\right), \phi_3\left(\theta_2,\theta_3;\begin{bmatrix}K_{3,11} & K_{3,12}\\ K_{3,21} & K_{3,22}\end{bmatrix},\begin{bmatrix}h_{2,1}\\ h_{3,2} \end{bmatrix},g_3\right)
#
#Extending the scope and multiplying proceeds as follows
#
#.. math::
#   &\phi_2\left(\theta_1,\theta_2,\theta_3;\begin{bmatrix}K_{2,11} & K_{2,12} & 0\\ K_{2,21} & K_{2,22} & 0\\ 0 & 0 & 0\end{bmatrix},\begin{bmatrix}h_{2,1}\\ h_{2,2}\\0 \end{bmatrix},g_2\right)\phi_3\left(\theta_1,\theta_2,\theta_3;\begin{bmatrix}0 & 0 & 0 \\ 0 & K_{3,11} & K_{3,12}\\ 0 &K_{3,21} & K_{3,22}\end{bmatrix},\begin{bmatrix}h_{2,1}\\ h_{3,2} \end{bmatrix},g_3\right)\\=&\phi_{2\times 3}\left(\theta_1,\theta_2,\theta_3;\begin{bmatrix}K_{2,11} & K_{2,12} & 0\\ K_{2,21} & K_{2,22}+K_{3,11} & K_{3,12}\\ 0 & K_{3,21} & K_{3,22}\end{bmatrix},\begin{bmatrix}h_{2,1}\\ h_{2,2}+h_{3,1}\\h_{3,2} \end{bmatrix},g_2+g_3\right)
#
#Now we understand how to multiply two factors of different scope we can now discuss how to convert CPDs into canonical factors. The canonical form of a normal distribution with mean :math:`m` and covariance :math:`C` has the parameters :math:`K=C^{-1}`, :math:`h = K m` and
#
#.. math:: g = -\frac{1}{2} m^T h -\frac{1}{2} n\log(2\pi) +\frac{1}{2} \log |K|.
#
#The derivation of these expressions is simple and we leave to the reader. The derivation of the canonical form of a linear-Gaussian CPD is more involved however and we derive it here. 
#
#First we assume that the joint density of two Gaussian variables :math:`\theta_i,\theta_j` can be represented as the product of two canonical forms referred to as factors. Specifically let :math:`\phi_{i}(\theta_i,\theta_j)` be the factor representing the CPD :math:`\mathbb{P}(\theta_i\mid\theta_j)`, let :math:`\phi_{j}` be the canonical form of the Gaussian :math:`\theta_j`, and assume
#
#..  math:: \mathbb{P}(\theta_i,\theta_j)=\mathbb{P}(\theta_i\mid\theta_j)\mathbb{P}(\theta_j)=\phi_{\theta_i}\phi_{\theta_j}
#
#Given the linear relationship of the CPD :math:`\theta_i=A_ij\theta_j+v_i` the inverse of the covariance of the Gaussian joint density :math:`\mathbb{P}(\theta_i,\theta_j)` is
#
#.. math::
#
#    K_{i}&=\begin{bmatrix}\Sigma_{jj} & \Sigma_{jj}A_{ij}^T\\ A_{ij}\Sigma_{jj} & A_{ij}\Sigma_{jj}A_{ij}^T + \Sigma_{vi}\end{bmatrix}^{-1}\\
#     &=\begin{bmatrix}\Sigma_{jj}^{-1}+ A_{ij}^T\Sigma_{vi}^{-1}A_{ij} & -A_{ij}^T\Sigma_{vi}^{-1}\\ -\Sigma_{vi}^{-1}A_{ij} & \Sigma_{vi}^{-1}\end{bmatrix}
#
#where the second equality is derived using the matrix inversion lemma. Because :math:`\mathbb{P}(\theta_i)`is Gaussian we have from before that the factor :math:`\phi_j` has :math:`K_j=\Sigma_{jj}^{-1}\in\reals^{P_j\times P_j}`. Multiply he two canonical factors, making sure to account for the different scopes we have
#
#.. math::
#
#   \begin{bmatrix}\Sigma_{jj}^{-1} & 0 \\ 0 & 0\end{bmatrix}\begin{bmatrix}K_{i11} & K_{i12} \\ K_{i21} & K_{i22}\end{bmatrix}=\begin{bmatrix}\Sigma_{jj}^{-1}+ A_{ij}^T\Sigma_{vi}^{-1}A_{ij} & -A_{ij}^T\Sigma_{vi}^{-1}\\ -\Sigma_{vi}^{-1}A_{ij} & \Sigma_{vi}^{-1}\end{bmatrix}
#
#Equating terms in the above equation yields :math:`K_{i11}=A_{ij}^T\Sigma_{vi}^{-1}A_{ij}`, :math:`K_{i12}=K_{i21}^T=-A_{ij}^T\Sigma_{vi}^{-1}` and :math:`K_{i22}=\Sigma_{vi}^{-1}`.
#
#A similar procedure can be used to find :math:`h=[(A_{ij}^T\Sigma_{vi}^{-1}b)^T,(\Sigma_{vi}^{-1}b)^T]^T` and :math:`g`.
#
#We are now in a position to be able to compute the joint density of all the variables in a Gaussian Network. Note that we would never want to do this in practice because it negates the benefit of having the compact representation provided by the Gaussian network.
#
#The following builds a hierarchical network with three ndoes. Each node has :math:`P_i=2` variables.  We set the matrices :math:`A_{ij}=a_{i}jI` to be a diagonal matrix with the same entries :math:`a_{ij}` along the diagonal. This means that we are saying that only the :math:`k`-th variable :math:`k=1,\ldots,P_i` of the :math:`i`-th node is related to the :math:`k`-th variable of the :math:`j`-th node.
nnodes = 3
graph = nx.DiGraph()
prior_covs = [1,2,3]
prior_means = [-1,-2,-3]
cpd_scales  =[0.5,0.4]
node_labels = [f'Node_{ii}' for ii in range(nnodes)]
nparams = np.array([2]*3)
cpd_mats  = [None,cpd_scales[0]*np.eye(nparams[1],nparams[0]),
             cpd_scales[1]*np.eye(nparams[2],nparams[1])]

#%%
#Now we set up the directed acyclic graph providing the information to construct
#the CPDs. Specifically we specify the Gaussian distributions of all the root nodes. in this example there is just one root node :math:`\theta_i`. We then specify the parameters of the CPDs for the remaining two nodes. Here we specify the CPD covariance :math:`\Sigma_{vi}` and shift :math:`b_i` such that the mean and variance of the paramters matches those specified above. To do this we note that
#
#.. math::
#
#   \mean{\theta_2}&=\mean{A_{21}\theta_1+b_2}=A_{21}\mean{\theta_1}+b_2\\
#   \mean{\theta_3}&=A_{32}\mean{\theta_2}+b_3
#
#and so set
#
#.. math::
#
#   b_2&=\mean{\theta_2}-A_{21}\mean{\theta_1}\\
#   b_3&=\mean{\theta_3}-A_{32}\mean{\theta_2}.
#
#Similarly we define the CPD covariance so that the diagonal of the prior covariance matches the values specified
#
#.. math::
#
#   \var{\theta_2}=\Sigma_{v2}+A_{21}\var{\theta_1}A_{21}^T,\qquad \var{\theta_3}=\Sigma_{32}+A_{32}\var{\theta_2}A_{32}^T
#
# so
#
#.. math::
#
#   \Sigma_{21}=\var{\theta_2}-A_{21}\var{\theta_1}A_{21}^T,\qquad \Sigma_{32}=\var{\theta_3}-A_{32}\var{\theta_2}A_{32}^T
#
ii=0
graph.add_node(
    ii,label=node_labels[ii],cpd_cov=prior_covs[ii]*np.eye(nparams[ii]),
    nparams=nparams[ii],cpd_mat=cpd_mats[ii],
    cpd_mean=prior_means[ii]*np.ones((nparams[ii],1)))
for ii in range(1,nnodes):
    cpd_mean = np.ones((nparams[ii],1))*(
        prior_means[ii]-cpd_scales[ii-1]*prior_means[ii-1])
    cpd_cov = np.eye(nparams[ii])*max(
        1e-8,prior_covs[ii]-cpd_scales[ii-1]**2*prior_covs[ii-1])
    graph.add_node(ii,label=node_labels[ii],cpd_cov=cpd_cov,
                   nparams=nparams[ii],cpd_mat=cpd_mats[ii],
                   cpd_mean=cpd_mean)

graph.add_edges_from([(ii,ii+1) for ii in range(nnodes-1)])


network = GaussianNetwork(graph)
network.convert_to_compact_factors()
labels = [l[1] for l in network.graph.nodes.data('label')]
factor_prior = network.factors[0]
for ii in range(1,len(network.factors)):
    factor_prior *= network.factors[ii]
prior_mean,prior_cov = convert_gaussian_from_canonical_form(
    factor_prior.precision_matrix,factor_prior.shift)
print('Prior Mean\n',prior_mean)
print('Prior Covariance\n',prior_cov)

#%%
#We can check the mean and covariance  diagonal  of the prior match the values we specified
true_prior_mean = np.hstack(
    [[prior_means[ii]]*nparams[ii] for ii in range(nnodes)])
assert np.allclose(true_prior_mean,prior_mean)
true_prior_var = np.hstack(
    [[prior_covs[ii]]*nparams[ii] for ii in range(nnodes)])
assert np.allclose(true_prior_var,np.diag(prior_cov))

#%%
#If the reader is interested they can also compare the entire prior covariance with
#
#.. math:: \begin{bmatrix}\Sigma_{11} & \Sigma_{11}A_{12}^T & \Sigma_{11}A_{12}^TA_{32}^T \\ A_{21}\Sigma_{11} & \Sigma_{22} & \Sigma_{22}A_{32}^T\\ A_{32}A_{21}\Sigma_{11} & A_{32}\Sigma_{22} & \Sigma_{33}\end{bmatrix}
#
#Conditioning The Canonical Form
#-------------------------------
#Using the canonical form of these factors we can easily condition them on available data. Given a canonical form over two variables :math:`X,Y` with
#
#.. math:: K=\begin{bmatrix} K_{XX} & K_{XY}\\ K_{YX} & K_{YY}\end{bmatrix}, \qquad h=\begin{bmatrix}h_{X} \\h_{Y}\end{bmatrix}
#
#and given data :math:`y` (also called evidence in the literature) the paramterization of the canonical form of the factor conditioned on the data is simply
#
#.. math:: K^\prime=K_{XX}, \quad h^\prime=h_X-K_{XY}y, \quad g^\prime=g+h_Y^Ty-\frac{1}{2}y^TK_{yy}y
#
#Thus the canonical form of a CPD :math:`\mathbb{P}(\theta_i\mid Y_i=y_i)` has the parameters
#
#.. math:: K^\prime=\Phi_i\Sigma_{\epsilon_i}^{-1}\Phi_i^T, \qquad h^\prime=\Phi_i^T\Sigma_{\epsilon_i}^{-1}b,
#We then combine this conditioned CPD factor with its parent factor (associated with the prior distribution of the parameters :math:`\theta_i` by multiplying these two factors together after eliminating the data variables from the scope of the CPD. This is called the sum-product eliminate variable algorithm. The combined factor has parameters
#
#.. math:: K=\Phi_i\Sigma_{\epsilon_i}^{-1}\Phi_i^T+\Sigma_{ii}^{-1}, \qquad h=\Sigma_{ii}^{-1}\mu_i+\Phi_i^T\Sigma_{\epsilon_i}^{-1}y
#
#which represents a Gaussian with mean and covariance given by
#
#.. math:: \Sigma^\mathrm{post}=\left(\Phi_i\Sigma_{\epsilon_i}^{-1}\Phi_i^T+\Sigma_{ii}^{-1}\right)^{-1}, \qquad \mu^\mathrm{post} = \Sigma^\mathrm{post}\left(\Sigma_{ii}^{-1}\mu_i+\Phi_i^T\Sigma_{\epsilon_i}^{-1}y\right)
#
#which is just the usual expression for the posterior of a gaussian linear model using only the linear model, noise and prior associated with a single node. Here we used the relationship between the canonical factors and the covariance :math:`C` and mean :math:`m` of the equivalent Gaussian distribution
#
#.. math:: C=K^{-1}, \qquad m=Ch
#
#After conditioning all nodes on the available data we have three factors remaining to obtain the joint posterior distribution over the variables of the remaining nodes we simply need to multiply the factors together again one at a time. Starting with one root node we collapse the CPDs one at a time.
#Marginalizing Canonical Forms
#-----------------------------

#%%
#References
#^^^^^^^^^^
#.. [KFPGM2009] `Probabilistic Graphical Models: Principles and Techinques. 2009 <https://mitpress.mit.edu/books/probabilistic-graphical-models>`_
