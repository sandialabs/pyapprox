r"""
Approximate Control Variate Allocation Matrices
=============================================================

Optimizing and constructing PACV estimators requires estimating :math:`\covar{Q_0}{\Delta}` and :math:`\covar{\Delta}{\Delta}` which depend on the statistical properties of the models being used but also on the sample alloaction  how independent sample sets are shared among the sample sets :math:`\rvset_\alpha\text{ and }\rvset_\alpha^*`.

For example, when computing the mean of a scalar model when the covariance  between models is given by :math:`\mat{C}` and the first row of :math:`\mat{C}` is :math:`\mat{c}` then

.. math:: \covar{Q_0}{\mat{\Delta}} = \mat{g} \circ \mat{c}

.. math:: g_i = \frac{N_{i^*\cap 0}}{N_{i^*}N_0}-\frac{N_{i\cap 0}}{N_{i}N_0}, \qquad i=1,\ldots,M

and 

.. math:: \covar{\mat{\Delta}}{\mat{\Delta}} = \mat{G} \circ \mat{C}

.. math:: G_{ij} = \frac{N_{i^*\cap j^*}}{N_{i^*}N_{j^*}}-\frac{N_{i^*\cap j}}{N_{i^*}N_j}-\frac{N_{i\cap j^*}}{N_{i}N_{j^*}}+\frac{N_{i\cap j}}{N_{i}N_j}, \qquad i,j=1,\ldots,M

where :math:`\circ` is the Haddamard (element-wise) product.


The implementation of PACV uses allocation matrices :math:`\mat{A}` that encode how independent sample sets are shared among the sample sets :math:`\rvset_\alpha\text{ and }\rvset_\alpha^*`.

For example, the allocation matrix of ACVMF using :math:`M=3` low-fidelity models (GMF with recursion index :math:`(0,0,0)`) is

.. math::

   \mat{A}=\begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}

An entry of one indicates in the ith row of the jth column indicates that the ith independent sample set is used in the corresponding set :math:`\rvset_j` if j is odd or :math:`\rvset_j^*` if j is even. The first column will always only contain zeros because the set :math:`\rvset_0^*` is never used by ACV estimators.


Note, here we focus on how to construct and use allocation matrices for ACVMF like PACV estimtors. However, much of the discussion carries over to other estimators like those based on recursive difference and ACVIS.


The allocation matrix together with the number of points :math:`\mat{p}=[p_0,\ldots,p_M]^\top` in the each independent sample set which we call partitions can be used to determine the  number of points in the intersection of the sets  :math:`\rvset_\alpha\text{ and }\rvset_\beta`, where :math:`\alpha\text{ and }\beta` may be unstarred or starred. Intersection matrices are used to compute :math:`\covar{Q_0}{\Delta}` and :math:`\covar{\Delta}{\Delta}`.
The number of intersection points can be represented by a :math:`2(M+1)\times 2(M+1)` matrix

.. math::

   \mat{B}=\begin{bmatrix}
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
   0 & N_{0\cup 0} & N_{0\cup 1} & N_{0\cup 1^*} & N_{0\cup 2} & N_{0\cup 2^*} &  N_{0\cup 3} & N_{0\cup 3^*}\\
   0 & N_{1^*\cup 0} & N_{1^*\cup 1^*} & N_{1^*\cup 1} & N_{1^*\cup 2^*} & N_{1^*\cup 2} &  N_{1^*\cup 3^*} & N_{1^*\cup 3}\\
   0 & N_{1\cup 0} & N_{1\cup 1^*} & N_{1\cup 1} & N_{1\cup 2^*} & N_{1\cup 2} &  N_{1\cup 3^*} & N_{1\cup 3}\\
   \vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots\\
   0 & N_{3^\star\cup 0} & N_{3^\star\cup 1^*} & N_{3^\star\cup 1} & N_{3^\star\cup 2^*} & N_{3^\star\cup 2} &  N_{3^\star\cup 3^*} & N_{3^\star\cup 3}\\
   0 & N_{3\cup 0} & N_{3\cup 1^*} & N_{3\cup 1} & N_{3\cup 2^*} & N_{3\cup 2} &  N_{3\cup 3^*} & N_{3\cup 3}
   \end{bmatrix}

Defining


.. math:: \mat{S}=\text{Diag}(\mat{p})\mat{A},

each entry of B can be computed using

.. math:: B_{ij}=\sum_{k=0}^{2M+1} S_{kj} \chi[A_{ki}]

where

.. math::  \chi[A_{ki}]=\begin{cases} 1 & A_{ki}=1 \\ 0 & A_{ki}=0 \end{cases}


Note computing :math:`\covar{Q_0}{\Delta}` and :math:`\covar{\Delta}{\Delta}` also requires the number of samples :math:`N_\alpha`,  :math:`N_{\alpha^*}` in each ACV sample set :math:`\rvset_\alpha,\rvset_\alpha^*` which can be computed by simply summing the rows of S or more simply extracting the relevant entry from the diagonal of B. For example

.. math:: N_3 = N_{3\cap3} \text{ and }  N_{3^*} = N_{3^*\cap3^*}

Finally, to compute the computational cost of the estimator we must be able to compute the number of samples per model.
The number of samples of the ith

.. math:: \sum_{k=1}^M p_{k}\chi[A_{2i,k}+A_{2i+1,k}]

where :math:`\chi[A_{2i,k}+A_{2i+1,k}]=1` if :math:`A_{2i,k}+A_{2i+1,k}>0` and is zero otherwise.

Using our example, the intersection matrix is

.. math:: \mat{p} = [2, 3, 4, 5]

.. math:: \mat{S}=\begin{bmatrix}
   0 & 2 & 2 & 2 & 2 & 2 & 2 & 2\\
   0 & 0 & 0 & 3 & 0 & 3 & 0 & 3\\
   0 & 0 & 0 & 0 & 0 & 4 & 0 & 4\\
   0 & 0 & 0 & 0 & 0 & 5 & 0 & 0
   \end{bmatrix}

So summing each column of S we have

.. math:: N_0^*=0, N_0=2,  N_1^*=2, N_1=5,  N_2^*=2, N_2=14,  N_3^*=2, N_3=9, 

And 

.. math::

   \mat{B}=\begin{bmatrix}
  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 \\
  0 &  2 &  2 &  2 &  2 &  2 &  2 &  2 \\
  0 &  2 &  2 &  2 &  2 &  2 &  2 &  2 \\
  0 &  2 &  2 &  5 &  2 &  5 &  2 &  5 \\
  0 &  2 &  2 &  2 &  2 &  2 &  2 &  2 \\
  0 &  2 &  2 &  5 &  2 & 14 &  2 &  9 \\
  0 &  2 &  2 &  2 &  2 &  2 &  2 &  2 \\
  0 &  2 &  2 &  5 &  2 &  9 &  2 &  9
   \end{bmatrix}


The first row and column are all zero because :math:`\rvset_0^*` is always empty.

As an example the thrid entry from the right on the bottom row corresponds to :math:`B_{75}=N_{3\cup 2}` which is computed by finding the rows in R that have non zero entries which are the first three rows. Then the :math:`B_{75}` is the sum of these rows in the 5 column of S, i.e. :math:`2+3+4=9`.

Examples of different allocation matrices
-----------------------------------------
The following lists some example alloaction matrices for the
parameterically defined ACV estimators based on the general structure of
ACVMF.

MFMC :math:`(0, 1, 2)`

.. math::

   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}


:math:`(0, 0, 1)`

.. math::

   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 0 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}

:math:`(0, 0, 2)`

.. math::

   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 0 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}


:math:`(0, 1, 1)`

.. math::

   \begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}

How to construct ACVMF like allocation matrices
-----------------------------------------------
The following shows the general procedure to construct ACVMF like estimators for a recursion index

.. math:: \gamma=(\gamma_1,\ldots,\gamma_M)

As a concrete example, consider the recursion index :math:`\gamma=(0, 0, 1)`.

First we set :math:`A_{2\alpha+1,\alpha}=1` for all :math:`\alpha=0,\ldots,M`

E.g. for (0,0,1)

.. math::

   \begin{bmatrix}
  0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
  \end{bmatrix}

We then set :math:`A_{2\alpha,\gamma_\alpha}=1` for :math:`\alpha=1,\ldots,M`. For example,

.. math::

   \begin{bmatrix}
 0 & 1 & 1 & 0 & 1 & 0 & 0 & 0 \\
 0 & 0 & 0 & 1 & 0 & 0 & 1 & 0 \\
 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}

And finally because ACMF always uses all independent partitions up to including :math:`\gamma_\alpha` we set :math:`A_{i,k}=1`, for :math:`k=0,\ldots,\gamma_\alpha` and :math:`\forall i`. For example

.. math::

   \begin{bmatrix}
 0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
 0 & 0 & 0 & 1 & 0 & 1 & 1 & 1 \\
 0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}

The following can be used to plot the allocation matrix of any PACV estimator (not just GMF). Note we load a benchmark because it is needed to initialize the PACV estimator, but the allocation matrix is independent of any benchmark properties other than the number of models it provides
"""
import matplotlib.pyplot as plt

from pyapprox.benchmarks import setup_benchmark
from pyapprox.multifidelity.factory import get_estimator, multioutput_stats

benchmark = setup_benchmark("tunable_model_ensemble")
model = benchmark.fun

stat = multioutput_stats["mean"](benchmark.nqoi)
stat.set_pilot_quantities(benchmark.covariance)
est = get_estimator("grd", stat, model.costs(), recursion_index=(2, 0))

ax = plt.subplots(1, 1, figsize=(8, 6))[1]
_ = est.plot_allocation(ax)

#%%
#The different colors represent different independent sample partitions. Subsets :math:`\rvset_\alpha\text{ or  }\rvset_\alpha^*` having the same color, means that the same set of samples are used in each subset.
#
#Try changing recursion index to (0, 1) or (2, 0) and the estimator from "gis" to "gmf" or" grd"

#%%
#Evaluating a PACV estimator
#---------------------------
#Allocation matrices are also useful for evaluating a PACV estimator.
#To evaluate the ACV estimator we must construct each independent sample partition from a set of :math:`N_\text{tot}` samples :math:`\rvset_\text{tot}` where
#
#.. math:: N_\text{tot}=\sum_{\alpha=0}^{M} p_\alpha
#
#We then must allocate each model on a subset of these samples dictated by the allocation matrix. For a model index :math:`\alpha` we must evaluate a model at a independent partition k if :math:`A_{2\alpha, k}=1` or :math:`A_{2\alpha+1, k}=1`. These correspond to the sets :math:`\rvset_{\alpha}^*, \rvset_{\alpha}`. We store these active partitions in a flattened sample array for each model ordered by increasing partition index, which is passed to each user. E.g. if the partitions 0, 1, 3 are active then we store :math:`[\rvset_{0}^\dagger, \rvset_{1}^\dagger, \rvset_{3}^\dagger]` where the dagger indicates the samples sets are associated with partitions and not the estimator sets :math:`\rvset_{\alpha^*}, \rvset_{\alpha}`.
#
#The user can then evaluate each model without knowing anything about the ACV estimator sets or the independent partitions.
#
#These model evaluations are then passed back to the estimator and internally we must assigne the values to each ACV estimator sample set. Specifically for each model :math:`\alpha`, we loop through all partition indices k and if :math:`A_{2\alpha,k}=1` we assign :math:`\mathcal{f_\alpha(\rvset^\dagger_k)}` to :math:`\rvset_{\alpha^*}` similarly if :math:`A_{2\alpha+1,k}=1` we assign :math:`\mathcal{f_\alpha(\rvset^\dagger_k)}` to :math:`\rvset_{\alpha}`.
