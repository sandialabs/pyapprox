r"""
Parameterized Approximate Control Variate Monte Carlo
=====================================================

Allocation matrices
-------------------
The implementation of PACV uses allocation matrices which encode how independent sample sets are shared among the sample sets :math:`\mathcal{Z}_\alpha\text{ and }\mathcal{Z}_\alpha^*`.

Here we focus on how to construct and use allocation matrices for ACVMF like PACV estimtors. However, much of the discussion carries over to other estimators like those based on recursive difference and ACVIS.

The allocation matrix of ACVMF using :math:`M=3` low-fidelity models (with recursion index :math:`(0,0,0)`) is

.. math::

   A=\begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}

An entry of one indicates in the ith row of the jth column indicates that the ith independent sample set is used in the corresponding set :math:`\mathcal{Z}_j` if j is odd or :math:`\mathcal{Z}_j^*` if j is even.

The base matrix above must be reordered to account for the number of samples per model N. Specifically, the columns must be rearanged acording to the order that would make N be in ascending order.
For example, :math:`N=[2,5,14,9]` we need to make sure that the columns corresponding to :math:`\mathcal{Z}_3\text{ and }\mathcal{Z}_3^*` are swaped with the columns associated with :math:`\mathcal{Z}_4\text{ and }\mathcal{Z}_4^*`. This produces the reordered matrix

.. math::

   R=\begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 0
   \end{bmatrix}

The reordered allocation matrix can be used to determine number of points :math:`P=[P_0,\ldots,P_M]` in the each independent sample set which we call partitions. For ACVMF like allocations we use the following algorithm::

  idx = np.unique(N, return_inverse=True)[1]
  N_unique = N[idx]
  pad = np.full((nmodels-N_unique.shape[0]), N_unique[-1])
  N_sort = np.hstack((N_unique, pad))
  P = np.hstack((N[0], np.diff(N_sort)))

This code is based on the observation that each model (when ordered by number of samples will have one more independent set (partition) than the previous model. Thus we sort the array the compute the difference between consecutive entries of N with np.diff. The padding ensures that if two models have the same number of samples that they use the same partitions as eachother. For example when :math:`N=[2,4,4,6]`::

  N_unique = [2,4,6]
  pad = [6]
  N_sort = [2,4,6,6]
  P = [2,2,2,0]

Meaning that only three partitions have a non-zero number of samples. Note this aglorithm is only useful for parmeterized versions of ACVMF sample alloactions. A different procedure must be used for recursive difference and ACVIS allocations.

The reordered allocation matrix can be used to determine number of points in the intersection of the sets  :math:`\mathcal{Z}_\alpha\text{ and }\mathcal{Z}_\beta`, where :math:`\alpha\text{ and }\beta` may be unstarred or starred. Intersection matrices are used to compute :math:`\covar{Q_0}{\Delta}` and :math:`\covar{\Delta}{\Delta}`.
The number of intersection points can be represented by a :math:`2(M+1)\times 2(M+1)` matrix

.. math::

   B=\begin{bmatrix}
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
   0 & N_{0\cup 0} & N_{0\cup 1} & N_{0\cup 1^*} & N_{0\cup 2} & N_{0\cup 2^*} &  N_{0\cup 3} & N_{0\cup 3^*}\\
   0 & N_{1^*\cup 0} & N_{1^*\cup 1^*} & N_{1^*\cup 1} & N_{1^*\cup 2^*} & N_{1^*\cup 2} &  N_{1^*\cup 3^*} & N_{1^*\cup 3}\\
   0 & N_{1\cup 0} & N_{1\cup 1^*} & N_{1\cup 1} & N_{1\cup 2^*} & N_{1\cup 2} &  N_{1\cup 3^*} & N_{1\cup 3}\\
   \vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots &\vdots\\
   0 & N_{3^\star\cup 0} & N_{3^\star\cup 1^*} & N_{3^\star\cup 1} & N_{3^\star\cup 2^*} & N_{3^\star\cup 2} &  N_{3^\star\cup 3^*} & N_{3^\star\cup 3}\\
   0 & N_{3\cup 0} & N_{3\cup 1^*} & N_{3\cup 1} & N_{3\cup 2^*} & N_{3\cup 2} &  N_{3\cup 3^*} & N_{3\cup 3}
   \end{bmatrix}

B can be computed from the allocation matrix by first determining the sample allocation matrix

.. math:: S=\text{Diag}[P]R

which is the matrix product of :math:`\text{Diag}[P]\text{ and }R`.
Then each entry of B can be computed using

.. math:: B_{ij}=\sum_{k=0}^{2M+1} S_{kj} \chi[A_{ki}]

where

.. math::  \chi[A_{ki}]=\begin{cases} 1 & A_{ki}=1 \\ 0 & A_{ki}=0 \end{cases}

Note computing :math:`\covar{Q_0}{\Delta}` and :math:`\covar{\Delta}{\Delta}` also requires the number of samples :math:`N_\alpha`,  :math:`N_{\alpha^*}` in each ACV sample set :math:`\mathcal{Z}_\alpha,\mathcal{Z}_\alpha^*` which can be computed by simply summing the rows of S. For example, when computing the mean of a model when the covariance  between models is given by C and the first row of C is c then

.. math:: \covar{Q_0}{\Delta} = g \circ c

.. math:: g_i = \frac{N_{i^*\cap 0}}{N_{i^*}N_0}-\frac{N_{i\cap 0}}{N_{i}N_0}, \qquad i=1,\ldots,M

and 

.. math:: \covar{\Delta}{\Delta} = G \circ C

.. math:: G_{ij} = \frac{N_{i^*\cap j^*}}{N_{i^*}N_{j^*}}-\frac{N_{i^*\cap j}}{N_{i^*}N_j}-\frac{N_{i\cap j^*}}{N_{i}N_{j^*}}+\frac{N_{i\cap j}}{N_{i}N_j}, \qquad i,j=1,\ldots,M

where :math:`\circ` is the Haddamard (element-wise) product.

The optimization of ACV estimators returns the number of evaluations of each model N. To evaluate the estimator we have to generate independent sample partitions that :math:`N_\alpha\cup N_\alpha^*=N` for each model index :math:`\alpha`.

To evaluate the ACV estimator we must construct each independent sample partition. The total number of independent samples is

.. math:: N_\text{tot}=\sum_{\alpha=0}^{M} P_\alpha

We generate a set of :math:`N_\text{tot}` samples :math:`\mathcal{Z}_\text{tot}`. We then must allocate each model on a subset of these samples dictated by the reordered allocation matrix. For a model index :math:`\alpha` we must evaluate a model at a independent partition k if :math:`R_{2\alpha, k}=1` or :math:`R_{2\alpha+1, k}=1`. These correspond to the sets :math:`\mathcal{Z}_{\alpha}^*, \mathcal{Z}_{\alpha}`. We store these active partitions in a flattened sample array for each model ordered by increasing partition index, which is passed to each user. E.g. if the partitions 0, 1, 3 are active then we store :math:`[\mathcal{Z}_{0}^\dagger, \mathcal{Z}_{1}^\dagger, \mathcal{Z}_{3}^\dagger]` where the dagger indicates the samples sets are associated with partitions and not the estimator sets :math:`\mathcal{Z}_{\alpha^*}, \mathcal{Z}_{\alpha}`.

The user can then evaluate each model without knowing anything about the ACV estimator sets or the independent partitions.

These model evaluations are then passed back to the estimator and internally we must assigne the values to each ACV estimator sample set. Specifically for each model :math:`\alpha`, we loop through all partition indices k and if :math:`R_{2\alpha,k}=1` we assign :math:`\mathcal{f_\alpha(\mathcal{Z}^\dagger_k)}` to :math:`\mathcal{Z}_{\alpha^*}` similarly if :math:`R_{2\alpha+1,k}=1` we assign :math:`\mathcal{f_\alpha(\mathcal{Z}^\dagger_k)}` to :math:`\mathcal{Z}_{\alpha}`.

Using our example, the intersection matrix is

.. math:: P = [2, 3, 4, 5]

.. math:: S=\begin{bmatrix}
   0 & 2 & 2 & 2 & 2 & 2 & 2 & 2\\
   0 & 0 & 0 & 3 & 0 & 3 & 0 & 3\\
   0 & 0 & 0 & 0 & 0 & 4 & 0 & 4\\
   0 & 0 & 0 & 0 & 0 & 5 & 0 & 0
   \end{bmatrix}

So summing each column of S we have

.. math:: N_0^*=0, N_0=2,  N_1^*=2, N_1=5,  N_2^*=2, N_2=14,  N_3^*=2, N_3=9, 

And 

.. math::

   B=\begin{bmatrix}
  0 &  0 &  0 &  0 &  0 &  0 &  0 &  0 \\
  0 &  2 &  2 &  2 &  2 &  2 &  2 &  2 \\
  0 &  2 &  2 &  2 &  2 &  2 &  2 &  2 \\
  0 &  2 &  2 &  5 &  2 &  5 &  2 &  5 \\
  0 &  2 &  2 &  2 &  2 &  2 &  2 &  2 \\
  0 &  2 &  2 &  5 &  2 & 14 &  2 &  9 \\
  0 &  2 &  2 &  2 &  2 &  2 &  2 &  2 \\
  0 &  2 &  2 &  5 &  2 &  9 &  2 &  9
   \end{bmatrix}


The first row and column are all zero because :math:`\mathcal{Z}_0^*` is always empty.

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

First we set :math:`R_{2\alpha+1,\alpha}=1` for all :math:`\alpha=0,\ldots,M`

E.g. for (0,0,1)

.. math::

   \begin{bmatrix}
  0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
  \end{bmatrix}

We then set :math:`R_{2\alpha,\gamma_\alpha}=1` for :math:`\alpha=1,\ldots,M`. For example,

.. math::

   \begin{bmatrix}
 0 & 1 & 1 & 0 & 1 & 0 & 0 & 0 \\
 0 & 0 & 0 & 1 & 0 & 0 & 1 & 0 \\
 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}

And finally because ACMF always uses all independent partitions up to including :math:`\gamma_\alpha` we set :math:`R_{i,k}=1`, for :math:`k=0,\ldots,\gamma_\alpha` and :math:`\forall i`. For example

.. math::

   \begin{bmatrix}
 0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
 0 & 0 & 0 & 1 & 0 & 1 & 1 & 1 \\
 0 & 0 & 0 & 0 & 0 & 1 & 0 & 1 \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}
"""
