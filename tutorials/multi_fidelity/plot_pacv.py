r"""
Parameterized Approximate Control Variate Monte Carlo
=====================================================

Allocation matrices
-------------------
The implementation of PACV uses allocation matrices which encode how independent sample sets are shared among the sample sets :math:`\mathcal{Z}_\alpha\text{ and }\mathcal{Z}_\alpha^*`.

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

The reordered allocation matrix can be used to determine number of points in the intersection of the sets  :math:`\mathcal{Z}_\alpha\text{ and }\mathcal{Z}_\beta`, where :math:`\alpha\text{ and }\beta` may be unstarred or starred. The number of intersection points can be represented by a :math:`2(M+1)\times 2(M+1)` matrix

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

The optimization of ACV estimators returns the number of evaluations of each model N. To evaluate the estimator we have to generate independent sample partitions that :math:`N_\alpha\cup N_\alpha^*=N` for each model index :math:`\alpha`.

Using our example, the intersection matrix is

.. math:: P = [2, 3, 4, 5]

.. math:: S=\begin{bmatrix}
   0 & 2 & 2 & 2 & 2 & 2 & 2 & 2\\
   0 & 0 & 0 & 3 & 0 & 3 & 0 & 3\\
   0 & 0 & 0 & 0 & 0 & 4 & 0 & 4\\
   0 & 0 & 0 & 0 & 0 & 5 & 0 & 0
   \end{bmatrix}

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
"""
