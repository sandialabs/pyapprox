r"""
Parametrically Defined Approximate Control Variates 
===================================================
MLMC and MFMC are just two possible ACV estimators derived from two different allocation matrices. Numerous other ACV estimators can be constructed by utilizing different allocation matrices. For example, [GGEJJCP2020]_ introduced the ACVMF and the ACVIS estimators with the allocation matrices, for three models, given respectively by

.. math::

  \mat{A}_\text{ACVMF}=\begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}, \qquad \mat{A}_\text{ACVIS}=\begin{bmatrix}
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
   0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
   \end{bmatrix}, \qquad 

These were shown to outperform MLMC and MFMC for certain problems, however none of these estimators is optimal. Consequently, [BLWLJCP2022] formulated a large class of so called parameterically defined ACV (PACV)estimators that can be enumerated and used to choose the best allocation matrix for a given problem. This advanced was based on the observation that MFMC uses model m as a contol variate to help estimate the statistic of model :math:`m-1`.

Generalized Multi-fidelity (GMF) Estimators
-------------------------------------------
To date 3 classes of PACV estimators have been defined. The first is based on the MLMC allocation matrix presented in :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_level_monte_carlo.py`. A specific instance of these so called generalized recursive difference (GRD) estimators can be obtained by specifying a recusion index :math:`\gamma=[\gamma_1, \ldots, \gamma_M]^\top`. The recursion index defines a zero-rooted directed acyclic graph (DAG) that

Generalized Recursive Difference (GRD) Estimators
-------------------------------------------------
"""


#%%
#In this example ACVGMFB is the most efficient estimator, i.e. it has a smaller variance for a fixed cost. However this improvement is problem dependent. For other model ensembles another estimator may be more efficient. Modify the above example to use another model to explore this more. The left plot shows the relative costs of evaluating each model using the ACVMF sampling strategy. Compare this to the MLMC sample allocation. Also edit above code to plot the MFMC sample allocation.

#%%
#Before this tutorial ends it is worth noting that a section of the MLMC literature explores adaptive methods which do not assume there is a fixed high-fidelity model but rather attempt to balance the estimator variance with the deterministic bias. These methods add a higher-fidelity model, e.g. a finer finite element mesh, when the variance is made smaller than the bias. We will not explore this here, but an example of this is shown in the tutorial on multi-index collocation.

#%%
#References
#^^^^^^^^^^
#.. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification, Journal of Computational Physics, 408:109257, 2020. <https://doi.org/10.1016/j.jcp.2020.109257>`_
#
#.. [BLWLJCP2022] `On the optimization of approximate control variates with parametrically defined estimators, Journal of Computational Physics,451:110882, 2022 <https://doi.org/10.1016/j.jcp.2021.110882>`_

#%%
#Accelerated Approximate Control Variate Monte Carlo
#---------------------------------------------------
#The recursive estimators work well when the number of low-fidelity samples are smal but ACV can achieve a greater variance reduction for a fixed number of high-fidelity samples. In this section we present an approach called ACV-GMFB that combines the strengths of these methods [BLWLJCP2022]_.
#
#This estimator differs from the previous recursive estimators because it uses some models as control variates and other models to estimate the mean of these control variates recursively. This estimator optimizes over the best use of models and returns the best model configuration.
#
#Let us add the ACV-GMFB estimator to the previous plot

#%%
#As the theory suggests MLMC and MFMC use multiple models to increase the speed to which we converge to the optimal 2 model CV estimator OCV-2. These two approaches reduce the variance of the estimator more quickly than the ACV estimator, but cannot obtain the optimal variance reduction.

#%%
#The benefit of using three models over two models depends on the correlation between each low fidelity model and the high-fidelity model. The benefit on using more models also depends on the relative cost of evaluating each model, however here we will just investigate the effect of changing correlation. The following code shows the variance reduction (relative to standard Monte Carlo) obtained using CVMC (not approximate CVMC) using 2 (OCV1) and three models (OCV2). Unlike MLMC and MFMC, ACV-IS will achieve these variance reductions in the limit as the number of samples of the low fidelity models goes to infinity.
