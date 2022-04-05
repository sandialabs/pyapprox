r"""
Sensitivity Analysis
====================
Quantifying the sensitivity of a model output :math:`f` to the model parameters :math:`\rv` can be an important component of any modeling exercise. This section demonstrates how to use popular local and global sensitivity analysis.

Sobol Indices
-------------
Any function :math:`f` with finite variance parameterized by a set of independent variables :math:`\rv` with :math:`\pdf(\rv)=\prod_{j=1}^d\pdf(\rv_j)` and support :math:`\rvdom=\bigotimes_{j=1}^d\rvdom_j` can be decomposed into a finite sum, referred to as the ANOVA decomposition,

.. math::  f(\rv) = \hat{f}_0 + \sum_{i=1}^d \hat{f}_i(\rv_i)+ \sum_{i,j=1}^d \hat{f}_{i,j}(\rv_i,\rv_j) + \cdots + \hat{f}_{1,\ldots,d}(\rv_1,\ldots,\rv_d)

or more compactly

.. math:: f(\rv)=\sum_{\V{u}\subseteq\mathcal{D}}\hat{f}_{\V{u}}(\rv_{\V{u}})

where :math:`\hat{f}_\V{u}` quantifies the dependence of the function :math:`f` on the variable dimensions :math:`i\in\V{u}` and :math:`\V{u}=(u_1,\ldots,u_s)\subseteq\mathcal{D}=\{1,\ldots,d\}`.

The functions :math:`\hat{f}_\V{u}` can be obtained by integration, specifically

.. math:: \hat{f}_\V{u}(\rv_\V{u}) = \int_{\rvdom_{\mathcal{D}\setminus\V{u}}}f(\rv)\dx{\pdf_{\mathcal{D} \setminus \V{u}}(\rv)}-\sum_{\V{v}\subset\V{u}}\hat{f}_\V{v}(\rv_\V{v}),

where :math:`\dx{\pdf_{\mathcal{D} \setminus \V{u}}(\rv)}=\prod_{j\notin\V{u}}\dx{\pdf_j(\rv)}` and :math:`\rvdom_{\mathcal{D} \setminus \V{u}}=\bigotimes_{j\notin\V{u}}\rvdom_j`.

The first-order terms :math:`\hat{f}_{\V{u}}(\rv_i)`, :math:`\lVert \V{u}\rVert_{0}=1` represent the effect of a single variable acting independently of all others. Similarly, the second-order terms :math:`\lVert\V{u}\rVert_{0}=2` represent the contributions of two variables acting together, and so on.

 The terms of the ANOVA expansion are orthogonal, i.e. the weighted :math:`L^2` inner product :math:`(\hat{f}_\V{u},\hat{f}_\V{v})_{L^2_\pdf}=0`, for :math:`\V{u}\neq\V{v}`. This orthogonality facilitates the following decomposition of the variance of the function :math:`f` 

.. math:: \var{f}=\sum_{\V{u}\subseteq\mathcal{D}}\var{\hat{f}_\V{u}}, \qquad \var{\hat{f}_\V{u}} = \int_{\rvdom_{\V{u}}} f^2_{\V{u}} \dx{\pdf_\V{u}},

where :math:`\dx{\pdf_{\V{u}}(\rv)}=\prod_{j\in\V{u}}\dx{\pdf_j(\rv)}`.

The quantities :math:`\var{\hat{f}_\V{u}}/ \var{f}` are referred to as Sobol indices [SMCS2001]_ and are frequently used to estimate the sensitivity of :math:`f` to single, or combinations of input parameters. Note that this is a *global* sensitivity, reflecting a variance attribution over the range of the input parameters, as opposed to the local sensitivity reflected by a derivative. Two popular measures of sensitivity are the main effect and total effect indices given respectively by

.. math:: S_{i} = \frac{\var{\hat{f}_{\V{e}_i}}}{\var{f}}, \qquad S^T_{i} = \frac{\sum_{\V{u}\in\mathcal{J}}\var{\hat{f}_{\V{u}}}}{\var{f}}

where :math:`\V{e}_i` is the unit vector, with only one non-zero entry located at the :math:`i`-th element, and :math:`\mathcal{J} = \{\V{u}:i\in\V{u}\}`.

Sobol indices can be computed different ways. In the following we will use polynomial chaos expansions, as in [SRESS2008]_.
"""
import matplotlib.pyplot as plt
import numpy as np
import pyapprox as pya
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.approx.approximate import approximate
benchmark = setup_benchmark("ishigami", a=7, b=0.1)

num_samples = 1000
train_samples = pya.generate_independent_random_samples(
    benchmark.variable, num_samples)
train_vals = benchmark.fun(train_samples)

approx_res = approximate(
    train_samples, train_vals, 'polynomial_chaos',
    {'basis_type': 'hyperbolic_cross', 'variable': benchmark.variable,
     'options': {'max_degree': 8}})
pce = approx_res.approx

res = pya.analyze_sensitivity_polynomial_chaos(pce)

#%%
#Now lets compare the estimated values with the exact value
print(res.main_effects[:, 0])
print(benchmark.main_effects[:, 0])

#%%
#We can visualize the sensitivity indices using the following

fig, axs = plt.subplots(1, 3, figsize=(3*8, 6))
pya.plot_main_effects(benchmark.main_effects, axs[0])
pya.plot_total_effects(benchmark.total_effects, axs[1])
pya.plot_interaction_values(benchmark.sobol_indices,
                            benchmark.sobol_interaction_indices, axs[2])
axs[0].set_title(r'$\mathrm{Main\;Effects}$')
axs[1].set_title(r'$\mathrm{Total\;Effects}$')
axs[2].set_title(r'$\mathrm{Sobol\;Indices}$')
plt.show()


#%%
#..
# Morris One-at-a-time
# --------------------
# [MT1991]_


#%%
#References
#^^^^^^^^^^
#.. [SMCS2001] `I.M. Sobol. Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. Mathematics and Computers in Simulation, 55(3): 271-280, 2001. <https://doi.org/10.1016/S0378-4754(00)00270-6>`_
#.. [SRESS2008] `B. Sudret. Global sensitivity analysis using polynomial chaos expansions. Reliability Engineering & System Safety, 93(7): 964-979, 2008. <https://doi.org/10.1016/j.ress.2007.04.002>`_
#..
#.. [MT1991]  `M.D. Morris. Factorial Sampling Plans for Preliminary Computational Experiments, Technometrics, 33:2, 161-174, 1991 <https://doi.org/10.1080/00401706.1991.10484804>`_
