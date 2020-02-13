r"""
Recursive Approximate Control Variate Monte Carlo
=================================================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_plot_approximate_control_variate_monte_carlo.py` and demonstrates that multi-level Monte Carlo and multi-fidelity Monte Carlo are both approximate control variate techniques and how this understanding can be used to improve their efficiency.


Multi-level Monte Carlo (MLMC)
------------------------------
The multi-level (MLMC) estimator based on :math:`M+1` models :math:`f_0,\ldots,f_M` ordered by decreasing fidelity (note typically MLMC literature reverses this order) is given by

.. math:: Q_0^\mathrm{ML}=\mean{f_M}+\sum_{\alpha=1}^M\mean{f_{\alpha-1}-f_\alpha}

Similarly to ACV we approximate each expectation using Monte Carlo sampling such that

.. math::  Q_{0,N,\mathcal{Z}}^\mathrm{ML}=Q_{M,\hat{\mathcal{Z}}_{M}}+\sum_{\alpha=1}^M\left(Q_{\alpha-1,\hat{\mathcal{Z}}_{\alpha-1}}-Q_{\alpha,\hat{\mathcal{Z}}_{\alpha-1}}\right),

for some sampling sets :math:`\mathcal{Z}=\cup_{\alpha=0}^M\hat{\mathcal{Z}}_{\alpha}`.

The three model MLMC estimator is

.. math:: Q_{0,\mathcal{Z}}^\mathrm{ML}=Q_{2,\hat{\mathcal{Z}_{2}}}+\left(Q_{1,\hat{\mathcal{Z}}_{1}}-Q_{2,\hat{\mathcal{Z}}_{1}}\right)+\left(Q_{0,\hat{\mathcal{Z}}_{0}}-Q_{1,\hat{\mathcal{Z}}_{0}}\right)

By rearranging terms it is clear that this is just a control variate estimator

.. math::

    Q_{0,\mathcal{Z}}^\mathrm{ML}&=Q_{0,\hat{\mathcal{Z}}_{0}} - \left(Q_{1,\hat{\mathcal{Z}}_{0}}-Q_{1,\hat{\mathcal{Z}}_{1}}\right)-\left(Q_{2,\hat{\mathcal{Z}}_{1}}-Q_{2,\hat{\mathcal{Z}}_{2}}\right)\\
   &=Q_{0,\mathcal{Z}_{0}} - \left(Q_{1,\mathcal{Z}_{1,1}}-Q_{1,\mathcal{Z}_{1,2}}\right)-\left(Q_{2,\mathcal{Z}_{2,1}}-Q_{2,\mathcal{Z}_{2,2}}\right)

where in the last line we have used the general ACV notation for sample partitioning. The control variate weights in this case are just :math:`\eta_1=\eta_2=-1$`.

The MLMC and ACV sample sets are depicted in ref:`mlmc-sample_allocation` and :ref:`acv-sample_allocation`, respectively

By inductive reasoning we get the :math:`M` model ACV version of the MLMC estimator.

.. math:: Q_{0,N,\V{r}^\mathrm{ML}}=Q_{0,\mathcal{Z}_{0,1}} +\sum_{\alpha=1}^M\eta_\alpha\left(Q_{\alpha,\mathcal{Z}_{\alpha-1,1}}-\mu_{\alpha,\mathcal{Z}_{\alpha,2}}\right)

where :math:`\eta_\alpha=-1,\forall\alpha` and :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z}_{\alpha-1,2}`, and :math:`\mu_{\alpha,\mathcal{Z}_{\alpha,2}}=Q_{\alpha,\mathcal{Z}_{\alpha,2}}`
 
.. list-table::

   * - 
       .. _mlmc-sample-allocation:

       .. figure:: ../figures/mlmc.png
          :width: 100%
          :align: center

          MLMC sampling strategy


     - 
       .. _acv-sample-allocation:

       .. figure:: ../figures/acv_is.png
          :width: 100%
          :align: center

          ACV sampling strategy


.. math::  R^2 = - \eta_1^2 \tau_{1}^2 - 2 \eta_1 \rho_{1} \tau_{1} - \eta_M^2 \frac{\tau_{M}}{\hat{r}_{M}} - \sum_{1=2}^M \frac{1}{\hat{r}_{i-1}}
                                                          \left( \eta_i^2 \tau_{i}^2 + \tau_{i-1}^2 \tau_{i-1}^2 - 2 \eta_i \eta_{i-1} \rho_{i,i-1} \tau_{i} \tau_{i-1} \right),

where  :math:`\tau_\alpha=\left(\frac{\var{Q_\alpha}}{\var{Q_0}}\right)^{\frac{1}{2}}`. Recall that and :math:`\hat{r}_\alpha=\lvert\mathcal{Z}_{\alpha,2}\rvert/N` is the ratio of the cardinality of the sets :math:`\mathcal{Z}_{\alpha,2}` and :math:`\mathcal{Z}_{0,2}`. Thus we can see that the variance reduction is bounded by the CV estimator using the lowest fidelity model with the highest correlation with :math:`f_0`

Let :math:`C_\alpha` be the cost of evaluating the function :math:`f_\alpha` at a single sample, then the total cost of the MLMC estimator is

.. math::

   C_{\mathrm{tot}}=\sum_{l=0}^M C_\alpha r_\alpha N
   
Variance of estimator is

.. math::
  
   \var{Q_0^\mathrm{ML}}=\sum_{\alpha=0}^M \var{Y_\alpha}r_\alpha N
   
Treating :math:`r_\alpha` as a continuous variable the variance of the MLMC estimator is minimized can be minimized by choosing :math:`r_\alpha` to minimize

.. math::

   \sum_{\alpha=0}^M\left(r_\alpha NC_\alpha+\lambda^2(r_\alpha N)^{-1}\var{Y_\alpha}\right)

for some Lagrange multiplier :math:`\lambda^2`. Thus for a fixed budget the minimum variance is obtained by setting

.. math::

   N_\alpha=r_\alpha N=\lambda\sqrt{\var{Y_\alpha}/C_\alpha}

If the desired MSE in the highest fidelity is :math:`2\epsilon^2` and the bias is

.. math::
   
   \left(\mean{Q_0}-\mean{Q}\right)^2=\epsilon^2
   
We must choose :math:`\lambda=\epsilon^{-2}\sum_{\alpha=0}^M \sqrt{\var{Y_\alpha}C_\alpha}` such that the total variance satisfies   :math:`\var{Q_0^\mathrm{ML}}=\epsilon^2`.
The total cost is then 

.. math:: C_\mathrm{tot}=\epsilon^{-2}\left(\sum_{\alpha=0}^M \sqrt{\var{Y_\alpha}C_\alpha}\right)^2
"""
#%%
# Lets setup the problem and compute an ACV estimate of :math:`\mean{f_0}`
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.tests.test_control_variate_monte_carlo import TunableExample
from scipy.stats import uniform

np.random.seed(1)
univariate_variables = [uniform(-1,2),uniform(-1,2)]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)
shifts= [.1,.2]
model = TunableExample(1,shifts=shifts)
model.theta1 = np.pi/2
model.theta2=np.pi
exact_integral_f0=0
cov = model.get_covariance_matrix()
print(cov)
assert False

from functools import partial
def compute_acv_many_model_variance_reduction(nhf_samples,nsample_ratios,
                                              functions):
    M = len(nsample_ratios) # number of lower fidelity models
    assert len(functions)==M+1
    
    ntrials=1000
    means = np.empty((ntrials,2))
    generate_samples=partial(
        pya.generate_independent_random_samples,variable)
    for ii in range(ntrials):
        samples,values =\
            pya.generate_samples_and_values_mlmc(
                nhf_samples,nsample_ratios,functions,generate_samples)
        # compute mean using only hf data
        hf_mean = values[0][0].mean()
        means[ii,0]= hf_mean
        # compute ACV mean
        eta = pya.get_mlmc_control_variate_weights(M+1)
        means[ii:,1] = pya.compute_control_variate_mean_estimate(
            eta,values)

    print("Theoretical ACV variance reduction",
          1-pya.get_rsquared_mlmc(cov[:M+1,:M+1],nsample_ratios))
    print("Achieved ACV variance reduction",
          means[:,1].var(axis=0)/means[:,0].var(axis=0))
    return means

print('Two models')
nsample_ratios = [10]
target_cost = int(1e3)
costs = [1,1,1]
nhf_samples,nsample_ratios = pya.allocate_samples_mlmc(
    cov[:2,:2], costs[:2], target_cost, nhf_samples_fixed=10)[:2]
print('target cost',target_cost)
print('sample_cost',np.sum(nsample_ratios*nhf_samples)+nhf_samples)
means1 = compute_acv_many_model_variance_reduction(
    10,nsample_ratios,[model.m0,model.m1])

print('Three models')
nhf_samples,nsample_ratios = pya.allocate_samples_mlmc(
    cov, costs, target_cost, nhf_samples_fixed=10)[:2]
print('target cost',target_cost)
print('sample_cost',np.sum(nsample_ratios*nhf_samples)+nhf_samples)
means2 = compute_acv_many_model_variance_reduction(
    10,nsample_ratios,[model.m0,model.m1,model.m2])
print("Theoretical CV variance reduction",
      1-max(cov[0,1],cov[0,2])**2/(cov[0,0]*cov[1,1]))

#%%
#
#Multi-fidelity Monte Carlo (MFMC)
#---------------------------------
#
#.. math::
#   
#   r_i=\left(\frac{C_1(\rho^2_{1i}-\rho^2_{1i+1})}{C_i(1-\rho^2_{12})}\right)^{\frac{1}{2}}
#   
#Let :math:`C=(C_1\cdots C_L)^T r=(r_1\cdots r_L)^T` then
#
#.. math::
#
#   N_1=\frac{C_{\mathrm{tot}}}{C^Tr} & & N_i=r_iN_1\\
#
#  
#The control variate weights are
#
#.. math::
#   
#   \alpha_i=\frac{\rho_{1i}\sigma_1}{\sigma_i}

#%%
#References
#^^^^^^^^^^
#.. [PWGSIAM2016] `B. Peherstorfer, K. Willcox, M. Gunzburger, Optimal model management for multifidelity Monte Carlo estimation, SIAM J. Sci. Comput. 38 (2016) 59 A3163–A3194. <https://doi.org/10.1137/15M1046472>`_
#
#.. [CGSTCVS2011] `K.A. Cliffe, M.B. Giles, R. Scheichl, A.L. Teckentrup, Multilevel Monte Carlo methods and applications to elliptic PDEs with random coefficients, Comput. Vis. Sci. 14 (2011) <https://doi.org/10.1007/s00791-011-0160-x>`_
#
#.. [GOR2008] `M.B. Giles, Multilevel Monte Carlo path simulation, Oper. Res. 56 (2008) 607–617. <https://doi.org/10.1287/opre.1070.0496>`_
#
#.. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification, Journal of Computational Physics, In press, (2020) <https://doi.org/10.1016/j.jcp.2020.109257>`_
