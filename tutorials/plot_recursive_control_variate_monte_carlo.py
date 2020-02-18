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

where in the last line we have used the general ACV notation for sample partitioning. The control variate weights in this case are just :math:`\eta_1=\eta_2=-1`.

The MLMC and ACV sample sets are depicted in :ref:`mlmc-sample-allocation` and :ref:`acv-sample-allocation`, respectively

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

By viewing MLMC as a control variate we can derive its variance reduction GGEJJCP2020]_

.. math::  gamma+1 = - \eta_1^2 \tau_{1}^2 - 2 \eta_1 \rho_{1} \tau_{1} - \eta_M^2 \frac{\tau_{M}}{\hat{r}_{M}} - \sum_{1=2}^M \frac{1}{\hat{r}_{i-1}}
                                                          \left( \eta_i^2 \tau_{i}^2 + \tau_{i-1}^2 \tau_{i-1}^2 - 2 \eta_i \eta_{i-1} \rho_{i,i-1} \tau_{i} \tau_{i-1} \right),

where  :math:`\tau_\alpha=\left(\frac{\var{Q_\alpha}}{\var{Q_0}}\right)^{\frac{1}{2}}`. Recall that and :math:`\hat{r}_\alpha=\lvert\mathcal{Z}_{\alpha,2}\rvert/N` is the ratio of the cardinality of the sets :math:`\mathcal{Z}_{\alpha,2}` and :math:`\mathcal{Z}_{0,2}`. 

From the above expression we can see that the variance reduction is bounded by the CV estimator using the lowest fidelity model with the highest correlation with :math:`f_0`. Using multiple models only helps increase the speed to which we converge to the 2 model CV  estimator. The following demonstrates this numerically. Try the different models (commented out) to see how peformance changes.
"""
#%%
# Lets setup the problem and compute an ACV estimate of :math:`\mean{f_0}`
import pyapprox as pya
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.tests.test_control_variate_monte_carlo import \
    TunableModelEnsemble, ShortColumnModelEnsemble
from scipy.stats import uniform
from functools import partial
from scipy.stats import uniform,norm,lognorm
np.random.seed(1)

# univariate_variables = [uniform(-1,2),uniform(-1,2)]
# variable = pya.IndependentMultivariateRandomVariable(univariate_variables)
# shifts= [.1,.2]
# tunable_model = TunableModelEnsemble(1,shifts=shifts)
# model_ensemble = pya.ModelEnsemble(
#     [tunable_model.m0,tunable_model.m1,tunable_model.m2])
# exact_integral_f0=0
# cov = tunable_model.get_covariance_matrix()
# costs = [100,10,1]

short_column_model = ShortColumnModelEnsemble()
model_ensemble = pya.ModelEnsemble(
    [short_column_model.m0,short_column_model.m1,short_column_model.m2])
costs = np.asarray([100, 50, 5])
# MLMC will perform poorly with the following combination of models
#model_ensemble = pya.ModelEnsemble(
#    [short_column_model.m0,short_column_model.m3,short_column_model.m4])
#costs = np.asarray([100, 20, 10])


univariate_variables = [
   uniform(5,10),uniform(15,10),norm(500,100),norm(2000,400),
   lognorm(s=0.5,scale=np.exp(5))]
variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

# generate pilot samples to estimate correlation
npilot_samples = int(1e4)
generate_samples=partial(
    pya.generate_independent_random_samples,variable)
cov = pya.estimate_model_ensemble_covariance(
    npilot_samples,generate_samples,model_ensemble)

def compute_mlmc_many_model_variance_reduction(nhf_samples,nsample_ratios,
                                              functions):
    M = len(nsample_ratios) # number of lower fidelity models
    if not callable(functions):
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

    print("Theoretical MLMC variance reduction",
          1-pya.get_rsquared_mlmc(cov[:M+1,:M+1],nsample_ratios))
    print("Achieved MLMC variance reduction",
          means[:,1].var(axis=0)/means[:,0].var(axis=0))
    return means

print('Two models')
nsample_ratios = [10]
target_cost = int(1e4)
nhf_samples,nsample_ratios = pya.allocate_samples_mlmc(
    cov[:2,:2], costs[:2], target_cost, nhf_samples_fixed=10)[:2]
means1 = compute_mlmc_many_model_variance_reduction(
    10,nsample_ratios,model_ensemble)
idx = np.argmax([max(cov[0,1],cov[0,2])])+1
print("Theoretical two model CV variance reduction",
      1-cov[0,idx]**2/(cov[0,0]*cov[idx,idx]))

#%%
#It is also worth empahsizing the MLMC only works when the variance between the model discrepancies decay. One needs to be careful that the variance between discrepancies does indeed decay. Sometimes when the models correspond to different mesh discretizations. Increasing the mesh resolution does not always produce a smaller discrepancy. The following example shows that, for this example, adding a third model actually increases the variance of the MLMC estimator. 

print('Three models')
nhf_samples,nsample_ratios = pya.allocate_samples_mlmc(
    cov, costs, target_cost, nhf_samples_fixed=10)[:2]
means2 = compute_mlmc_many_model_variance_reduction(
    10,nsample_ratios,model_ensemble)

#%%
#
#Multi-fidelity Monte Carlo (MFMC)
#---------------------------------
#
#Recall the two model ACV estimator
#
#.. math:: Q_{0,\mathcal{Z}}^\mathrm{MF}=Q_{0,\mathcal{Z}_{0}} + \eta\left(Q_{1,\mathcal{Z}_{0}}-\mu_{1,\mathcal{Z}_{1}}\right)
#
#The MFMC estimator can be derived with the following recursive argument. Partition the samples assigned to each model such that
#:math:`\mathcal{Z}_\alpha=\mathcal{Z}_{\alpha,1}\cup\mathcal{Z}_{\alpha,2}` and :math:`\mathcal{Z}_{\alpha,1}\cap\mathcal{Z}_{\alpha,2}=\emptyset`. That is the samples at the next lowest fidelity model are the samples used at all previous levels plus an additional independent set, i.e. :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z}_{\alpha-1}`. See :ref:`mfmc-sample-allocation`
#
#We then introduced the next low fidelity model to reduce the variance of the estimate :math:`\mu_{\alpha}`
#
#.. math::
#
#   Q_{0,\mathcal{Z}}^\mathrm{MF}&=Q_{0,\mathcal{Z}_{0}} + \eta_1\left(Q_{1,\mathcal{Z}_{1}}-\left(\mu_{1,\mathcal{Z}_{1}}+\eta_2\left(Q_{2,\mathcal{Z}_1}-\mu_{2,\mathcal{Z}_2}\right)\right)\right)\\
#   &=Q_{0,\mathcal{Z}_{0}} + \eta_1\left(Q_{1,\mathcal{Z}_{1}}-\mu_{1,\mathcal{Z}_{1}}\right)+\eta_1\eta_2\left(Q_{2,\mathcal{Z}_1}-\mu_{2,\mathcal{Z}_2}\right)\\
#
#We repeat this process for all low fidelity models to obtain
#
#.. math:: Q_{0,\mathcal{Z}}^\mathrm{MF}=Q_{0,\mathcal{Z}_{0}} + \sum_{\alpha=1}^M\eta_\alpha\left(Q_{\alpha,\mathcal{Z}_{\alpha,1}}-\mu_{\alpha,\mathcal{Z}_{\alpha}}\right)
#
#.. list-table::
#
#   * - 
#       .. _mfmc-sample-allocation:
#
#       .. figure:: ../figures/mfmc.png
#          :width: 100%
#          :align: center
#
#          MFMC sampling strategy
#
#
#The optimal weights for the MFMC estimator are
#
#.. math:: \eta_\alpha = -\frac{\covar{Q_0}{Q_\alpha}}{\var{Q_\alpha}}
#
#which result in the variance reduction
#
#.. math:: \gamma = 1-\rho_1^2\left(\frac{r_1-1}{r_1}+\sum_{\alpha=2}^M \frac{r_\alpha-r_{\alpha-1}}{r_\alpha r_{\alpha-1}}\frac{\rho_\alpha^2}{\rho_1^2}\right)
#
#Similarly to MLMC Using multiple models only helps increase the speed to which we converge to the 2 model CV estimator

def compute_mfmc_many_model_variance_reduction(nhf_samples,nsample_ratios,
                                              functions):
    M = len(nsample_ratios) # number of lower fidelity models
    if not callable(functions):
        assert len(functions)==M+1

    ntrials=int(1e3)
    means = np.empty((ntrials,2))
    generate_samples=partial(
        pya.generate_independent_random_samples,variable)
    for ii in range(ntrials):
        samples,values =\
            pya.generate_samples_and_values_mfmc(
                nhf_samples,nsample_ratios,functions,generate_samples)
        # compute mean using only hf data
        hf_mean = values[0][0].mean()
        means[ii,0]= hf_mean
        # compute ACV mean
        eta = pya.get_mfmc_control_variate_weights(cov)
        means[ii:,1] = pya.compute_control_variate_mean_estimate(
            eta,values)

    print("Theoretical MFMC variance reduction",
          1-pya.get_rsquared_mfmc(cov[:M+1,:M+1],nsample_ratios))
    print("Achieved MFMC variance reduction",
          means[:,1].var(axis=0)/means[:,0].var(axis=0))
    return means

print('Two models')
nhf_samples,nsample_ratios = pya.allocate_samples_mfmc(
    cov[:2,:2], costs[:2], target_cost, nhf_samples_fixed=None)[:2]
means1 = compute_mfmc_many_model_variance_reduction(
    nhf_samples,nsample_ratios,model_ensemble)
print("Theoretical two model CV variance reduction",
      1-cov[0,1]**2/(cov[0,0]*cov[1,1]))

print('Three models')
nhf_samples,nsample_ratios = pya.allocate_samples_mfmc(
    cov, costs, target_cost, nhf_samples_fixed=None)[:2]
print(nhf_samples,nsample_ratios*nhf_samples)
functions = model_ensemble
#functions = [model_ensemble.m0,model_ensemble.m1,model_ensemble.m2]
#functions = [model_ensemble.m0,model_ensemble.m3,model_ensemble.m4]
means2 = compute_mfmc_many_model_variance_reduction(
    nhf_samples,nsample_ratios,functions)

#%%
#The optimal number of samples that minimize the variance of the MFMC estimator can be determined analytically. Let :math:`C_\mathrm{tot}` be the total budget then the optimal number of high fidelity samples is
#
#.. math:: N_0 = \frac{C_\mathrm{tot}}{\V{w}^T\V{r}}
#
#where :math:`\V{r}=[r_0,\ldots,r_M]^T` are the sample ratios defining the number of samples assigned to each level, i.e. :math:`N_\alpha=r_\alpha N_0`. The sample ratios are
#
#.. math::
#   
#   r_\alpha=\left(\frac{w_0(\rho^2_{0,\alpha}-\rho^2_{0,\alpha+1})}{w_\alpha(1-\rho^2_{0,1})}\right)^{\frac{1}{2}}
#
#where :math:`\V{w}=[w_0,w_M]^T` are the relative costs of each model, and :math:`\rho_{j,k}` is the correlation between models :math:`j` and :math:`k`.
#

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
