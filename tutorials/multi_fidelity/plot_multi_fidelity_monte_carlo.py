r"""
Multi-fidelity Monte Carlo
--------------------------
This tutorial builds on from :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_level_monte_carlo.py` and :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py` and introduces an approximate control variate estimator called Multi-fidelity Monte Carlo (MFMC). Unlike MLMC this method does not assume a strict ordering of models.

To derive the MFMC estimator first recall the two model ACV estimator

.. math:: Q_{0,\mathcal{Z}}^\mathrm{MF}=Q_{0,\mathcal{Z}_{0}} + \eta\left(Q_{1,\mathcal{Z}_{0}}-\mu_{1,\mathcal{Z}_{1}}\right)

The MFMC estimator can be derived with the following recursive argument. Partition the samples assigned to each model such that
:math:`\mathcal{Z}_\alpha=\mathcal{Z}_{\alpha,1}\cup\mathcal{Z}_{\alpha,2}` and :math:`\mathcal{Z}_{\alpha,1}\cap\mathcal{Z}_{\alpha,2}=\emptyset`. That is the samples at the next lowest fidelity model are the samples used at all previous levels plus an additional independent set, i.e. :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z}_{\alpha-1}`. See :ref:`mfmc-sample-allocation`. Note the differences between this scheme and the MLMC scheme.

.. list-table::

   * - 
       .. _mfmc-sample-allocation:

       .. figure:: ../../figures/mfmc.png
          :width: 100%
          :align: center

          MFMC sampling strategy

     -
       .. _mlmc-sample-allocation-mfmc-comparison:

       .. figure:: ../../figures/mlmc.png
          :width: 100%
          :align: center

          MLMC sampling strategy

Starting from two models we introduce the next low fidelity model in a way that reduces the variance of the estimate :math:`\mu_{\alpha}`, i.e.

.. math::

   Q_{0,\mathcal{Z}}^\mathrm{MF}&=Q_{0,\mathcal{Z}_{0}} + \eta_1\left(Q_{1,\mathcal{Z}_{1}}-\left(\mu_{1,\mathcal{Z}_{1}}+\eta_2\left(Q_{2,\mathcal{Z}_1}-\mu_{2,\mathcal{Z}_2}\right)\right)\right)\\
   &=Q_{0,\mathcal{Z}_{0}} + \eta_1\left(Q_{1,\mathcal{Z}_{1}}-\mu_{1,\mathcal{Z}_{1}}\right)+\eta_1\eta_2\left(Q_{2,\mathcal{Z}_1}-\mu_{2,\mathcal{Z}_2}\right)\\

We repeat this process for all low fidelity models to obtain

.. math:: Q_{0,\mathcal{Z}}^\mathrm{MF}=Q_{0,\mathcal{Z}_{0}} + \sum_{\alpha=1}^M\eta_\alpha\left(Q_{\alpha,\mathcal{Z}_{\alpha,1}}-\mu_{\alpha,\mathcal{Z}_{\alpha}}\right)

 The optimal control variate weights for the MFMC estimator, which minimize the variance of the estimator, are :math:`\eta=(\eta_1,\ldots,\eta_M)^T`, where for :math:`\alpha=1\ldots,M`

.. math:: \eta_\alpha = -\frac{\covar{Q_0}{Q_\alpha}}{\var{Q_\alpha}}

With this choice of weights the variance reduction obtained is given by

.. math:: \gamma = 1-\rho_1^2\left(\frac{r_1-1}{r_1}+\sum_{\alpha=2}^M \frac{r_\alpha-r_{\alpha-1}}{r_\alpha r_{\alpha-1}}\frac{\rho_\alpha^2}{\rho_1^2}\right)

Let us use MFMC to estimate the mean of our high-fidelity model.
"""

# define the sample allocation
nhf_samples,nsample_ratios = pya.allocate_samples_mfmc(
    cov, costs, target_cost)[:2]
# generate sample sets
samples,values =pya.generate_samples_and_values_mfmc(
    nhf_samples,nsample_ratios,model_ensemble,
    short_column_model.generate_samples)
# compute mean using only hf data
hf_mean = values[0][0].mean()
# compute mlmc control variate weights
eta = pya.get_mfmc_control_variate_weights(cov)
# compute MLMC mean
mfmc_mean = pya.compute_approximate_control_variate_mean_estimate(eta,values)

# get the true mean of the high-fidelity model
true_mean = short_column_model.get_means()[0]
print('MLMC error',abs(mfmc_mean-true_mean))
print('MC error',abs(hf_mean-true_mean))

#%%
#Similarly to MLMC, using multiple models with MFMC only helps increase the speed to which the variance of the MFMC estimator converges to that of the 2 model CVMC estimator that uses the low-fidelity model that has the highest correlation with the high-fidelity model. In this and following tutorials we refer to the optimal CV estimator using K low-fidelity models with known means (in this case 1) as OCV-K.
#
#Let us compare the variance reduction obtained by MLMC, MFMC and ACV with the MF sampling scheme as we increase the number of samples assigned to the low-fidelity models, while keeping the number of high-fidelity samples fixed. Here we will use the model ensemble
#
#.. math:: f_\alpha(\rv)=\rv^{5-\alpha}, \quad \alpha=0,\ldots,4
#
#where each model is the function of a single uniform random variable defined on the unit interval :math:`[0,1]`.

poly_model = PolynomialModelEnsemble()
cov = poly_model.get_covariance_matrix()
nhf_samples = 10
nsample_ratios_base = [2, 4, 8, 16]
cv_labels = [r'$\mathrm{OCV-1}$',r'$\mathrm{OCV-2}$',r'$\mathrm{OCV-4}$']
cv_rsquared_funcs=[
    lambda cov: pya.get_control_variate_rsquared(cov[:2,:2]),
    lambda cov: pya.get_control_variate_rsquared(cov[:3,:3]),
    lambda cov: pya.get_control_variate_rsquared(cov)]
cv_gammas = [1-f(cov) for f in cv_rsquared_funcs]
for ii in range(len(cv_gammas)):
    plt.axhline(y=cv_gammas[ii],linestyle='--',c='k')
    xloc = -.35
    plt.text(xloc, cv_gammas[ii]*1.1, cv_labels[ii],fontsize=16)
plt.axhline(y=1,linestyle='--',c='k')
plt.text(xloc,1,r'$\mathrm{MC}$',fontsize=16)

acv_labels = [r'$\mathrm{MLMC}$',r'$\mathrm{MFMC}$',r'$\mathrm{ACV}$-$\mathrm{MF}$']
acv_rsquared_funcs = [
    pya.get_rsquared_mlmc,pya.get_rsquared_mfmc,
    partial(pya.get_rsquared_acv,
            get_discrepancy_covariances=pya.get_discrepancy_covariances_MF)]

nplot_points = 20
acv_gammas = np.empty((nplot_points,len(acv_rsquared_funcs)))
for ii in range(nplot_points):
    nsample_ratios = [r*(2**ii) for r in nsample_ratios_base]
    acv_gammas[ii,:] = [1-f(cov,nsample_ratios) for f in acv_rsquared_funcs]
for ii in range(len(acv_labels)):
    plt.semilogy(np.arange(nplot_points),acv_gammas[:,ii],label=acv_labels[ii])
plt.legend()
plt.xlabel(r'$\log_2(r_i)-i$')
_ = plt.ylabel(r'$\mathrm{Variance}$ $\mathrm{reduction}$ $\mathrm{ratio}$ $\gamma$')

#%%
#As the theory suggests MLMC and MFMC use multiple models to increase the speed to which we converge to the optimal 2 model CV estimator OCV-2. These two approaches reduce the variance of the estimator more quickly than the ACV estimator, but cannot obtain the optimal variance reduction. 
#
#Before moving on, note that typically we want to choose both the number of low fidelity samples and the number of high-fidelity samples to minimize variance for a fixed budget. We will cover this in the next tutorial entitled :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_sample_allocation.py`.
