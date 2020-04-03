r"""
Generalized Approximate Control Variate Monte Carlo
===================================================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_approximate_control_variate_monte_carlo.py`, :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_level_monte_carlo.py`, and :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multi_fidelity_monte_carlo.py`. MLMC and MFMC are two approaches which can utilize an esemble of models of vary cost and accuracy to efficiently estimate the expectation of the highest fidelity model. In this tutorial we introduce a general framework for ACVMC when using 2 or more mmodels. We show that MFMC are both instances of this framework and use the flexibility of the framework to derive new ACV estimators.

Control variate Monte Carlo can be easily extended and applied to more than two models. Consider :math:`M` lower fidelity models with sample ratios :math:`r_\alpha>=1`, for :math:`\alpha=1,\ldots,M`. The approximate control variate estimator of the mean of the high-fidelity model :math:`Q_0=\mean{f_0}` is

.. math::
   Q^{\text{ACV}} &= Q_{0,\mathcal{Z}_{0,1}} + \sum_{\alpha=1}^M \eta_\alpha \left( Q_{\alpha,\mathcal{Z}_{\alpha,1}} - \mu_{\alpha,\mathcal{Z}_{\alpha,2}} \right) =Q_{0,\mathcal{Z}_{0,1}} + \sum_{\alpha=1}^M \eta_\alpha \Delta_{\alpha,\mathcal{Z}_{\alpha,1},\mathcal{Z}_{\alpha,2}}\\&=Q_{0,N}+\V{\eta}\V{\Delta}

Here :math:`\V{\eta}=[\eta_1,\ldots,\eta_M]^T`, :math:`\V{\Delta}=[\Delta_1,\ldots,\Delta_M]^T`, and :math:`\mathcal{Z}_{\alpha,1}`, :math:`\mathcal{Z}_{\alpha,2}` are sample sets that may or may not be disjoint. Specifying the exact nature of these sets, including their cardinality, can be used to design different ACV estimators which will discuss later.

The variance of the ACV estimator is

.. math::

   \var{Q^{\text{ACV}}} = \var{Q_{0}}\left(1+\V{\eta}^T\frac{\covar{\V{\Delta}}{\V{\Delta}}}{\var{Q_0}}\V{\eta}+2\V{\eta}^T\frac{\covar{\V{\Delta}}{Q_0}}{\var{Q_0}}\right)

The control variate weights that produce the minimum variance are given by

.. math::

   \V{\eta} = -\covar{\V{\Delta}}{\V{\Delta}}^{-1}\covar{\V{\Delta}}{Q_0}

The resulting variance reduction is

.. math::

   \gamma =1-\covar{\V{\Delta}}{Q_0}^T\frac{\covar{\V{\Delta}}{\V{\Delta}}^{-1}}{\var{Q_0}}\covar{\V{\Delta}}{Q_0}

The previous formulae require evaluating covarices with the discrepancies :math:`\Delta`. To avoid this we write

.. math::

   \covar{\V{\Delta}}{Q_0}&=N^{-1}\left(\mathrm{diag}\left(F\right)\circ \covar{\V{Q}_\mathrm{LF}}{Q_0}\right)\\
   \covar{\V{\Delta}}{\V{\Delta}}&=N^{-1}\left(\covar{\V{Q}_\mathrm{LF}}{\V{Q}_\mathrm{LF}}\circ F \right)\\

where :math:`\V{Q}_\mathrm{LF}=[Q_1,\ldots,Q_M]^T` and :math:`\circ` is the Hadamard  (element-wise) product. The matrix :math:`F` is dependent on the sampling scheme used to generate the sets :math:`\mathcal{Z}_{\alpha,1}`, :math:`\mathcal{Z}_{\alpha,2}`. We discuss one useful sampling scheme found in [GGEJJCP2020]_ here.

The most straightforward way to obtain an ACV estimator with the same covariance structure of an CV estimator is to evaluate each model (including the high-fidelity model) at a set of :math:`N` samples  :math:`\mathcal{Z}_{\alpha,1}`. We then evaluate each low fidelity model at an additional :math:`N(1-r_\alpha)` samples :math:`\mathcal{Z}_{\alpha,2}`. That is the sample sets satisfy :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z}_{0}\;\forall\alpha>0` and :math:`\left(\mathcal{Z}_{\alpha,2}\setminus\mathcal{Z}_{\alpha,1}\right)\cap\left(\mathcal{Z}_{\kappa,2}\setminus\mathcal{Z}_{\kappa,1}\right)=\emptyset\;\forall\kappa\neq\alpha`. See :ref:`acv-is-sample-allocation` for a visual depiction of the sample sets.

The matrix :math:`F` corresponding to this sample scheme is

.. math::

   F_{ij}=\begin{cases}\frac{r_i-1}{r_i}\frac{r_j-1}{r_j} & i\neq j\\
   \frac{r_i-1}{r_i} & i=j
   \end{cases}
"""

#%%
#Lets apply ACV to the tunable model ensemble using some helper functions to reduce the amount of code we have to write
from functools import partial
generate_samples_and_values = pya.generate_samples_and_values_acv_IS
get_cv_weights = partial(
    pya.get_approximate_control_variate_weights,
    get_discrepancy_covariances=pya.get_discrepancy_covariances_IS)
get_rsquared = partial(
    pya.get_rsquared_acv,
    get_discrepancy_covariances=pya.get_discrepancy_covariances_IS)

#%%
# First let us just use 2 models

print('Two models')
model_ensemble = pya.ModelEnsemble(model.models[:2])
nsample_ratios = [10]
allocate_samples = \
    lambda cov, costs, target_cost : [nhf_samples, nsample_ratios, None]
means1, numerical_var_reduction1, true_var_reduction1 = \
    pya.estimate_variance_reduction(
        model_ensemble, cov[:2,:2], model.generate_samples, allocate_samples,
        generate_samples_and_values, get_cv_weights, get_rsquared, ntrials=1e3,
        max_eval_concurrency=1)
print("Theoretical ACV variance reduction",true_var_reduction1)
print("Achieved ACV variance reduction",numerical_var_reduction1)

#%%
# Now let us use 3 models

print('Three models')
model_ensemble = pya.ModelEnsemble(model.models)
nsample_ratios = [10,10]
allocate_samples = \
    lambda cov, costs, target_cost : [nhf_samples, nsample_ratios, None]
means2, numerical_var_reduction2, true_var_reduction2 = \
    pya.estimate_variance_reduction(
        model_ensemble, cov, model.generate_samples, allocate_samples,
        generate_samples_and_values, get_cv_weights, get_rsquared, ntrials=1e3,
        max_eval_concurrency=1)
print("Theoretical ACV variance reduction",true_var_reduction2)
print("Achieved ACV variance reduction",numerical_var_reduction2)

#%%
#The benefit of using three models over two models depends on the correlation between each low fidelity model and the high-fidelity model. The benefit on using more models also depends on the relative cost of evaluating each model, however here we will just investigate the effect of changing correlation. The following code shows the variance reduction (relative to standard Monte Carlo) obtained using CVMC (not approximate CVMC) using 2 (OCV1) and three models (OCV2). ACVMC will achieve these variance reductions in the limit as the number of samples of the low fidelity models goes to infinity.

theta1 = np.linspace(model.theta2*1.05,model.theta0*0.95,5)
covs = []
var_reds = []
for th1 in theta1:
    model.theta1=th1
    covs.append(model.get_covariance_matrix())
    OCV2_var_red = 1-pya.get_control_variate_rsquared(covs[-1])
    # use model with largest covariance with high fidelity model
    idx = [0,np.argmax(covs[-1][0,1:])+1]
    assert idx == [0,1] #it will always be the first model
    OCV1_var_red = pya.get_control_variate_rsquared(covs[-1][np.ix_(idx,idx)])
    var_reds.append([OCV2_var_red,OCV1_var_red])
covs = np.array(covs)
var_reds = np.array(var_reds)

fig,axs = plt.subplots(1,2,figsize=(2*8,6))
for ii,jj, in [[0,1],[0,2],[1,2]]:
    axs[0].plot(theta1,covs[:,ii,jj],'o-',
                label=r'$\rho_{%d%d}$'%(ii,jj))
axs[1].plot(theta1,var_reds[:,0],'o-',label=r'$\mathrm{OCV2}$')
axs[1].plot(theta1,var_reds[:,1],'o-',label=r'$\mathrm{OCV1}$')
axs[1].plot(theta1,var_reds[:,0]/var_reds[:,1],'o-',
            label=r'$\mathrm{OCV2/OCV1}$')
axs[0].set_xlabel(r'$\theta_1$')
axs[0].set_ylabel(r'$\mathrm{Correlation}$')
axs[1].set_xlabel(r'$\theta_1$')
axs[1].set_ylabel(r'$\mathrm{Variance reduction ratio} \ \gamma$')
axs[0].legend()
_ = axs[1].legend()

#%%
#The variance reduction clearly depends on the correlation between all the models.

#%%
#References
#^^^^^^^^^^
#.. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification, Journal of Computational Physics, In press, 2020. <https://doi.org/10.1016/j.jcp.2020.109257>`_
