r"""
Sampling Allocation for Approximate Control Variate Monte Carlo Methods
=======================================================================
This tutorial builds upon the tutorials :ref:`sphx_glr_auto_tutorials_plot_approximate_control_variate_monte_carlo.py` and :ref:`sphx_glr_auto_tutorials_plot_recursive_control_variate_monte_carlo.py`.


Let :math:`C_\alpha` be the cost of evaluating the function :math:`f_\alpha` at a single sample, then the total cost of the MLMC estimator is

.. math::

   C_{\mathrm{tot}}=\sum_{l=0}^M C_\alpha r_\alpha N
   
Variance of estimator is

.. math::
  
   \var{Q_0^\mathrm{ML}}=\sum_{\alpha=0}^M \var{Y_\alpha}r_\alpha N
   
Let :math:`Y_\alpha` be the disrepancy between two consecutive models, e.g. :math:`f_{\alpha-1}-f_\alpha` and let :math:`N_\alpha` be the number of samples allocated to resolving the discrepancy, i.e. :math:`N_\alpha=\lvert\hat{\mathcal{Z}}_\alpha\rvert`

Then the variance of the MLMC estimator can be written as

.. math:: \var{Q_{0,\mathcal{Z}}^\mathrm{ML}}=\sum_{\alpha=0}^M N_\alpha^{-1} \var{Y_\alpha}

For a fixed variance :math:`\epsilon^2` the cost of the MLMC estimator can be minimized, by minimizing

.. math:: 

   \mathcal{J}(N_0,\ldots,N_M,\lambda)&=\sum_{\alpha=0}^M\left(N_\alpha C_\alpha\right)+\lambda^2\left(\sum_{\alpha=0}^M\left(N_\alpha^{-1}\var{Y_\alpha}\right)-\epsilon^2\right)\\
   &=\sum_{\alpha=0}^M\left(N_\alpha C_\alpha+\lambda^2N_\alpha^{-1}\var{Y_\alpha}\right)-\lambda^2\epsilon^2

for some Lagrange multiplier :math:`\lambda`. To find the minimum we set the gradient of this expression to zero:

.. math::

  \frac{\partial \mathcal{J}^\mathrm{ML}}{N_\alpha}&=C_\alpha-\lambda^2N_\alpha^{-2}\var{Y_\alpha}=0\\
  \implies C_\alpha&=\lambda^2N_\alpha^{-2}\var{Y_\alpha}\\
  \implies N_\alpha&=\lambda\sqrt{\var{Y_\alpha}C_\alpha^{-1}}

and 

.. math:: \frac{\partial \mathcal{J}}{\lambda^2}=\sum_{\alpha=0}^M N_\alpha^{-1}\var{Y_\alpha}-\epsilon^2=0

The total variance is

.. math::

  \var{Q_{0,\mathcal{Z}}^\mathrm{ML}}&=\sum_{\alpha=0}^M N_\alpha^{-1} \var{Y_\alpha}\\
  &=\sum_{\alpha=0}^M \lambda^{-1}\var{Y_\alpha}^{-\frac{1}{2}}C_\alpha^{\frac{1}{2}}\var{Y_\alpha}\\
  &=\lambda^{-1}\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}=\epsilon^2\\
  \implies \lambda &= \epsilon^{-2}\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}


Now substituting :math:`\lambda` into the following

.. math::

  N_\alpha C_\alpha&=\lambda\sqrt{\var{Y_\alpha}C_\alpha^{-1}}C_\alpha\\
  &=\lambda\sqrt{\var{Y_\alpha}C_\alpha}\\
  &=\epsilon^{-2}\left(\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}\right)\sqrt{\var{Y_\alpha}C_\alpha}



allows us to determine the total cost

.. math::

  C_\mathrm{tot}&=\sum_{\alpha=0}^M N_\alpha C_\alpha\\
  &=\sum_{\alpha=0}^M \epsilon^{-2}\left(\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}\right)\sqrt{\var{Y_\alpha}C_\alpha}\\
  &=\epsilon^{-2}\left(\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}\right)^2

"""

#from pyapprox.fenics_models import advection_diffusion, qoi_functional_misc
nmodels  = 3
num_vars = 100
max_eval_concurrency = 1
from pyapprox.examples.multi_index_advection_diffusion import *
base_model = setup_model(num_vars,max_eval_concurrency)
from pyapprox.models.wrappers import MultiLevelWrapper
multilevel_model=MultiLevelWrapper(
    base_model,base_model.base_model.num_config_vars,
    base_model.cost_function)
from scipy.stats import uniform
import pyapprox as pya
variable = pya.IndependentMultivariateRandomVariable(
    [uniform(-np.sqrt(3),2*np.sqrt(3))],[np.arange(num_vars)])

npilot_samples = 10
pilot_samples = pya.generate_independent_random_samples(
    variable,npilot_samples)
config_vars = np.arange(nmodels)[np.newaxis,:]
pilot_samples = pya.get_all_sample_combinations(pilot_samples,config_vars)
pilot_values = multilevel_model(pilot_samples)
assert pilot_values.shape[1]==1
pilot_values = np.reshape(pilot_values,(npilot_samples,nmodels))
# mlmc requires model accuracy to decrease with index
# but model assumes the opposite. so reverse order here
pilot_values = pilot_values[:,::-1]
cov = np.cov(pilot_values,rowvar=False)
print(pya.get_correlation_from_covariance(cov))
for ii in range(nmodels-1):
    vardelta = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
    print(vardelta)

target_cost = 10
# mlmc requires model accuracy to decrease with index
# but model assumes the opposite. so reverse order here
costs = [multilevel_model.cost_function(ii) for ii in range(nmodels)][::-1]
print(costs)
nhf_samples,nsample_ratios = pya.allocate_samples_mlmc(
    cov, costs, target_cost, nhf_samples_fixed=10)[:2]

import seaborn as sns
from pandas import DataFrame
df = DataFrame(
    index=np.arange(pilot_values.shape[0]),
    data=dict([(r'$f_%d$'%ii,pilot_values[:,ii])
               for ii in range(pilot_values.shape[1])]))
# heatmap does not currently work with matplotlib 3.1.1 downgrade to
# 3.1.0 using pip install matplotlib==3.1.0
#sns.heatmap(df.corr(),annot=True,fmt='.2f',linewidth=0.5)
#plt.show()
