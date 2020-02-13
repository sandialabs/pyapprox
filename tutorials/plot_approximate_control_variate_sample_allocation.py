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

.. math:: \var{Q_{0,\mathcal{Z}}^\mathrm{ML}}=\sum_{\alpha=0}^M\left(N_\alpha C_\alpha+\lambda^2N_\alpha^{-1}\var{Y_\alpha}\right)

for some Lagrange multiplier :math:`\lambda`. To find the minimum we set the gradient of this expression to zero:

.. math::

  \frac{\partial \var{Q_{0,\mathcal{Z}}^\mathrm{ML}}}{N_\alpha}&=C_\alpha-\lambda^2N_\alpha^{-2}\var{Y_\alpha}=0\\
  \implies C_\alpha&=\lambda^2N_\alpha^{-2}\var{Y_\alpha}\\
  \implies N_\alpha&=\lambda\sqrt{\var{Y_\alpha}C_\alpha^{-1}}


The total variance is

.. math::

  \var{Q_{0,\mathcal{Z}}^\mathrm{ML}}&=\sum_{\alpha=0}^M N_\alpha^{-1} \var{Y_\alpha}\\
  &=\sum_{\alpha=0}^M \lambda^{-1}\var{Y_\alpha}^{-\frac{1}{2}}C_\alpha^{\frac{1}{2}}\var{Y_\alpha}\\
  &=\lambda^{-1}\sum_{\alpha=0}^M\sqrt{\var{Y_\alpha}C_\alpha}=\epsilon^2\\
  \implies \lambda &= \epsilon^{-2}\sum_{\kappa=0}^M\sqrt{\var{Y_\kappa}C_\kappa}


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
num_vars = 2
max_eval_concurrency = 2
from pyapprox.examples.multi_index_advection_diffusion import *
base_model = setup_model(num_vars,max_eval_concurrency)
base_model.cost_function = WorkTracker()
from pyapprox.models.wrappers import MultiLevelWrapper
multilevel_model=MultiLevelWrapper(
    base_model,base_model.base_model.num_config_vars,
    base_model.cost_function)
from scipy.stats import uniform
import pyapprox as pya
variable = pya.IndependentMultivariateRandomVariable(
    [uniform(-1,2)],[np.arange(num_vars)])

npilot_samples = 10
pilot_samples = pya.generate_independent_random_samples(
    variable,npilot_samples)
config_vars = np.arange(nmodels)[np.newaxis,:]
pilot_samples = pya.get_all_sample_combinations(pilot_samples,config_vars)
print(pilot_samples.T,pilot_samples.shape)
pilot_values = multilevel_model(pilot_samples)
print(pilot_values.shape)
assert pilot_values.shape==1
pilot_values = np.reshape(pilot_values,(npilot_samples,nmodels))
cov = np.cov(pilot_values,rowvar=False)
print(cov.shape)
print(pilot_samples.shape)
