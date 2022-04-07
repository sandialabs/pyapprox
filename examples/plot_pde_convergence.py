r"""
Convergence studies
-------------------
This tutorial demonstrates how to investigate the convergence of parameterized numerical approximations, for example tensor product quadrature or numerical models used to solve partial differential equations with unknown inputs.

First lets define a Integrator class which can be used to integrate multivariate functions with tensor product quadrature, that is compute

.. math:: I(\rv) = \int_D f(x, \rv)dx
"""
import numpy as np
from pyapprox.analysis.convergence_studies import \
    run_convergence_study, plot_convergence_data
from pyapprox.util.configure_plots import plt
from pyapprox.surrogates import (
    get_tensor_product_piecewise_polynomial_quadrature_rule,
)
from pyapprox.interface import (
    evaluate_1darray_function_on_2d_array, WorkTrackingModel,
    TimerModelWrapper
)
from scipy import stats
from pyapprox.variables import (
    IndependentMarginalsVariable, ConfigureVariableTransformation
)


class Integrator(object):
    def __init__(self, integrand):
        self.integrand = integrand

    def set_quad_rule(self, nsamples_1d):
        self.xquad, self.wquad = \
            get_tensor_product_piecewise_polynomial_quadrature_rule(
                nsamples_1d, [0, 1, 0, 1], degree=1)

    def integrate(self, sample):
        self.set_quad_rule(sample[-2:].astype(int))
        val = self.integrand(sample[:-2], self.xquad)[:, 0].dot(self.wquad)
        return val

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(
            self.integrate, samples, None)

    @staticmethod
    def get_num_degrees_of_freedom(config_sample):
        return np.prod(config_sample)

#%%
#To assess convergence we will use the function `run_convergence_study`. This routine requires a function that takes samples that consist of realizations of the random variables concatenated with any configuration variables which define the numerical resolution of the quadrature rule, in this case the number of quadrature points used in the first and second dimension.
#
#To demonstrate its usage lets integrate the function
#
#.. math:: f(x, \rv)=\rv_1(x_1^2+x_2^2)
#
#were :math:`z` is a uniform variable on :math:`[0, 1]`.
#Now define this integrand and the true value as a function of the random samples.


variable = IndependentMarginalsVariable([stats.uniform(0, 1)])


def integrand(sample, x):
    return np.sum(sample[0]*x**2, axis=0)[:, None]


def true_value(samples):
    return 2/3*samples.T

#%%
#We must also define the permissible values of the configuration variables that define the number of points :math:`n_1, n_2` in the quadrature rule. Here set :math:`n_1=2^{j+1}+1` and :math:`n_2=2^{k+1}+1` where :math:`j,k=0\ldots,9`. Now construct a `ConfigureVariableTransformation` that can map :math:`j,k` to  :math:`n_1, n_2` and back. 


config_values = [2**np.arange(1, 11)+1, 2**np.arange(1, 11)+1]
config_var_trans = ConfigureVariableTransformation(config_values)

#%%
#We can then define the values of j and k we wish to use to assess convergence. `validation_levels` :math:`v_1,v_2` specifies the values used to compute a reference solution if an exact solution is not known. `coarsest_levels` specifies the mininimum values :math:`c_1,c_2` of j, k to be used to integrate. Integrator will be used to integrate the integrand for all combinatinos of j,k in the tensor product of :math:`\{c_1,\ldots v_1-1\},` and :math:`\{c_2,\ldots v_2-1\}`.

validation_levels = [5, 5]
coarsest_levels = [0, 0]

#%%
#Convergence can be assessed with respect to the CPU time used to compute the integral. To return the time taken we must wrap `Integrator` in a WorkTrackingModel.
model = Integrator(integrand)
timer_model = TimerModelWrapper(model, model)
work_model = WorkTrackingModel(timer_model, model,
                               config_var_trans.num_vars())

#%%
#The routine `run_convergence_study` also requires a function `get_num_degrees_of_freedom` which returns the number of degrees of freedom (DoF) for each realization of the configuration variables. In this case the number of DoF is :math:`n_1n_2`. Finally we must specify the number of samples of :math:`z` used to evaluate the integral. Errors are reported as the average error over these samples.

convergence_data = run_convergence_study(
    work_model, variable, validation_levels,
    model.get_num_degrees_of_freedom, config_var_trans,
    num_samples=10, coarsest_levels=coarsest_levels,
    reference_model=true_value)
plot_convergence_data(convergence_data)
plt.show()

#%%
#The left plots depicts the convergence of the estimated integral as :math:`n_1` is increased for varying values of :math:`n_1` and vice-versa for the right plot. These plots confirm that the Integrator converges as the expected linear rate. Until the error introduced by fixing the other configuration variables dominates.
#
#We can also generate similar plots for methods used to solve parameterized partial differential equations. I the following we will assess convergence of a spectral collocation method used to solve the transient advection diffusion equation on a rectangle.


from pyapprox.benchmarks import setup_benchmark
np.random.seed(1)
final_time = .01
# final_time = None
benchmark = setup_benchmark(
    "multi_index_advection_diffusion", nvars=3, corr_len=1, degree=10,
    final_time=final_time)
N = 5
validation_levels = [N, N, N]
coarsest_levels = [0, 0, 0]
if final_time is None:
    validation_levels = validation_levels[:2]
    coarsest_levels = coarsest_levels[:2]
convergence_data = run_convergence_study(
    benchmark.fun, benchmark.variable, validation_levels,
    benchmark.get_num_degrees_of_freedom, benchmark.config_var_trans,
    num_samples=10, coarsest_levels=coarsest_levels)
plot_convergence_data(convergence_data)
plt.show()

#%%
#Note when because the benchmark fun is run using multiprocessing.Pool
#The .py script of this tutorial cannot be run with max_eval_concurrency > 1
#via the shell command using python plot_pde_convergence.py because Pool
#must be called inside
#
#`if __name__ == "__main__":
