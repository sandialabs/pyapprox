r"""
Convergence studies
-------------------
"""
import numpy as np
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox.convergence_studies import \
    run_convergence_study, plot_convergence_data
from pyapprox.configure_plots import plt

if __name__ == "__main__":
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

    from pyapprox.utilities import (
        get_tensor_product_piecewise_polynomial_quadrature_rule,
    )
    from pyapprox.models.wrappers import (
        evaluate_1darray_function_on_2d_array, WorkTrackingModel,
        TimerModelWrapper
    )
    from scipy import stats
    from pyapprox import IndependentMultivariateRandomVariable
    from pyapprox.variable_transformations import (
        ConfigureVariableTransformation
    )

    class Integrator(object):
        def set_quad_rule(self, nsamples_1d):
            self.xquad, self.wquad = \
                get_tensor_product_piecewise_polynomial_quadrature_rule(
                    nsamples_1d, [0, 1, 0, 1], degree=1)

        def integrate(self, sample):
            self.set_quad_rule(sample[-2:].astype(int))
            val = self.integrand(sample[:-2], self.xquad)[:, 0].dot(self.wquad)
            return val

        def integrand(self, sample, x):
            return np.sum(sample[0]*x**2, axis=0)[:, None]

        def __call__(self, samples):
            return evaluate_1darray_function_on_2d_array(
                self.integrate, samples, None)

        @staticmethod
        def get_num_degrees_of_freedom(config_sample):
            return np.prod(config_sample)

        @staticmethod
        def true_value(samples):
            return 2/3*samples.T

    model = Integrator()
    config_values = [2**np.arange(1, 11)+1, 2**np.arange(1, 11)+1]
    config_var_trans = ConfigureVariableTransformation(config_values)
    validation_levels = [5, 5]
    coarsest_levels = [0, 0]
    variable = IndependentMultivariateRandomVariable([stats.uniform(0, 1)])

    timer_model = TimerModelWrapper(model, model)
    work_model = WorkTrackingModel(timer_model, model,
                                   config_var_trans.num_vars())
    convergence_data = run_convergence_study(
        work_model, variable, validation_levels,
        model.get_num_degrees_of_freedom, config_var_trans,
        num_samples=10, coarsest_levels=coarsest_levels,
        reference_model=model.true_value)
    plot_convergence_data(convergence_data)
    plt.show()
#%%
# Note when because the benchmark fun is run using multiprocessing.Pool
# The .py script of this tutorial cannot be run with
# python plot_pde_convergence.py because Pool must be called inside
# if __name__ == "__main__": block
