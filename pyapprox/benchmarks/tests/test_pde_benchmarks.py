import unittest
import numpy as np
from functools import partial
from pyapprox.optimization.pya_minimize import pyapprox_minimize

from pyapprox.benchmarks.pde_benchmarks import (
    _setup_inverse_advection_diffusion_benchmark,
    _setup_multi_index_advection_diffusion_benchmark)
from pyapprox.util.utilities import check_gradients

class TestPDEBenchmarks(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_setup_inverse_advection_diffusion_benchmark(self):
        nobs = 20
        noise_std = 1e-8 #0.01  # make sure it does not dominate observed values
        length_scale = .5
        sigma = 1
        nvars = 10
        orders = [20, 20]

        inv_model, variable, true_params, noiseless_obs, obs = (
            _setup_inverse_advection_diffusion_benchmark(
                nobs, noise_std, length_scale, sigma, nvars, orders))

        # TODO add std to params list
        init_guess = variable.rvs(1)
        # init_guess = true_params
        errors = check_gradients(
            lambda zz: inv_model._eval(zz[:, 0], jac=True),
            True, init_guess, plot=False,
            fd_eps=np.logspace(-13, 0, 14)[::-1])
        assert errors[0] > 5e-2 and errors.min() < 3e-6

        jac = True
        opt_result = pyapprox_minimize(
            partial(inv_model._eval, jac=jac),
            init_guess,
            method="trust-constr", jac=jac,
            options={"verbose": 0, "gtol": 1e-6, "xtol": 1e-16})
        # print(opt_result.x)
        # print(true_params.T)
        print(opt_result.x-true_params.T)
        assert np.allclose(opt_result.x, true_params.T, atol=2e-5)

    def test_setup_multi_index_advection_diffusion_benchmark(self):
        length_scale = .5
        sigma = 1
        nvars = 10
        hf_orders = [20, 20]
        model, variable = _setup_multi_index_advection_diffusion_benchmark(
            length_scale, sigma, nvars, hf_orders)
        

if __name__ == "__main__":
    pde_benchmarks_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestPDEBenchmarks)
    unittest.TextTestRunner(verbosity=2).run(pde_benchmarks_test_suite)
