import unittest
import numpy as np
from scipy import stats

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.gaussianprocess.kernels import MultilevelKernel, RBF
from pyapprox.util.utilities import get_all_sample_combinations
from pyapprox.surrogates.gaussianprocess.calibration import (
    GPCalibrationVariable, CalibrationGaussianProcess)


class TestGPCalibration(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_linear_discrepancy(self):
        def model(samples):
            # first rows are random samples
            # las rows are calibration inputs
            return samples.sum(axis=0)[:, None]**4

        def obs_fun(true_theta, xx):
            # discrepancy is only a function of calibration inputs xx
            samples = get_all_sample_combinations(xx, true_theta)
            return model(samples) + xx.sum(axis=0)[:, None]

        nmodels = 2
        nrandom_vars, ndesign_vars = 1, 1
        nmodel_samples, nobs_samples = 21, 10
        model_samples = np.random.uniform(
            0, 1, (nrandom_vars+ndesign_vars, nmodel_samples))
        obs_samples = np.linspace(0.1, .9, nobs_samples)[None, :]
        # have some samples of calibration inputs used to build GP of model
        # be the same as calibration inputs of trainign data for observation GP
        model_samples[:ndesign_vars, :obs_samples.shape[1]] = obs_samples
        true_theta = np.ones((nrandom_vars, 1))/4
        train_samples = [model_samples, obs_samples]
        train_values = [model(model_samples), obs_fun(true_theta, obs_samples)]

        nvars = nrandom_vars+ndesign_vars
        length_scales = np.ones(nmodels*nvars)
        rho = np.ones(nmodels-1)
        length_scale_bounds = (1e-1, 10)
        # when length_scale bounds is fixed then the V2(D2) block in Kennedys
        # paper should always be the same regardless of value of theta
        # length_scale_bounds = "fixed"
        kernels = [RBF() for nn in range(nmodels)]
        nsamples_per_model = [v.shape[0] for v in train_values]
        ml_kernel = MultilevelKernel(
            nvars, nsamples_per_model, kernels, length_scale=length_scales,
            length_scale_bounds=length_scale_bounds, rho=rho)

        gp = CalibrationGaussianProcess(
            ml_kernel, normalize_y=False, n_restarts_optimizer=0)
        gp.set_data(
            train_samples, train_values, true_theta)
        gp.fit()
        print(gp.kernel_)
        # check inactive length scale of obs model kernel V2 is unchanged
        # this will not be true if n_restarts_optimizer > 1
        assert np.allclose(gp.kernel_.length_scale[-1], length_scales[-1])
        # ml_kernel.length_scale_bounds = "fixed"
        # ml_kernel.length_scale = gp.kernel_.length_scale

        random_variable = IndependentMarginalsVariable(
            [stats.uniform(0, 1)]*nrandom_vars)
        mcmc_variable = GPCalibrationVariable(
            random_variable, ml_kernel, train_samples, train_values,
            "metropolis")
        map_sample = mcmc_variable.maximum_aposteriori_point()
        print(map_sample-true_theta)
        assert np.allclose(true_theta, map_sample, atol=1e-2)
        #assert False

        # from pyapprox.interface.wrappers import (
        #     evaluate_1darray_function_on_2d_array)
        # from pyapprox.util.visualization import get_meshgrid_function_data, plt
        # fixed_rho = mcmc_variable.gp.kernel_.rho.copy()
        # plt.plot(fixed_rho, map_sample, 'ko', ms=30)
        # plt.plot(fixed_rho, 0.5, 'kX', ms=30)
        # plt.axvline(x=fixed_rho, color="k")
        # from functools import partial
        # hyperparams = np.hstack((mcmc_variable.gp.kernel_.length_scale.copy(),
        #                          mcmc_variable.gp.kernel_.rho.copy()))
        # def plotfun_1d_array(xx):
        #     zz = np.hstack((hyperparams, xx[-1:]))
        #     zz[-2] = xx[0]
        #     return mcmc_variable.loglike_calibration_and_hyperparams(
        #         zz[:, None])
        # plotfun = partial(
        #     evaluate_1darray_function_on_2d_array, plotfun_1d_array)
        # print(map_sample, "MAP")
        # zz = np.hstack((fixed_rho, map_sample[0]))
        # print(plotfun(zz[:, None]))
        # print(mcmc_variable._loglike(map_sample))
        # X, Y, Z = get_meshgrid_function_data(
        #     plotfun, [fixed_rho*0.8, fixed_rho*1.2, 0, 1], 30)
        # im = plt.contourf(
        #     X, Y, Z,  levels=np.linspace(Z.min(), Z.max(), 20))
        # plt.colorbar(im)
        # plt.figure()
        # xx = np.vstack((np.full((1, 31), fixed_rho),
        #                 np.linspace(0, 1, 31)[None, :]))
        # plt.plot(xx[1, :], plotfun(xx))
        # print(mcmc_variable.gp.kernel_)
        # plt.show()
        # #assert False

        # # nrandom_samples = 10
        # # random_samples = mcmc_variable.rvs(nrandom_samples)
        # random_samples = map_sample
        # # random_samples = true_theta
        # for ii in range(random_samples.shape[1]):
        #     # stop length_scale bounds from being fixed
        #     # gp.kernel_.length_scale_bounds = length_scale_bounds
        #     mcmc_variable.gp.set_data(
        #         train_samples, train_values, random_samples[:, ii:ii+1])
        #     mcmc_variable.gp.fit()
        #     print(mcmc_variable.gp.kernel_)
        #     ax = mcmc_variable.gp.plot_1d(
        #         random_samples[:, ii:ii+1], 100, [0, 1],
        #         plt_kwargs={"color": "b", "ls": "--", "zorder": 10},
        #         fill_kwargs={"color": "k", "alpha": 0.3},
        #         # prior_fill_kwargs={"color": "g", "alpha": 0.3}
        #     )
        #     xx = np.linspace(0, 1, 101)[None, :]
        #     ax.plot(xx[0, :], obs_fun(true_theta, xx), '-k', zorder=1)
        #     ax.plot(train_samples[1][0, :], train_values[1], 'ko')
        #     ss = get_all_sample_combinations(random_samples[:, ii:ii+1], xx)
        #     ax.plot(train_samples[0][0, :], train_values[0], 'gX')
        #     ax.plot(xx[0, :], model(ss), '-g')
        #     ax.plot(xx[0, :], mcmc_variable.gp(ss, model_eval_id=0), ':r')
        #     import matplotlib.pyplot as plt
        #     plt.show()

        # # TODO: obs_model kernel should not have length scales associated
        # # with random variables. This should not effect result but
        # # requires wasted optimization effort


if __name__ == "__main__":
    gp_calibration_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGPCalibration)
    unittest.TextTestRunner(verbosity=2).run(gp_calibration_test_suite)

# use following to ignore all warnings when running tests
# python -W ignore -m unittest pyapprox.surrogates.gaussianprocess.tests.test_calibration.TestGPCalibration.
