import unittest
import numpy as np
from scipy import stats

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.gaussianprocess.kernels import (
    MultilevelKernel, RBF, MonomialScaling)
from pyapprox.util.utilities import (
    get_all_sample_combinations, cartesian_product)
from pyapprox.surrogates.gaussianprocess.calibration import (
    GPCalibrationVariable, CalibrationGaussianProcess)


class TestGPCalibration(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _check_linear_discrepancy(self, nrandom_vars, ndesign_vars,
                                  nsamples_per_model, check_map):
        def model_0(samples):
            # first rows are random samples
            # las rows are calibration inputs
            # return samples.sum(axis=0)[:, None]**4
            return np.cos(np.pi*samples.sum(axis=0)[:, None])

        def model_1(samples):
            # first rows are random samples
            # las rows are calibration inputs
            return model_0(samples) + samples.sum(axis=0)[:, None]**2

        def obs_fun(sim_models, true_theta, xx):
            # discrepancy is only a function of calibration inputs xx
            samples = get_all_sample_combinations(xx, true_theta)
            return (sum([m(samples) for m in sim_models]) +
                    xx.sum(axis=0)[:, None])

        nsim_models = len(nsamples_per_model)-1
        sim_models = [model_0, model_1][:nsim_models]
        nmodels = nsim_models+1

        model_samples = [
            np.random.uniform(0, 1, (nrandom_vars+ndesign_vars, n))
            for m, n in zip(sim_models, nsamples_per_model[:-1])]
        if ndesign_vars == 1:
            obs_samples = np.linspace(0.1, .9, nsamples_per_model[-1])[None, :]
        else:
            obs_samples = cartesian_product([
                np.linspace(0.1, .9, nsamples_per_model[-1])]*ndesign_vars)
            II = np.random.permutation(np.arange(obs_samples.shape[1]))
            obs_samples = obs_samples[:, II[:nsamples_per_model[-1]]]
        # have some samples of calibration inputs used to build GP of model
        # be the same as calibration inputs of trainign data for observation GP
        for ii in range(nsim_models):
            model_samples[ii][:ndesign_vars, :obs_samples.shape[1]] = (
                obs_samples)
        true_theta = np.ones((nrandom_vars, 1))/4
        train_samples = model_samples+[obs_samples]
        train_values = [m(s) for m, s in zip(sim_models, model_samples)] + [
            obs_fun(sim_models, true_theta, obs_samples)]

        nvars = nrandom_vars+ndesign_vars
        length_scales = np.ones(nmodels*nvars)
        rho = np.ones(nmodels-1)
        length_scale_bounds = (1e-1, 10)
        # when length_scale bounds is fixed then the V2(D2) block in Kennedys
        # paper should always be the same regardless of value of theta
        # length_scale_bounds = "fixed"
        kernels = [RBF for nn in range(nmodels)]
        nsamples_per_model = [v.shape[0] for v in train_values]
        kernel_scalings = [
            MonomialScaling(nvars, 0) for nn in range(nmodels-1)]
        # Todo current tests do not pass if sigma_bounds != "fixed"
        ml_kernel = MultilevelKernel(
            nvars, kernels, kernel_scalings, length_scale=length_scales,
            length_scale_bounds=length_scale_bounds, rho=rho,
            rho_bounds=(1e-1, 2), sigma=None, sigma_bounds="fixed")

        gp = CalibrationGaussianProcess(
            ml_kernel, normalize_y=False, n_restarts_optimizer=0)
        gp.set_data(
            train_samples, train_values, true_theta)
        gp.fit()
        print(gp.kernel_)
        # check inactive length scale of obs model kernel V2 is unchanged
        # this will not be true if n_restarts_optimizer > 1
        assert np.allclose(gp.kernel_.length_scale[-1], length_scales[-1])

        random_variable = IndependentMarginalsVariable(
            [stats.uniform(0, 1)]*nrandom_vars)
        mcmc_variable = GPCalibrationVariable(
            random_variable, ml_kernel, train_samples, train_values,
            "metropolis", loglike_grad=True)
        print(mcmc_variable.gp.kernel_.length_scale_bounds)
        map_sample = mcmc_variable.maximum_aposteriori_point()
        print(map_sample, true_theta)
        print(map_sample-true_theta)
        print(mcmc_variable.gp.kernel_)
        # Calibrated GP is not guaranteed to find true random param_idx
        # but in scenarios below it does
        if check_map:
            assert np.allclose(true_theta, map_sample, atol=3e-2)

        ntest_samples = 100
        XX_design = np.random.uniform(0, 1, (ndesign_vars, ntest_samples))
        XX_test = get_all_sample_combinations(XX_design, mcmc_variable.MAP)
        gp_mean = mcmc_variable.gp(XX_test).squeeze()
        true_vals = obs_fun(sim_models, true_theta, XX_design).squeeze()
        # print(gp_mean-true_vals)
        print(np.abs(gp_mean-true_vals).max())
        print(np.abs(gp_mean-true_vals))
        assert np.allclose(gp_mean, true_vals, atol=2.5e-2)

        # from pyapprox.util.visualization import plt
        # ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        # fixed_rho = np.atleast_1d(mcmc_variable.gp.kernel_.rho.copy())[-1]
        # ax.plot(fixed_rho, map_sample, 'ko', ms=30)
        # ax.plot(fixed_rho, mcmc_variable.MAP, 'kX', ms=30)
        # ax.axvline(x=fixed_rho, color="k")
        # bounds = [fixed_rho*0.8, fixed_rho*1.2, 0, 1]
        # mcmc_variable._plot_negloglikelihood_cross_section(ax, bounds)
        # plt.show()

        # # useful code for paper
        # mcmc_variable.gp.set_data(
        #     train_samples, train_values, map_sample)
        # mcmc_variable.gp.fit()
        # ax = mcmc_variable.gp.plot_1d(
        #     map_sample, 100, [0, 1],
        #     plt_kwargs={"color": "b", "ls": "--", "zorder": 10,
        #                 "label": r"Calibrated Obs Model"},
        #     fill_kwargs={"color": "k", "alpha": 0.3},
        #     # prior_fill_kwargs={"color": "g", "alpha": 0.3}
        # )
        # xx = np.linspace(0, 1, 101)[None, :]
        # ax.plot(xx[0, :], obs_fun(sim_models, true_theta, xx), '-k', zorder=1,
        #         label="Obs Process")
        # ax.plot(train_samples[1][0, :], train_values[1], 'ko')
        # ss = get_all_sample_combinations(map_sample, xx)
        # ax.plot(train_samples[0][0, :], train_values[0], 'gX')
        # for ii in range(nsim_models):
        #     ax.plot(xx[0, :], sim_models[ii](ss), 'g-', label=r"$f_{%d}$" % ii)
        #     ax.plot(xx[0, :], mcmc_variable.gp(ss, model_eval_id=ii), ':r',
        #             label=r"Calibrated $f_{%d}$" % ii)
        # ax.legend()
        # plt.show()

    def test_linear_discrepancy(self):
        # answer is sensitive to length_scale and rho bounds
        # TODO mitigate this with multiple restart optimization
        scenarios = [
            [1, 1, [21, 15], True],
            [1, 1, [21, 11, 7], True],
            [2, 1, [51, 11], False],
            [2, 2, [51, 31], False],
        ]
        for scenario in scenarios:
            print('scenario', scenario)
            np.random.seed(1)
            self._check_linear_discrepancy(*scenario)


if __name__ == "__main__":
    gp_calibration_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGPCalibration)
    unittest.TextTestRunner(verbosity=2).run(gp_calibration_test_suite)

# use following to ignore all warnings when running tests
# python -W ignore -m unittest pyapprox.surrogates.gaussianprocess.tests.test_calibration.TestGPCalibration.
