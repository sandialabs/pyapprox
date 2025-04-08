import unittest

import numpy as np

from pyapprox.benchmarks.multifidelity_benchmarks import TunableModelEnsemble
from pyapprox.multifidelity.etc import AETCBLUE
from pyapprox.multifidelity.factory import get_estimator, multioutput_stats
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.multifidelity.groupacv import (
    GroupACVGradientOptimizer,
    ChainedACVOptimizer,
    MLBLUESPDOptimizer,
    MLBLUEGradientOptimizer,
)
from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer,
    ScipyConstrainedNelderMeadOptimizer,
)


class TestETC(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    @staticmethod
    def _setup_model_ensemble_tunable(
            shifts=None, angle=np.pi / 4, bkd=NumpyMixin
    ):
        example = TunableModelEnsemble(angle, shifts, bkd)
        cov = example.covariance()
        costs = 10.0 ** (-bkd.arange(cov.shape[0]))
        # costs = bkd.logspace(0, -1, cov.shape[0])
        return example.models(), cov, costs, example.variable()

    def test_AETC_optimal_loss(self):
        """
        Tests if the optimal loss returned from using oracle stats is the
        same as using without oracle stats given many samples.
        """
        # bkd = NumpyMixin
        bkd = TorchMixin
        alpha = 1000
        nsamples = int(1e6)
        shifts = bkd.array([1, 2])
        funs, cov, costs, variable = self._setup_model_ensemble_tunable(
            shifts, bkd=bkd
        )
        target_cost = bkd.sum(costs) * (nsamples + 10)

        true_means = bkd.hstack((bkd.array(0), shifts))[:, None]
        oracle_stats = [cov, true_means]

        samples = variable.rvs(nsamples)
        values = bkd.hstack([fun(samples) for fun in funs])

        exploit_cost = 0.5 * target_cost
        covariate_subset = bkd.asarray([0, 1], dtype=int)
        hf_values = values[:, :1]
        covariate_values = values[:, covariate_subset + 1]

        est_nor = AETCBLUE(funs, variable.rvs, costs, oracle_stats=None)
        est_or = AETCBLUE(funs, variable.rvs, costs, oracle_stats=oracle_stats)
        result_oracle = est_or._optimal_loss(
            target_cost,
            hf_values,
            covariate_values,
            costs,
            covariate_subset,
            alpha,
            exploit_cost,
        )
        result_mc = est_nor._optimal_loss(
            target_cost,
            hf_values,
            covariate_values,
            costs,
            covariate_subset,
            alpha,
            exploit_cost,
        )
        assert bkd.allclose(result_mc[-2], result_oracle[-2], rtol=1e-2)

    def _get_chained_optimizer(self):
        opt1 = GroupACVGradientOptimizer(
            ScipyConstrainedNelderMeadOptimizer(opts={"maxiter": 20})
        )
        scipy_opt = ScipyConstrainedOptimizer()
        scipy_opt.set_verbosity(0)
        # opt2 = GroupACVGradientOptimizer(scipy_opt)
        opt2 = MLBLUEGradientOptimizer(scipy_opt)
        opt = ChainedACVOptimizer(opt1, opt2)
        return opt

    # @unittest.skipIf(True, "not released yet")
    def test_aetc_blue(self):
        # bkd = NumpyMixin
        bkd = TorchMixin
        target_cost = 300  # 1e3
        shifts = bkd.array([1, 2])
        funs, cov, costs, variable = self._setup_model_ensemble_tunable(
            shifts, bkd=bkd
        )

        true_means = bkd.hstack((bkd.array(0), shifts))[:, None]
        # todo switch on and off oracle stats
        oracle_stats = None
        # subsets = None
        subsets = [bkd.array([0, 1])]

        optimizer = self._get_chained_optimizer()
        # optimizer = MLBLUESPDOptimizer(solver_name="CLARABEL")
        estimator = AETCBLUE(
            funs, variable.rvs, costs, oracle_stats, 0, optimizer, backend=bkd
        )
        mean, values, result = estimator.estimate(
            target_cost, return_dict=False, subsets=subsets
        )
        result_dict = estimator._explore_result_to_dict(result)
        cov_exe = bkd.cov(values, rowvar=False, ddof=1)

        subset = result_dict["subset"] + 1
        stat = multioutput_stats["mean"](1, backend=bkd)
        stat.set_pilot_quantities(cov_exe[np.ix_(subset, subset)])
        mlblue_est = get_estimator(
            "mlblue",
            stat,
            costs[subset],
            asketch=result_dict["beta_Sp"][1:].T,
            backend=bkd,
        )
        unrounded_true_var = mlblue_est._covariance_from_npartition_samples(
            result_dict["nsamples_per_subset"]
        )
        # print(result_dict["sigma_S"], cov_exe)
        assert bkd.allclose(
            result_dict["sigma_S"], cov_exe[np.ix_(subset, subset)]
        )

        assert bkd.allclose(unrounded_true_var, result_dict["BLUE_variance"])
        assert bkd.allclose(
            result_dict["BLUE_variance"], unrounded_true_var, rtol=3e-2
        )

        noracle_samples = 1e5
        oracle_samples = variable.rvs(noracle_samples)
        oracle_hf_values = funs[0](oracle_samples)
        active_funs_idx = []
        for ii in range(1, len(funs)):
            for subset in subsets:
                if ii - 1 in subset and ii not in active_funs_idx:
                    active_funs_idx.append(ii)
                    break
        # print(active_funs_idx)
        oracle_covariate_values = bkd.hstack(
            [funs[ii](oracle_samples) for ii in active_funs_idx]
        )
        true_beta_Sp = estimator._least_squares(
            oracle_hf_values, oracle_covariate_values
        )[0]

        ntrials = int(1e3)
        means = bkd.empty(ntrials)
        sq_biases, variances = [], []
        print(true_means[0], true_means[active_funs_idx])
        true_active_means = bkd.hstack(
            (true_means[0], true_means[active_funs_idx, 0])
        )
        for ii in range(ntrials):
            if ii % 100 == 0:
                print("Testing", 100 * ii // ntrials, "%")
                # if ii > 0:
                #     assert False
            # print(estimator)
            means[ii], values_per_model, result = estimator.estimate(
                target_cost, subsets=subsets
            )
            sq_biases.append(
                (true_active_means @ (true_beta_Sp - result["beta_Sp"])) ** 2
            )
            variances.append(result["BLUE_variance"])

        mse = bkd.mean((means - true_means[0]) ** 2)
        # Verifies that the true MSE is close to the theoretical MSE
        # However, the theoretical MSE is not exactly the True anyway
        print(mse, result_dict["loss"])
        print(mse-result_dict["loss"])
        assert bkd.allclose(mse, result_dict["loss"], rtol=3e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
