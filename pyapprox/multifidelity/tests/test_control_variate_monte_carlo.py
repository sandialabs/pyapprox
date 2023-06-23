import unittest
import numpy as np
from scipy import stats
from functools import partial

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.multifidelity.control_variate_monte_carlo import (
    allocate_samples_mlmc,
    get_discrepancy_covariances_MF, get_nsamples_per_model,
    acv_sample_allocation_objective_all, _ndarray_as_pkg_format,
    use_torch, mlmc_sample_allocation_objective_all_lagrange,
    mlmc_sample_allocation_jacobian_all_lagrange_torch,
    acv_sample_allocation_gmf_ratio_constraint, round_nsample_ratios,
    acv_sample_allocation_gmf_ratio_constraint_jac, get_nhf_samples,
    acv_sample_allocation_nhf_samples_constraint,
    acv_sample_allocation_nhf_samples_constraint_jac,
    get_sample_allocation_matrix_mfmc, get_sample_allocation_matrix_acvmf,
    get_sample_allocation_matrix_acvis, get_sample_allocation_matrix_mlmc,
    get_npartition_samples_mlmc, get_npartition_samples_mfmc,
    get_npartition_samples_acvmf,
    get_npartition_samples_acvis, get_nsamples_intersect, get_nsamples_subset,
    get_acv_discrepancy_covariances_multipliers, get_acv_recursion_indices,
    get_acv_initial_guess, get_discrepancy_covariances_IS,
    acv_sample_allocation_nlf_gt_nhf_ratio_constraint,
    acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac,
    reorder_allocation_matrix_acvgmf,
    get_nsamples_interesect_from_z_subsets_acvgmf,
    estimate_model_ensemble_covariance, generate_samples_and_values_mfmc,
    get_mfmc_control_variate_weights,
    get_rsquared_mfmc, get_lagrange_multiplier_mlmc
)
from pyapprox.multifidelity.monte_carlo_estimators import (
    get_estimator, estimate_variance, bootstrap_monte_carlo_estimator,
    AETCBLUE
)
from pyapprox.util.utilities import (
    check_gradients, get_all_sample_combinations,
    approx_jacobian, get_correlation_from_covariance
)
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.interface.wrappers import ModelEnsemble
from pyapprox.benchmarks.multifidelity_benchmarks import (
    ShortColumnModelEnsemble, TunableModelEnsemble, PolynomialModelEnsemble
)
from pyapprox.multifidelity.multilevelblue import BLUE_betas, BLUE_variance


skiptest = unittest.skipIf(
    not use_torch, reason="torch not installed")


def setup_model_ensemble_short_column(
        nmodels=5, npilot_samples=None):
    example = ShortColumnModelEnsemble()
    model_ensemble = ModelEnsemble(
        [example.models[ii] for ii in range(nmodels)])
    univariate_variables = [
        stats.uniform(5, 10), stats.uniform(15, 10), stats.norm(500, 100),
        stats.norm(2000, 400), stats.lognorm(s=0.5, scale=np.exp(5))]
    variable = IndependentMarginalsVariable(univariate_variables)
    generate_samples = partial(
        generate_independent_random_samples, variable)

    if npilot_samples is not None:
        # The number of pilot samples effects ability of numerical estimate
        # of variance reduction to match theoretical value
        cov, samples, weights = estimate_model_ensemble_covariance(
            npilot_samples, generate_samples, model_ensemble,
            model_ensemble.nmodels)
    else:
        # it is difficult to create a quadrature rule for the lognormal
        # distribution so instead define the variable as normal and then
        # apply log transform
        univariate_variables = [
            stats.uniform(5, 10), stats.uniform(15, 10), stats.norm(500, 100),
            stats.norm(2000, 400), stats.norm(loc=5, scale=0.5)]
        variable = IndependentMarginalsVariable(
            univariate_variables)

        example.apply_lognormal = True
        cov = example.get_covariance_matrix(variable)[:nmodels, :nmodels]
        example.apply_lognormal = False

    return model_ensemble, cov, generate_samples


def setup_model_ensemble_tunable(shifts=None):
    example = TunableModelEnsemble(np.pi/4, shifts)
    model_ensemble = ModelEnsemble(example.models)
    cov = example.get_covariance_matrix()
    costs = 10.**(-np.arange(cov.shape[0]))
    return model_ensemble, cov, costs, example.variable


def setup_model_ensemble_polynomial():
    example = PolynomialModelEnsemble()
    model_ensemble = ModelEnsemble(example.models)
    cov = example.get_covariance_matrix()
    costs = np.asarray([100//2**ii for ii in range(example.nmodels)])
    # npilot_samples=int(1e6)
    # cov, samples, weights = estimate_model_ensemble_covariance(
    #    npilot_samples,generate_samples,model_ensemble, model_ensemble.nmodels)
    return model_ensemble, cov, costs, example.variable


class TestCVMC(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_estimate_model_ensemble_covariance(self):
        model, cov, costs, variable = setup_model_ensemble_tunable()
        print(get_correlation_from_covariance(cov))
        mc_cov = estimate_model_ensemble_covariance(
            int(1e6), variable.rvs, model, cov.shape[0])[0]
        assert np.allclose(cov, mc_cov, atol=1e-2)

    def test_model_ensemble(self):
        model = ModelEnsemble([lambda x: x.T, lambda x: x.T**2])
        variable = IndependentMarginalsVariable([stats.uniform(0, 1)])
        samples_per_model = [variable.rvs(10), variable.rvs(5)]
        values_per_model = model.evaluate_models(samples_per_model)
        for ii in range(model.nmodels):
            assert np.allclose(
                values_per_model[ii], model.functions[ii](
                    samples_per_model[ii]))

    def test_mlmc_sample_allocation(self):
        # The following will give mlmc with unit variance
        # and discrepancy variances 1, 4, 4
        target_cost = 81
        cov = np.asarray([[1.00, 0.50, 0.25],
                          [0.50, 1.00, 0.50],
                          [0.25, 0.50, 4.00]])
        # ensure cov is positive definite
        np.linalg.cholesky(cov)
        # print(np.linalg.inv(cov))
        costs = [6, 3, 1]
        nsample_ratios, log10_var = allocate_samples_mlmc(
            cov, costs, target_cost)
        assert np.allclose(10**log10_var, 1)
        nsamples = get_nsamples_per_model(
            target_cost, costs, nsample_ratios)
        nsamples_discrepancy = 9*np.sqrt(np.asarray([1/(6+3), 4/(3+1), 4]))
        nsamples_true = [
            nsamples_discrepancy[0], nsamples_discrepancy[:2].sum(),
            nsamples_discrepancy[1:3].sum()]
        # print(nsamples, nsamples_true)
        assert np.allclose(nsamples, nsamples_true)

    def test_get_nhf_samples(self):
        target_cost, costs = 15, np.array([1, 0.5, 0.25])
        nsample_ratios = [2, 4]
        nhf_samples = get_nhf_samples(target_cost, costs, nsample_ratios)
        assert nhf_samples == 5

    def test_round_nsample_ratios(self):
        target_cost, costs = 10, np.array([1, 0.5, 0.25])
        nsample_ratios = np.array([2.1, 5.6])
        rounded_nsample_ratios, rounded_target_cost = round_nsample_ratios(
            target_cost, costs, nsample_ratios)
        nsamples = get_nsamples_per_model(
            rounded_target_cost, costs, rounded_nsample_ratios)
        assert np.allclose(nsamples, nsamples.astype(int))

    def test_get_sample_allocation_matrix_mlmc(self):
        nsamples_per_model = np.array([2, 4, 6, 8])
        mat = get_sample_allocation_matrix_mlmc(
            nsamples_per_model.shape[0])
        assert np.allclose(mat, np.array([[0, 1, 1, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 1, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 1]]))
        npartition_samples = get_npartition_samples_mlmc(nsamples_per_model)
        assert np.allclose(npartition_samples, np.array([2, 2, 4, 4]))

    def test_get_sample_allocation_matrix_mfmc(self):
        nsamples_per_model = np.array([2, 4, 6, 8])
        mat = get_sample_allocation_matrix_mfmc(
            nsamples_per_model.shape[0])
        assert np.allclose(mat, np.array([[0, 1, 1, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 0, 0, 1, 1, 1],
                                          [0, 0, 0, 0, 0, 0, 0, 1]]))
        npartition_samples = get_npartition_samples_mfmc(nsamples_per_model)
        assert np.allclose(npartition_samples, np.array([2, 2, 2, 2]))

    def test_get_sample_allocation_matrix_acvmf(self):
        nsamples_per_model = np.array([2, 4, 6, 8])
        recursion_index = np.array([0, 0, 0])
        mat = get_sample_allocation_matrix_acvmf(recursion_index)
        assert np.allclose(mat, np.array([[0, 1, 1, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 1, 0, 1, 0, 1],
                                          [0, 0, 0, 0, 0, 1, 0, 1],
                                          [0, 0, 0, 0, 0, 0, 0, 1]]))
        npartition_samples = get_npartition_samples_acvmf(nsamples_per_model)
        assert np.allclose(npartition_samples, np.array([2, 2, 2, 2]))

        nsamples_per_model = np.array([2, 4, 8, 6])
        reorder_mat = reorder_allocation_matrix_acvgmf(
            mat, nsamples_per_model, recursion_index)
        assert np.allclose(reorder_mat,
                           [[0., 1., 1., 1., 1., 1., 1., 1.],
                            [0., 0., 0., 1., 0., 1., 0., 1.],
                            [0., 0., 0., 0., 0., 1., 0., 1.],
                            [0., 0., 0., 0., 0., 1., 0., 0.]])

        npartition_samples = get_npartition_samples_acvmf(nsamples_per_model)
        nsamples_intersect = get_nsamples_intersect(
            reorder_mat, npartition_samples)
        nsamples_z_subsets = nsamples_per_model
        nsamples_intersect_true = \
            get_nsamples_interesect_from_z_subsets_acvgmf(
                nsamples_z_subsets, recursion_index)
        assert np.allclose(nsamples_intersect, nsamples_intersect_true)
        assert np.allclose(
            get_nsamples_subset(reorder_mat, npartition_samples),
            [0, 2, 2, 4, 2, 8, 2, 6])

        recursion_index = np.array([0, 1, 1, 3])
        nsamples_per_model = np.array([2, 4, 5, 8, 5])
        mat = get_sample_allocation_matrix_acvmf(recursion_index)
        reorder_mat = reorder_allocation_matrix_acvgmf(
            mat, nsamples_per_model, recursion_index)
        assert np.allclose(reorder_mat,
                           [[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                            [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
                            [0., 0., 0., 0., 0., 1., 0., 1., 1., 1.],
                            [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

        npartition_samples = get_npartition_samples_acvmf(nsamples_per_model)
        nsamples_intersect = get_nsamples_intersect(
            reorder_mat, npartition_samples)
        nsamples_z_subsets = nsamples_per_model
        nsamples_intersect_true = \
            get_nsamples_interesect_from_z_subsets_acvgmf(
                nsamples_z_subsets, recursion_index)
        assert np.allclose(nsamples_intersect, nsamples_intersect_true)
        assert np.allclose(
            get_nsamples_subset(reorder_mat, npartition_samples),
            [0, 2, 2, 4, 4, 5, 4, 8, 8, 5])

        nsamples_per_model = np.array([2, 4, 5, 8, 5])
        nmodels = len(nsamples_per_model)
        for recursion_index in get_acv_recursion_indices(nmodels):
            nsamples_per_model = np.array([2, 4, 5, 8, 5])
            mat = get_sample_allocation_matrix_acvmf(recursion_index)
            reorder_mat = reorder_allocation_matrix_acvgmf(
                mat, nsamples_per_model, recursion_index)
            npartition_samples = get_npartition_samples_acvmf(
                nsamples_per_model)
            nsamples_intersect = get_nsamples_intersect(
                reorder_mat, npartition_samples)
            nsamples_z_subsets = nsamples_per_model
            nsamples_intersect_true = \
                get_nsamples_interesect_from_z_subsets_acvgmf(
                    nsamples_z_subsets, recursion_index)
            assert np.allclose(nsamples_intersect, nsamples_intersect_true)

        # recover mfmc
        nsamples_per_model = np.array([2, 4, 6, 8])
        recursion_index = [0, 1, 2]
        mat = get_sample_allocation_matrix_acvmf(recursion_index)
        assert np.allclose(mat, np.array([[0, 1, 1, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 0, 0, 1, 1, 1],
                                          [0, 0, 0, 0, 0, 0, 0, 1]]))
        npartition_samples = get_npartition_samples_acvmf(nsamples_per_model)
        assert np.allclose(npartition_samples, np.array([2, 2, 2, 2]))

        recursion_index = [0, 0, 2]
        mat = get_sample_allocation_matrix_acvmf(recursion_index)
        assert np.allclose(mat, np.array([[0, 1, 1, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 1, 0, 1, 1, 1],
                                          [0, 0, 0, 0, 0, 1, 1, 1],
                                          [0, 0, 0, 0, 0, 0, 0, 1]]))
        npartition_samples = get_npartition_samples_acvmf(nsamples_per_model)
        assert np.allclose(npartition_samples, np.array([2, 2, 2, 2]))

    def test_get_sample_allocation_matrix_acvis(self):
        nsamples_per_model = np.array([2, 4, 6, 8])
        recursion_index = [0, 0, 0]
        mat = get_sample_allocation_matrix_acvis(recursion_index)
        assert np.allclose(mat, np.array([[0, 1, 1, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 1]]))
        npartition_samples = get_npartition_samples_acvis(nsamples_per_model)
        assert np.allclose(npartition_samples, [2, 2, 4, 6])

        recursion_index = [0, 1, 1]
        mat = get_sample_allocation_matrix_acvis(recursion_index)
        assert np.allclose(mat, np.array([[0, 1, 1, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 1, 1, 1, 1],
                                          [0, 0, 0, 0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 1]]))

    def test_get_nsamples_intersect_and_subset(self):
        nsamples_per_model = np.array([2, 4, 6, 8])
        recursion_index = [0, 0, 0]
        allocation_matrix = get_sample_allocation_matrix_acvmf(recursion_index)
        npartition_samples = get_npartition_samples_mfmc(nsamples_per_model)
        nsamples_intersect = get_nsamples_intersect(
            allocation_matrix, npartition_samples)
        print(nsamples_intersect)
        nsamples_interesect_true = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 2., 2., 2., 2., 2., 2., 2.],
             [0., 2., 2., 2., 2., 2., 2., 2.],
             [0., 2., 2., 4., 2., 4., 2., 4.],
             [0., 2., 2., 2., 2., 2., 2., 2.],
             [0., 2., 2., 4., 2., 6., 2., 6.],
             [0., 2., 2., 2., 2., 2., 2., 2.],
             [0., 2., 2., 4., 2., 6., 2., 8.]])
        assert np.allclose(nsamples_intersect, nsamples_interesect_true)
        nsamples_subset = get_nsamples_subset(
            allocation_matrix, npartition_samples)
        assert np.allclose(nsamples_subset, [0, 2, 2, 4, 2, 6, 2, 8])

        allocation_matrix = get_sample_allocation_matrix_mlmc(
            nsamples_per_model.shape[0])
        npartition_samples = get_npartition_samples_mlmc(nsamples_per_model)
        nsamples_intersect = get_nsamples_intersect(
            allocation_matrix, npartition_samples)
        print(nsamples_intersect)
        nsamples_interesect_true = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 2., 2., 0., 0., 0., 0., 0.],
             [0., 2., 2., 0., 0., 0., 0., 0.],
             [0., 0., 0., 2., 2., 0., 0., 0.],
             [0., 0., 0., 2., 2., 0., 0., 0.],
             [0., 0., 0., 0., 0., 4., 4., 0.],
             [0., 0., 0., 0., 0., 4., 4., 0.],
             [0., 0., 0., 0., 0., 0., 0., 4.]])
        assert np.allclose(nsamples_intersect, nsamples_interesect_true)
        nsamples_subset = get_nsamples_subset(
            allocation_matrix, npartition_samples)
        assert np.allclose(nsamples_subset, [0, 2, 2, 2, 2, 4, 4, 4])

    def test_get_discrepancy_covariances_multipliers(self):
        target_cost, costs = 100, [1, 1, 1, 1]
        nsample_ratios = np.array([2, 3, 4])
        recursion_index = np.array([0, 0, 0])
        allocation_mat = get_sample_allocation_matrix_acvmf(recursion_index)
        nsamples_per_model = get_nsamples_per_model(
            target_cost, costs, nsample_ratios)
        Gmat, gvec = get_acv_discrepancy_covariances_multipliers(
            allocation_mat, nsamples_per_model, get_npartition_samples_mfmc,
            recursion_index)
        Fmat, fvec = get_discrepancy_covariances_MF(
            np.ones((len(costs), len(costs))), nsample_ratios)
        nhf_samples = get_nhf_samples(target_cost, costs, nsample_ratios)
        assert np.allclose(gvec*nhf_samples, fvec)
        assert np.allclose(Gmat*nhf_samples, Fmat)

        target_cost, costs = 100, [1, 1, 1, 1]
        nsample_ratios = np.array([2, 3, 4])
        recursion_index = np.array([0, 0, 0])
        allocation_mat = get_sample_allocation_matrix_acvis(recursion_index)
        nsamples_per_model = get_nsamples_per_model(
            target_cost, costs, nsample_ratios)
        Gmat, gvec = get_acv_discrepancy_covariances_multipliers(
            allocation_mat, nsamples_per_model, get_npartition_samples_acvis,
            recursion_index)
        Fmat, fvec = get_discrepancy_covariances_IS(
            np.ones((len(costs), len(costs))), nsample_ratios)
        nhf_samples = get_nhf_samples(target_cost, costs, nsample_ratios)
        assert np.allclose(gvec*nhf_samples, fvec)
        assert np.allclose(Gmat*nhf_samples, Fmat)

        target_cost, costs = 30, [1, 1, 1]
        nsample_ratios = np.array([2, 3])
        recursion_index = np.array([0, 1])
        allocation_mat = get_sample_allocation_matrix_acvmf(recursion_index)
        nsamples_per_model = get_nsamples_per_model(
            target_cost, costs, nsample_ratios)
        Gmat, gvec = get_acv_discrepancy_covariances_multipliers(
            allocation_mat, nsamples_per_model, get_npartition_samples_acvmf,
            recursion_index)
        nhf_samples = get_nhf_samples(target_cost, costs, nsample_ratios)
        fvec = np.array([1/10, 1/30.])*nhf_samples
        Fmat = np.diag(fvec)
        assert np.allclose(gvec*nhf_samples, fvec)
        assert np.allclose(Gmat*nhf_samples, Fmat)

    def test_generate_samples_and_values_mfmc(self):
        functions = ShortColumnModelEnsemble()
        model_ensemble = ModelEnsemble(
            [functions.m0, functions.m1, functions.m2])
        univariate_variables = [
            stats.uniform(5, 10), stats.uniform(15, 10), stats.norm(500, 100),
            stats.norm(2000, 400), stats.lognorm(s=0.5, scale=np.exp(5))]
        variable = IndependentMarginalsVariable(
            univariate_variables)
        generate_samples = partial(
            generate_independent_random_samples, variable)

        nhf_samples = 10
        costs = 10.**(-np.arange(3))
        nsample_ratios = np.array([2, 4])
        target_cost = nhf_samples*(costs[0]+np.dot(costs[1:], nsample_ratios))
        nsamples_per_model = get_nsamples_per_model(
            target_cost, costs, nsample_ratios).astype(int)
        samples, values =\
            generate_samples_and_values_mfmc(
                nsamples_per_model, model_ensemble.functions, generate_samples)

        nhf_samples = nsamples_per_model[0]
        for jj in range(1, len(samples)):
            assert samples[jj][1].shape[1] == nsamples_per_model[jj]
            assert np.allclose(samples[jj][0], samples[jj-1][1])

    def test_rsquared_mfmc(self):
        functions = ShortColumnModelEnsemble()
        model_ensemble = ModelEnsemble(
            [functions.m0, functions.m3, functions.m4])
        univariate_variables = [
            stats.uniform(5, 10), stats.uniform(15, 10), stats.norm(500, 100),
            stats.norm(2000, 400), stats.lognorm(s=0.5, scale=np.exp(5))]
        variable = IndependentMarginalsVariable(
            univariate_variables)
        generate_samples = partial(
            generate_independent_random_samples, variable)
        npilot_samples = int(1e4)
        pilot_samples = generate_samples(npilot_samples)
        config_vars = np.arange(model_ensemble.nmodels)[np.newaxis, :]
        pilot_samples = get_all_sample_combinations(
            pilot_samples, config_vars)
        pilot_values = model_ensemble(pilot_samples)
        pilot_values = np.reshape(
            pilot_values, (npilot_samples, model_ensemble.nmodels))
        cov = np.cov(pilot_values, rowvar=False)

        nhf_samples = 10
        nsample_ratios = np.asarray([2, 4])

        nsamples_per_model = np.concatenate(
            [[nhf_samples], nsample_ratios*nhf_samples])

        eta = get_mfmc_control_variate_weights(cov)
        cor = get_correlation_from_covariance(cov)
        var_mfmc = cov[0, 0]/nsamples_per_model[0]
        for k in range(1, model_ensemble.nmodels):
            var_mfmc += (1/nsamples_per_model[k-1]-1/nsamples_per_model[k])*(
                eta[k-1]**2*cov[k, k]+2*eta[k-1]*cor[0, k]*np.sqrt(
                    cov[0, 0]*cov[k, k]))

        assert np.allclose(var_mfmc/cov[0, 0]*nhf_samples,
                           1-get_rsquared_mfmc(cov, nsample_ratios))

    def check_variance(self, estimator_type, setup_model,
                       target_cost, ntrials, rtol, kwargs={}):
        max_eval_concurrency = 1

        model_ensemble, cov, costs, variable = setup_model()
        estimator = get_estimator(
            estimator_type, cov, costs, variable, **kwargs)

        means, numerical_var, true_var = \
            estimate_variance(
                model_ensemble, estimator, target_cost, ntrials, None,
                max_eval_concurrency)

        # from pyapprox.multifidelity.control_variate_monte_carlo import plot_sample_allocation
        # from pyapprox.util.configure_plots import plt
        # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # plot_sample_allocation(
        #     estimator._get_reordered_sample_allocation_matrix(),
        #     estimator._get_npartition_samples(estimator.nsamples_per_model),
        #     ax)
        # plt.show()

        print('true red', true_var, 'numerical red',
              numerical_var)
        print(np.absolute(true_var-numerical_var),
              rtol*np.absolute(true_var))
        assert np.allclose(numerical_var, true_var, rtol=rtol)

    def test_variance_reduction(self):
        ntrials = 1e4
        setup_model = setup_model_ensemble_tunable
        for estimator_type in ["acvis", "acvmf", "mfmc"]:
            self.check_variance(
                estimator_type, setup_model, 1e3, ntrials, 2e-2)

        setup_model = setup_model_ensemble_polynomial
        for estimator_type in ["mlmc"]:
            self.check_variance(
                estimator_type, setup_model, 1e3, ntrials, 1e-2)

        setup_model = setup_model_ensemble_tunable
        estimator_type = "acvgmf"
        for index in list(get_acv_recursion_indices(3)):
            print(index)
            self.check_variance(
                estimator_type, setup_model, 1e3, ntrials, 2e-2,
                {"recursion_index": index})

    def test_variance_reduction_acvgmf(self):
        estimator_type = "acvgmf"
        target_cost = 1e3
        ntrials = 1e4
        setup_model = setup_model_ensemble_polynomial
        model_ensemble, cov, costs, variable = setup_model()
        nmodels = cov.shape[0]
        KL_sets = [[4, 1], [3, 1], [3, 2], [3, 3], [2, 1], [2, 2]]
        for K, L in KL_sets:
            if K == nmodels-1:
                recursion_index = np.zeros(nmodels-1, dtype=int)
            else:
                recursion_index = np.hstack(
                    (np.zeros(K), np.ones(nmodels-1-K)*L)).astype(int)
            print(K, L, recursion_index)
            estimator = get_estimator(
                estimator_type, cov, costs, variable,
                recursion_index=recursion_index)
            nsample_ratios, variance, rounded_target_cost = \
                estimator.allocate_samples(target_cost)
            nsamples_per_model = estimator.get_nsamples_per_model(
                rounded_target_cost, nsample_ratios)
            print(variance, rounded_target_cost, nsamples_per_model)

            samples, values = estimator.generate_data(model_ensemble)
            # Check sizes of samples allocated to each model are correct
            for ii in range(1, K+1):
                assert values[ii][0].shape[0] == nsamples_per_model[0]
            for ii in range(K+1, nmodels):
                assert values[ii][0].shape[0] == values[L][1].shape[0]
            for ii in range(1, K+1):
                print(values[ii][1].shape[0], nsamples_per_model[ii], ii)
                assert values[ii][1].shape[0] == nsamples_per_model[ii]
            for ii in range(K+1, nmodels):
                assert values[ii][1].shape[0] == values[ii][1].shape[0]

            self.check_variance(
                estimator_type, setup_model, target_cost, ntrials, 3e-2,
                {"recursion_index": recursion_index})

    def test_acv_sample_allocation_nhf_samples_constraint_jac(self):
        setup_model = setup_model_ensemble_polynomial
        model_ensemble, cov, costs, variable = setup_model()
        x0 = np.arange(2, model_ensemble.nmodels+1)
        target_cost = 1e4

        def obj(x):
            val = np.atleast_2d(acv_sample_allocation_nhf_samples_constraint(
                x[:, 0], target_cost, costs))
            return val

        def jac(x):
            jac = acv_sample_allocation_nhf_samples_constraint_jac(
                x[:, 0], target_cost, costs)[None, :]
            return jac

        errors = check_gradients(obj, jac, x0[:, None], disp=True)
        assert errors.max() > 1e-2 and errors.min() < 1e-7

    def test_acv_sample_allocation_gmf_ratio_constraint(self):
        setup_model = setup_model_ensemble_polynomial
        model_ensemble, cov, costs, variable = setup_model()
        x0 = np.arange(2, model_ensemble.nmodels+1)
        target_cost = 1e4

        recursion_index = [0, 1, 0, 2]
        for idx in range(1, model_ensemble.nmodels):
            parent_idx = recursion_index[idx-1]
            if parent_idx == 0:
                continue

            def obj(x):
                val = np.atleast_2d(
                    acv_sample_allocation_gmf_ratio_constraint(
                        x[:, 0], idx, parent_idx, target_cost, costs))
                return val

            def jac(x):
                jac = acv_sample_allocation_gmf_ratio_constraint_jac(
                    x[:, 0], idx, parent_idx, target_cost, costs)[None, :]
                return jac

            errors = check_gradients(obj, jac, x0[:, None], disp=True)
            print(errors.max(), errors.min())
            assert errors.max() > 1e-3 and errors.min() < 1e-8

    def test_acv_sample_allocation_nlf_gt_nhf_ratio_constraint(self):
        setup_model = setup_model_ensemble_polynomial
        model_ensemble, cov, costs, variable = setup_model()
        x0 = np.arange(2, model_ensemble.nmodels+1)
        target_cost = 1e3

        for idx in range(1, model_ensemble.nmodels):
            def obj(x):
                val = np.atleast_2d(
                    acv_sample_allocation_nlf_gt_nhf_ratio_constraint(
                        x[:, 0], idx, target_cost, costs))
                return val

            def jac(x):
                jac = acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac(
                    x[:, 0], idx, target_cost, costs)[None, :]
                return jac

            errors = check_gradients(obj, jac, x0[:, None], disp=True)
            print(errors.max(), errors.min())
            assert errors.max() > 1e-3 and errors.min() < 3e-8

    @skiptest
    def test_allocate_samples_mlmc_lagrange_formulation(self):
        cov = np.asarray([[1.00, 0.50, 0.25],
                          [0.50, 1.00, 0.50],
                          [0.25, 0.50, 1.00]])
        np.linalg.cholesky(cov)
        costs = np.array([6, 3, 1])
        target_cost = 81
        variable = IndependentMarginalsVariable([stats.uniform(0, 1)])
        estimator = get_estimator("mlmc", cov, costs, variable)

        (nsample_ratios_exact, log10_var) = allocate_samples_mlmc(
            cov, costs, target_cost)
        variance = 10**log10_var
        nsamples_per_model = get_nsamples_per_model(
            target_cost, costs, nsample_ratios_exact, False)
        estimator_cost = nsamples_per_model.dot(estimator.costs)
        assert np.allclose(estimator_cost, target_cost, rtol=1e-12)

        lagrange_mult_exact = get_lagrange_multiplier_mlmc(
            cov, costs, target_cost, variance)
        print('lagrange_mult', lagrange_mult_exact)

        nhf_samples = target_cost/(
            costs[0]+(nsample_ratios_exact*costs[1:]).sum())
        x0 = np.concatenate([[nhf_samples], nsample_ratios_exact,
                             [lagrange_mult_exact]])

        lagrangian = partial(
            mlmc_sample_allocation_objective_all_lagrange, estimator,
            variance, _ndarray_as_pkg_format(costs))
        lagrangian_jac = partial(
            mlmc_sample_allocation_jacobian_all_lagrange_torch,
            estimator, variance, _ndarray_as_pkg_format(costs))
        assert np.allclose(
            approx_jacobian(
                lambda x: np.atleast_1d(lagrangian(x[:, 0])), x0[:, None]),
            lagrangian_jac(x0))

        # The critical points of Lagrangians occur at saddle points, rather
        # than at local maxima (or minima). So must transform so that
        # critical points occur at local minima
        def objective(x):
            return np.sum(lagrangian_jac(x)**2)

        factor = 1-1e-2
        initial_guess = x0*np.random.uniform(factor, 1/factor, x0.shape[0])

        nmodels = len(costs)
        cons = []
        bounds = [(1.000, np.inf)]*(nmodels)+[(-np.inf, np.inf)]
        from scipy.optimize import minimize
        res = minimize(objective, initial_guess, method='SLSQP', jac=None,
                       bounds=bounds, constraints=cons,
                       options={"iprint": 2, "disp": True, "ftol": 1e-16,
                                "maxiter": 1000})
        # print(jacobian(res.x), 'jac')
        nhf_samples, ratios, lagrange_mult = res.x[0], res.x[1:-1], res.x[-1]
        # print(estimator.get_variance(nhf_samples, ratios), variance)
        # print(lagrange_mult_exact, lagrange_mult)
        # print(nsample_ratios_exact, ratios)
        assert np.allclose(nsample_ratios_exact, ratios)
        assert np.allclose(lagrange_mult_exact, lagrange_mult)

    @skiptest
    def test_ACVMF_objective_jacobian(self):

        cov = np.asarray([[1.00, 0.50, 0.25],
                          [0.50, 1.00, 0.50],
                          [0.25, 0.50, 4.00]])

        costs = [4, 2, 1]

        target_cost = 20

        nsample_ratios = allocate_samples_mlmc(
            cov, costs, target_cost)[0]

        print(nsample_ratios)
        variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*3)
        estimator = get_estimator("acvmf", cov, costs, variable)
        factor = 1-0.1
        x0 = (nsample_ratios*np.random.uniform(
            factor, 1/factor, len(nsample_ratios)))[:, np.newaxis]

        def obj(x):
            val, grad = acv_sample_allocation_objective_all(
                estimator, target_cost, x, True)
            return np.atleast_2d(val), grad.T
        errors = check_gradients(obj, True, x0, disp=True)
        # print(errors.min())
        assert errors.max() > 1e-2 and errors.min() < 9e-8

    @skiptest
    def test_MLMC_objective_jacobian_all(self):

        cov = np.asarray([[1.00, 0.50, 0.25],
                          [0.50, 1.00, 0.50],
                          [0.25, 0.50, 4.00]])

        costs = np.array([6., 3., 1.])

        target_cost = 81
        nsample_ratios = allocate_samples_mlmc(
            cov, costs, target_cost)[0]

        variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*3)
        estimator = get_estimator("mlmc", cov, costs, variable)

        factor = 1-0.1
        x0 = (nsample_ratios*np.random.uniform(
            factor, 1/factor, len(nsample_ratios)))[:, np.newaxis]

        def obj(x):
            val, grad = acv_sample_allocation_objective_all(
                estimator, target_cost, x, True)
            return np.atleast_2d(val), grad.T
        errors = check_gradients(obj, True, x0, disp=True)
        # print(errors.min())
        assert errors.max() > 6e-1 and errors.min() < 3e-7

    def test_bootstrap_monte_carlo_estimator(self):
        nsamples = int(1e4)
        nbootstraps = int(1e3)
        values = np.random.normal(1., 1., (nsamples, 1))
        est_variance = np.var(values)/nsamples
        bootstrap_mean, bootstrap_variance = \
            bootstrap_monte_carlo_estimator(values, nbootstraps)
        # print(abs(est_variance-bootstrap_variance)/est_variance)
        assert abs((est_variance-bootstrap_variance)/est_variance) < 1e-2

    def test_bootstrap_approximate_control_variate_estimator(self):
        example = TunableModelEnsemble(np.pi/3)
        model_costs = [1, 0.5, 0.4]
        target_cost = 1000
        # example = PolynomialModelEnsemble()
        # model_costs = 10.**(-np.arange(example.nmodels))
        # target_cost = 100
        model_ensemble = ModelEnsemble(example.models)

        cov_matrix = example.get_covariance_matrix()
        est = get_estimator("acvgmf", cov_matrix, model_costs,
                            example.variable)
        est.set_recursion_index(np.array([0, 1]))
        est.set_initial_guess(np.arange(2, est.nmodels + 1))
        est.allocate_samples(target_cost)

        samples_per_model, partition_indices_per_model = \
            est.generate_sample_allocations()
        values_per_model = []
        for ii in range(est.nmodels):
            values_per_model.append(
                model_ensemble.functions[ii](samples_per_model[ii]))

        bootstrap_mean, bootstrap_variance = est.bootstrap(
            values_per_model, partition_indices_per_model, 10000)

        print(example.get_means()[0], bootstrap_mean)
        print(est.nsamples_per_model, est.rounded_target_cost,
              est.nsample_ratios)
        est_variance = est.get_variance(
            est.rounded_target_cost, est.nsample_ratios)
        print(est_variance, bootstrap_variance, 'var')
        print(abs((est_variance-bootstrap_variance)/est_variance))
        assert abs((est_variance-bootstrap_variance)/est_variance) < 6e-2

    def test_mfmc_estimator_optimization(self):
        target_cost = 30
        cov = np.asarray([[1.00, 0.90, 0.85],
                          [0.90, 1.00, 0.50],
                          [0.85, 0.50, 1.10]])
        costs = [4, 2, .5]
        variable = IndependentMarginalsVariable([stats.uniform(0, 1)])

        recursion_index = np.array([0, 1])
        estimator = get_estimator(
            "acvgmf", cov, costs, variable, recursion_index=recursion_index)
        nsample_ratios, variance, rounded_target_cost = \
            estimator.allocate_samples(target_cost)

        mfmc_estimator = get_estimator("mfmc", cov, costs, variable)
        (mfmc_nsample_ratios, mfmc_variance,
         mfmc_rounded_target_cost) = mfmc_estimator.allocate_samples(
             target_cost)
        assert np.allclose(
            mfmc_estimator.get_variance(rounded_target_cost, nsample_ratios),
            estimator.get_variance(rounded_target_cost, nsample_ratios))

        recursion_index = np.array([0, 0])
        estimator = get_estimator(
            "acvgmf", cov, costs, variable, recursion_index=recursion_index)
        nsample_ratios, variance, rounded_target_cost = \
            estimator.allocate_samples(target_cost)
        acvmf_estimator = get_estimator("acvmf", cov, costs, variable,)
        (acvmf_nsample_ratios, acvmf_variance,
         acvmf_rounded_target_cost) = acvmf_estimator.allocate_samples(
             target_cost)
        assert np.allclose(
            acvmf_estimator.get_variance(rounded_target_cost, nsample_ratios),
            estimator.get_variance(rounded_target_cost, nsample_ratios))

    @skiptest
    def test_estimator_objective_jacobian(self):
        target_cost = 20
        cov = np.asarray([[1.00, 0.90, 0.85],
                          [0.90, 1.00, 0.50],
                          [0.85, 0.50, 1.10]])
        np.linalg.cholesky(cov)
        costs = np.array([4, 2, .5])

        recursion_index = np.array([0, 1])
        variable = IndependentMarginalsVariable([stats.uniform(0, 1)])
        estimator = get_estimator("acvgmf", cov, costs, variable)
        estimator.set_recursion_index(recursion_index)
        x0 = get_acv_initial_guess(None, cov, costs, target_cost)

        def obj(x):
            val, jac = estimator.objective(target_cost, x[:, 0])
            return np.atleast_2d(val), jac
        errors = check_gradients(obj, True, x0[:, None])
        assert errors.min() < 3e-7 and errors.max() > 1e-1

        target_cost = 1e3
        setup_model = setup_model_ensemble_polynomial
        model_ensemble, cov, costs, variable = setup_model()
        recursion_index = np.zeros(len(costs)-1, dtype=int)
        estimator = get_estimator(
            "acvgmf", cov, costs, variable,
            recursion_index=recursion_index)
        x0 = get_acv_initial_guess(None, cov, costs, target_cost)

        def obj(x):
            val, jac = estimator.objective(target_cost, x[:, 0])
            return np.atleast_2d(val), jac
        errors = check_gradients(obj, True, x0[:, None])
        assert errors.min() < 3e-7 and errors.max() > 1e-1

    def test_get_acvgmf_recusion_indices(self):
        nmodels = 4
        indices = []
        for index in get_acv_recursion_indices(nmodels, 2):
            indices.append(list(index))
        assert len(indices) == 10

        true_indices = [
            [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1],
            [2, 0, 0], [3, 3, 0], [0, 3, 0], [3, 0, 0],
            [0, 1, 0], [2, 0, 2], [2, 3, 0], [0, 1, 2],
            [3, 0, 2], [2, 0, 1], [3, 1, 0], [0, 3, 1]]
        for tindex in true_indices[:10]:
            assert tindex in indices

        indices = []
        for index in get_acv_recursion_indices(nmodels, nmodels-1):
            indices.append(list(index))
        assert len(indices) == 16
        for tindex in true_indices:
            assert tindex in indices

        # from pyapprox.multifidelity.control_variate_monte_carlo import plot_model_recursion
        # from pyapprox.util.configure_plots import plt
        # ngraphs = len(indices)
        # nrows = int(np.ceil(ngraphs/8))
        # ncols = int(np.ceil(ngraphs/nrows))
        # fig, axs = plt.subplots(nrows, ncols, figsize=(3*8, nrows*4))
        # axs = axs.flatten()
        # for ii, index in enumerate(indices):
        #     plot_model_recursion(index, axs[ii])
        # for ii in range(len(indices), len(axs)):
        #     axs[ii].remove()
        # plt.tight_layout()
        # plt.savefig("graph.png")
        # plt.show()

    def test_MLBLUE(self):
        # from pyapprox.multifidelity.multilevelblue import (
        #     BLUE_cost_constraint, BLUE_cost_constraint_jac)
        # nsubsets = 3
        # nsamples_per_subset = np.ones(nsubsets)[:, None]
        # errors = check_gradients(
        #     BLUE_cost_constraint,
        #     lambda x: BLUE_cost_constraint_jac(x).T,
        #     nsamples_per_subset)

        target_cost = 1e3
        shifts = np.array([1, 2])
        model, cov, costs, variable = setup_model_ensemble_tunable(shifts)
        estimator = get_estimator(
            "mlblue", cov, costs, variable)
        asketch = np.zeros((costs.shape[0], 1))
        asketch[0] = 1.0
        result = estimator.allocate_samples(
            target_cost, asketch, round_nsamples=False)
        assert np.allclose(result[2], target_cost)

        subset_costs, subsets = estimator._get_model_subset_costs(
            costs)

        # round nsamples because it was not done before to allow testing of
        # cost constraint
        estimator.nsamples_per_subset = np.asarray(
            estimator.nsamples_per_subset).astype(int)
        # rounded_target_cost = np.sum(
        #     estimator.nsamples_per_subset*subset_costs)

        # test equivalent formulations for variance
        betas = BLUE_betas(cov, asketch, 1e-10, estimator.nsamples_per_subset)
        tmp = np.array(
            [np.linalg.multi_dot((betas[kk].T, cov, betas[kk]))
             for kk in range(betas.shape[0])])
        tmp[estimator.nsamples_per_subset > 0] /= (
            estimator.nsamples_per_subset[estimator.nsamples_per_subset > 0])
        variance = tmp.sum()
        assert np.allclose(
            variance,
            BLUE_variance(asketch, cov, None, 1e-10,
                          estimator.nsamples_per_subset))

        assert np.allclose(
            (estimator.nsamples_per_model*estimator.costs).sum(),
            estimator.rounded_target_cost)

        values = estimator.generate_data(model.functions, variable)
        for ii in range(model.nmodels):
            asketch = np.zeros((costs.shape[0], 1))
            asketch[ii] = 1.0
            # true_means = np.hstack((0, shifts))
            # mean = estimator(values, asketch)
            # print("Mean", mean, true_means[ii])
            # assert np.allclose(mean, true_means[ii], atol=2e-2)
            true_var = estimator.get_variance(
                estimator.nsamples_per_subset, asketch)
            print("true Var", true_var)

            ntrials = int(1e4)
            means = np.empty(ntrials)
            for ii in range(ntrials):
                values = estimator.generate_data(model.functions, variable)
                means[ii] = estimator(values, asketch)
            numerical_var = means.var()
            rtol = 4e-2
            # print("Empirical Var", numerical_var)
            # print(np.absolute(true_var-numerical_var),
            #       rtol*np.absolute(true_var))
            assert np.allclose(numerical_var, true_var, rtol=rtol)
            # bootstrapped_means = estimator.bootstrap_estimator(
            #     values, asketch, ntrials)
            # print("Bootstrapped Var", bootstrapped_means.var())

    def test_AETC_optimal_loss(self):
        alpha = 1000
        nsamples = int(1e6)
        shifts = np.array([1, 2])
        model, cov, costs, variable = setup_model_ensemble_tunable(shifts)
        target_cost = np.sum(costs)*(nsamples+10)

        true_means = np.hstack((0, shifts))[:, None]
        oracle_stats = [cov, true_means]

        samples = variable.rvs(nsamples)
        values = np.hstack([m(samples) for m in model.functions])

        covariate_subset = np.asarray([0, 1])
        hf_values = values[:, :1]
        covariate_values = values[:, covariate_subset+1]
        from pyapprox.multifidelity.multilevelblue import AETC_optimal_loss
        result_oracle = AETC_optimal_loss(
            target_cost, hf_values, covariate_values, costs, covariate_subset,
            alpha, 0, 0, oracle_stats)
        result_mc = AETC_optimal_loss(
            target_cost, hf_values, covariate_values, costs, covariate_subset,
            alpha, 0, 0, None)
        # for r in result_oracle:
        #     print(r)
        #     print("##")
        # for r in result_mc:
        #     print(r)
        # assert np.allclos
        assert np.allclose(result_mc[-2], result_oracle[-2], rtol=1e-2)

    def test_aetc_blue(self):
        target_cost = 1e4
        shifts = np.array([1, 2])
        model, cov, costs, variable = setup_model_ensemble_tunable(shifts)

        true_means = np.hstack((0, shifts))[:, None]
        oracle_stats = [cov, true_means]
        # oracle_stats = None

        # provide oracle covariance so numerical and theoretical estimates
        # will coincide
        estimator = AETCBLUE(
            model.functions, variable.rvs, costs,  oracle_stats, 1e-12, 0)
        mean, values, result = estimator.estimate(
            target_cost, return_dict=False)
        result_dict = estimator._explore_result_to_dict(result)
        print(result_dict)

        from pyapprox.multifidelity.multilevelblue import BLUE_variance
        true_var = BLUE_variance(
            result_dict["beta_Sp"][1:],
            cov[np.ix_(result_dict["subset"]+1, result_dict["subset"]+1)],
            None, estimator._reg_blue, result_dict["nsamples_per_subset"])

        ntrials = int(1e3)
        means = np.empty(ntrials)
        for ii in range(ntrials):
            means[ii] = estimator.exploit(result)[0]
        numerical_var = means.var()
        print(numerical_var, "NV")
        print(true_var, "TV")
        assert np.allclose(numerical_var, true_var, rtol=3e-2)
        assert np.allclose(
            result_dict["BLUE_variance"]/result_dict["exploit_budget"],
            true_var, rtol=3e-2)

        ntrials = int(1e3)
        means = np.empty(ntrials)
        for ii in range(ntrials):
            means[ii] = estimator.estimate(target_cost)[0]
        true_mean = 0
        mse = np.mean((means-true_mean)**2)
        print(mse, result_dict["loss"])
        # this is just a regression test to make sure this does not get worse
        # when code is changed
        assert np.allclose(mse, result_dict["loss"], rtol=2e-1)



if __name__ == "__main__":
    cvmc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestCVMC)
    unittest.TextTestRunner(verbosity=2).run(cvmc_test_suite)
