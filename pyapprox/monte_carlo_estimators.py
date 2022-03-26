from abc import ABC, abstractmethod
import numpy as np
from functools import partial

from pyapprox.utilities import get_correlation_from_covariance
from pyapprox.low_discrepancy_sequences import sobol_sequence, halton_sequence
from pyapprox.control_variate_monte_carlo import (
    compute_approximate_control_variate_mean_estimate,
    get_rsquared_mlmc, allocate_samples_mlmc, get_mlmc_control_variate_weights,
    generate_samples_and_values_mlmc, get_rsquared_mfmc, allocate_samples_mfmc,
    get_mfmc_control_variate_weights, generate_samples_and_values_mfmc,
    acv_sample_allocation_objective_all, get_nsamples_per_model,
    allocate_samples_acv, get_approximate_control_variate_weights,
    get_discrepancy_covariances_MF, get_discrepancy_covariances_KL,
    get_rsquared_acv_KL_best, allocate_samples_acv_best_kl,
    generate_samples_and_values_acv_KL, round_nsample_ratios,
    check_mfmc_model_costs_and_correlations, pkg, use_torch, get_nhf_samples,
    generate_samples_and_values_acv_IS, get_discrepancy_covariances_IS,
    get_sample_allocation_matrix_acvmf, acv_estimator_variance,
    generate_samples_acv, separate_model_values_acv,
    get_npartition_samples_acvmf, get_rsquared_acv,
    _ndarray_as_pkg_format, reorder_allocation_matrix_acvgmf,
    acv_sample_allocation_gmf_ratio_constraint,
    acv_sample_allocation_gmf_ratio_constraint_jac,
    acv_sample_allocation_nlf_gt_nhf_ratio_constraint,
    acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac,
    acv_sample_allocation_nhf_samples_constraint,
    acv_sample_allocation_nhf_samples_constraint_jac,
    get_generalized_approximate_control_variate_weights
)


class AbstractMonteCarloEstimator(ABC):
    def __init__(self, cov, costs, variable, sampling_method="random"):
        """
        Constructor.

        Parameters
        ----------
        cov : np.ndarray (nmodels,nmodels)
            The covariance C between each of the models. The highest fidelity
            model is the first model, i.e its variance is cov[0,0]

        costs : np.ndarray (nmodels)
            The relative costs of evaluating each model

        variable : :class:`pyapprox.variables.IndependentMultivariateRandomVariable`
            The uncertain model parameters

        sampling_method : string
            Supported types are ["random", "sobol", "halton"]
        """
        if cov.shape[0] != len(costs):
            raise ValueError("cov and costs are inconsistent")

        self.cov = cov.copy()
        self.costs = np.array(costs)
        self.variable = variable
        self.sampling_method = sampling_method
        self.set_random_state(None)
        self.nmodels = len(costs)

        self.cov_opt = self.cov
        self.costs = self.costs
        if use_torch:
            if not pkg.is_tensor(self.cov):
                self.cov_opt = pkg.tensor(self.cov, dtype=pkg.double)
            if not pkg.is_tensor(self.costs):
                self.costs_opt = pkg.tensor(self.costs, dtype=pkg.double)

    def set_sampling_method(self):
        sampling_methods = {
            "random": partial(
                self.variable.rvs, random_state=self.random_state),
            "sobol": partial(sobol_sequence, self.variable.num_vars(),
                             start_index=0, variable=self.variable),
            "halton": partial(halton_sequence, self.variable.num_vars(),
                              start_index=0, variable=self.variable)}
        if self.sampling_method not in sampling_methods:
            msg = f"{self.sampling_method} not supported"
            raise ValueError(msg)
        self.generate_samples = sampling_methods[self.sampling_method]

    def set_random_state(self, random_state):
        """
        Set the state of the numpy random generator. This effects
        self.generate_samples

        Parameters
        ----------
        random_state : :class:`numpy.random.RandmState`
            Set the random state of the numpy random generator

        Notes
        -----
        To create reproducible results when running numpy.random in parallel
        must use RandomState. If not the results will be non-deterministic.
        This is happens because of a race condition. numpy.random.* uses only
        one global PRNG that is shared across all the threads without
        synchronization. Since the threads are running in parallel, at the same
        time, and their access to this global PRNG is not synchronized between
        them, they are all racing to access the PRNG state (so that the PRNG's
        state might change behind other threads' backs). Giving each thread its
        own PRNG (RandomState) solves this problem because there is no longer
        any state that's shared by multiple threads without synchronization.
        Also see new features
        https://docs.scipy.org/doc/numpy/reference/random/parallel.html
        https://docs.scipy.org/doc/numpy/reference/random/multithreading.html
        """
        self.random_state = random_state
        self.set_sampling_method()

    def _variance_reduction(self, cov, nsample_ratios):
        """
        Get the variance reduction of the Monte Carlo estimator

        Parameters
        ----------
        cov : np.ndarray or torch.tensor (nmodels, nmodels)
            The covariance C between each of the models. The highest fidelity
            model is the first model, i.e its variance is cov[0,0]

        nsample_ratios : np.ndarray (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

        Returns
        -------
        var_red : float
            The variance redution

        Notes
        -----
        This is not the variance reduction relative to the equivalent
        Monte Carlo estimator. A variance reduction can be smaller than
        one and still correspond to a multi-fidelity estimator that
        has a larger variance than the single fidelity Monte Carlo
        that uses the equivalent number of high-fidelity samples
        """
        return 1-self._get_rsquared(cov, nsample_ratios)

    def get_nmodels(self):
        """
        Return the number of models used by the estimator

        Returns
        -------
        nmodels : integer
            The number of models
        """
        return self.cov.shape[0]

    def _get_variance(self, target_cost, costs, cov, nsample_ratios):
        """
        Get the variance of the Monte Carlo estimator from costs and cov.

        Parameters
        ----------
        target_cost : float
            The total cost budget

        costs : np.ndarray or torch.tensor (nmodels, nmodels) (nmodels)
            The relative costs of evaluating each model

        cov : np.ndarray or torch.tensor (nmodels, nmodels)
            The covariance C between each of the models. The highest fidelity
            model is the first model, i.e its variance is cov[0,0]

        nsample_ratios : np.ndarray or torch.tensor (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

        Returns
        -------
        variance : float
            The variance of the estimator
            """
        nhf_samples = get_nhf_samples(
            target_cost, costs, nsample_ratios)
        var_red = self._variance_reduction(cov, nsample_ratios)
        variance = var_red*cov[0, 0]/nhf_samples
        return variance

    def _get_variance_for_optimizer(self, target_cost, nsample_ratios):
        """
        Get the variance of the Monte Carlo estimator from costs and cov in
        the format required by the optimizer used to allocate samples

        Parameters
        ----------
        target_cost : float
            The total cost budget

        costs : np.ndarray or torch.tensor (nmodels, nmodels) (nmodels)
            The relative costs of evaluating each model

        cov : np.ndarray or torch.tensor (nmodels, nmodels)
            The covariance C between each of the models. The highest fidelity
            model is the first model, i.e its variance is cov[0,0]

        nsample_ratios : np.ndarray or torch.tensor (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

        Returns
        -------
        variance : float
            The variance of the estimator
        """
        return self._get_variance(
            target_cost, self.costs_opt, self.cov_opt, nsample_ratios)

    def get_variance(self, target_cost, nsample_ratios):
        """
        Get the variance of the Monte Carlo estimator.

        Parameters
        ----------
        target_cost : float
            The total cost budget

        nsample_ratios : np.ndarray (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

        Returns
        -------
        variance : float
            The variance of the estimator
        """
        return self._get_variance(
            target_cost, self.costs, self.cov, nsample_ratios)

    def get_covariance(self):
        """
        Return the covariance between the models.

        Returns
        -------
        cov : np.ndarray (nmodels, nmodels)
            The covariance between the models
        """
        return self.cov

    def get_model_costs(self):
        """
        Return the cost of each model.

        Returns
        -------
        costs : np.ndarray (nmodels)
            The cost of each model
        """
        return self.costs

    def get_nsamples_per_model(self, target_cost, nsample_ratios):
        """
        Get the number of samples allocated to each model

        Parameters
        ----------
        target_cost : float
            The total cost budget

        nsample_ratios : np.ndarray (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

        Returns
        -------
        nsamples_per_model : np.ndarray (nsamples)
            The number of samples allocated to each model
        """
        return get_nsamples_per_model(target_cost, self.costs, nsample_ratios,
                                      True)

    @abstractmethod
    def _get_rsquared(self, cov, nsample_ratios):
        r"""
        Compute r^2 used to compute the variance reduction of the Monte Carlo
        like estimator from a provided covariance. This is useful when
        optimizer is using cov as a torch.tensor

        Parameters
        ----------
        cov : np.ndarray (nmodels, nmodels)
            The covariance between the models

        nsample_ratios : np.ndarray (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

        Returns
        -------
        rsquared : float
            The value r^2
        """
        # Notes: cov must be an argument so we can pass in torch.tensor when
        # computing gradient and a np.ndarray in all other cases
        raise NotImplementedError()

    def _estimate(self, values):
        r"""
        Return the value of the Monte Carlo like estimator

        Parameters
        ----------
        values : list (nmodels)
        Each entry of the list contains

        values0 : np.ndarray (num_samples_i0,num_qoi)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

        values1: np.ndarray (num_samples_i1,num_qoi)
            Evaluations used compute the approximate
            mean :math:`\mu_{i,r_iN}` of the low fidelity models.

        Returns
        -------
        est : float
            The estimate of the mean
        """
        eta = self._get_approximate_control_variate_weights()
        return compute_approximate_control_variate_mean_estimate(eta, values)

    def __call__(self, values):
        r"""
        Return the value of the Monte Carlo like estimator

        Parameters
        ----------
        values : list (nmodels)
        Each entry of the list contains

        values0 : np.ndarray (num_samples_i0,num_qoi)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

        values1: np.ndarray (num_samples_i1,num_qoi)
            Evaluations used compute the approximate
            mean :math:`\mu_{i,r_iN}` of the low fidelity models.

        Returns
        -------
        est : float
            The estimate of the mean
        """
        return self._estimate(values)

    @abstractmethod
    def _allocate_samples(self, target_cost):
        """
        Determine the samples (each a float) that must be allocated to
        each model to compute the Monte Carlo like estimator

        Parameters
        ----------
        target_cost : float
            The total cost budget

        Returns
        -------
        nsample_ratios : np.ndarray (nmodels-1, dtype=float)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
            the number of samples in the two different discrepancies involving
            the ith model.

        log10_variance : float
            The base 10 logarithm of the variance of the estimator using the
            float sample allocations
        """
        raise NotImplementedError()

    def allocate_samples(self, target_cost):
        """
        Determine the samples (integers) that must be allocated to
        each model to compute the Monte Carlo like estimator

        Parameters
        ----------
        target_cost : float
            The total cost budget

        Returns
        -------
        nsample_ratios : np.ndarray (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
            the number of samples in the two different discrepancies involving
            the ith model.

        variance : float
            The variance of the estimator using the integer sample allocations

        rounded_target_cost : float
            The cost of the new sample allocation
        """
        nsample_ratios, log10_var = self._allocate_samples(target_cost)
        if use_torch and pkg.is_tensor(nsample_ratios):
            nsample_ratios = nsample_ratios.detach().numpy()
        nsample_ratios, rounded_target_cost = round_nsample_ratios(
            target_cost, self.costs, nsample_ratios)
        variance = self.get_variance(rounded_target_cost, nsample_ratios)
        self.set_optimized_params(nsample_ratios, rounded_target_cost)
        return (nsample_ratios, variance, rounded_target_cost)

    def set_optimized_params(self, nsample_ratios, rounded_target_cost):
        """
        Set the parameters needed to generate samples for evaluating the
        estimator

        nsample_ratios : np.ndarray (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
            the number of samples in the two different discrepancies involving
            the ith model.

        rounded_target_cost : float
            The cost of the new sample allocation
        """
        self.nsample_ratios = nsample_ratios
        self.rounded_target_cost = rounded_target_cost
        self.nsamples_per_model = get_nsamples_per_model(
            self.rounded_target_cost, self.costs, self.nsample_ratios,
            True).astype(int)

    @abstractmethod
    def generate_data(self, functions):
        r"""
        Generate the samples and values needed to compute the Monte Carlo like
        estimator.

        Parameters
        ----------
        functions : list of callables
            The functions used to evaluate each model with signature

            `function(samples)->np.ndarray (nsamples, 1)`

            whre samples : np.ndarray (nvars, nsamples)

        generate_samples : callable
            Function used to generate realizations of the random variables

        Returns
        -------
        samples : list (nmodels)
            List containing the samples :math:`\mathcal{Z}_{i,1}` and
            :math:`\mathcal{Z}_{i,2}` for each model :math:`i=0,\ldots,M-1`.
            The list is [[:math:`\mathcal{Z}_{0,1}`,:math:`\mathcal{Z}_{0,2}`],...,[:math:`\mathcal{Z}_{M-1,1}`,:math:`\mathcal{Z}_{M-1,2}`]],
            where :math:`M` is the number of models

        values : list (nmodels)
        Each entry of the list contains

        values0 : np.ndarray (num_samples_i0,num_qoi)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

        values1: np.ndarray (num_samples_i1,num_qoi)
            Evaluations used compute the approximate
            mean :math:`\mu_{i,r_iN}` of the low fidelity models.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_approximate_control_variate_weights(self):
        """
        Get the control variate weights corresponding to the parameters
        set by allocate samples of set_optimized_params
        Returns
        -------
        weights : np.ndarray (nmodels-1)
            The control variate weights
        """
        raise NotImplementedError()


class MCEstimator(AbstractMonteCarloEstimator):
    def _get_rsquared(self, cov, nsample_ratios):
        return 0

    def _estimate(self, values):
        return values[0].mean()

    def _allocate_samples(self, target_cost):
        nhf_samples = np.floor(target_cost/self.costs[0])
        nsample_ratios = np.zeros(0)
        log10_variance = np.log10(self.get_variance(
            nhf_samples, nsample_ratios))
        rounded_target_cost = nhf_samples*self.costs[0]
        self.set_optimized_params(nsample_ratios, rounded_target_cost)
        return nsample_ratios, log10_variance

    def generate_data(self, functions):
        samples = self.generate_samples(self.nsamples_per_model[0])
        if not callable(functions[0]):
            values = functions[0](samples)
        else:
            samples_with_id = np.vstack(
                [samples,
                 np.zeros((1, self.nsamples_per_model[0]), dtype=np.double)])
            values = functions(samples_with_id)
        return samples, values

    def _get_approximate_control_variate_weights(self):
        return None


class MLMCEstimator(AbstractMonteCarloEstimator):
    def _get_rsquared(self, cov, nsample_ratios):
        rsquared = get_rsquared_mlmc(cov, nsample_ratios)
        return rsquared

    def _allocate_samples(self, target_cost):
        return allocate_samples_mlmc(self.cov, self.costs, target_cost)

    def generate_data(self, functions):
        return generate_samples_and_values_mlmc(
            self.nsamples_per_model, functions, self.generate_samples)

    def _get_approximate_control_variate_weights(self):
        return get_mlmc_control_variate_weights(self.get_covariance().shape[0])


class MFMCEstimator(AbstractMonteCarloEstimator):
    def __init__(self, cov, costs, variable, sampling_method="random"):
        super().__init__(cov, costs, variable, sampling_method)
        self._check_model_costs_and_correlation(
            get_correlation_from_covariance(self.cov),
            self.costs)

    def _check_model_costs_and_correlation(self, corr, costs):
        models_accetable = check_mfmc_model_costs_and_correlations(costs, corr)
        if not models_accetable:
            msg = "Model correlations and costs cannot be used with MFMC"
            raise ValueError(msg)

    def _get_rsquared(self, cov, nsample_ratios):
        rsquared = get_rsquared_mfmc(cov, nsample_ratios)
        return rsquared

    def _allocate_samples(self, target_cost):
        # nsample_ratios returned will be listed in according to
        # self.model_order which is what self.get_rsquared requires
        return allocate_samples_mfmc(
            self.cov, self.costs, target_cost)

    def _get_approximate_control_variate_weights(self):
        return get_mfmc_control_variate_weights(self.cov)

    def generate_data(self, functions):
        return generate_samples_and_values_mfmc(
            self.nsamples_per_model, functions, self.generate_samples,
            acv_modification=False)


class AbstractACVEstimator(AbstractMonteCarloEstimator):
    def __init__(self, cov, costs, variable, sampling_method="random"):
        super().__init__(cov, costs, variable, sampling_method)
        self.cons = []

    def objective(self, target_cost, x, jac=True):
        # jac argument used for testing with finte difference
        out = acv_sample_allocation_objective_all(
            self, target_cost, x, (use_torch and jac))
        return out

    def _allocate_samples(self, target_cost, **kwargs):
        cons = self.get_constraints(target_cost)
        return allocate_samples_acv(
            self.cov_opt, self.costs, target_cost, self,  cons, **kwargs)

    def get_constraints(self, target_cost):
        cons = [{'type': 'ineq',
                 'fun': acv_sample_allocation_nhf_samples_constraint,
                 'jac': acv_sample_allocation_nhf_samples_constraint_jac,
                 'args': (target_cost, self.costs)}]
        cons += self._get_constraints(target_cost)
        return cons

    @abstractmethod
    def _get_constraints(self, target_cost):
        raise NotImplementedError()

    @abstractmethod
    def _get_approximate_control_variate_weights(self):
        raise NotImplementedError()


class ACVGMFEstimator(AbstractACVEstimator):
    def __init__(self, cov, costs, variable, sampling_method="random",
                 recursion_index=None):
        super().__init__(cov, costs, variable, sampling_method)
        if recursion_index is None:
            recursion_index = np.zeros(self.nmodels-1, dtype=int)
        self.set_recursion_index(recursion_index)

    def _get_constraints(self, target_cost):
        # Must ensure that the samples of any model acting as a recursive
        # control variate has at least one more sample than its parent.
        # Otherwise Fmat will not be invertable sample ratios are rounded to
        # integers. Fmat is not invertable when two or more sample ratios
        # are equal
        cons = [
            {'type': 'ineq',
             'fun': acv_sample_allocation_gmf_ratio_constraint,
             'jac': acv_sample_allocation_gmf_ratio_constraint_jac,
             'args': (ii, jj, target_cost, self.costs)}
            for ii, jj in zip(range(1, self.nmodels), self.recursion_index)
            if jj > 0]
        # Ensure that all low-fidelity models have at least one more sample
        # than high-fidelity model. Otherwise Fmat will not be invertable after
        # rounding to integers
        cons += [
            {'type': 'ineq',
             'fun': acv_sample_allocation_nlf_gt_nhf_ratio_constraint,
             'jac': acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac,
             'args': (ii, target_cost, self.costs)}
            for ii in range(1, self.nmodels)]

        return cons

    def set_recursion_index(self, index):
        if index.shape[0] != self.nmodels-1:
            raise ValueError("index is the wrong shape")
        self.recursion_index = index
        self._create_allocation_matrix()

    def _create_allocation_matrix(self):
        # TODO make this abstract in base class when acvgis is implemented
        self.allocation_mat = get_sample_allocation_matrix_acvmf(
            self.recursion_index)

    def _get_npartition_samples(self, nsamples_per_model):
        # TODO make this abstract in base class when acvgis is implemented
        return get_npartition_samples_acvmf(nsamples_per_model)

    def _get_variance_for_optimizer(self, target_cost, nsample_ratios):
        allocation_mat_opt = _ndarray_as_pkg_format(self.allocation_mat)
        var = acv_estimator_variance(
            allocation_mat_opt, target_cost, self.costs_opt,
            self._get_npartition_samples, self.cov_opt,
            self.recursion_index, nsample_ratios)
        return var

    def get_variance(self, target_cost, nsample_ratios):
        return acv_estimator_variance(
            self.allocation_mat, target_cost, self.costs,
            self._get_npartition_samples, self.cov, self.recursion_index,
            nsample_ratios)

    def _get_rsquared(self, cov, nsample_ratios):
        raise NotImplementedError()

    def generate_data(self, functions):
        npartition_samples = self._get_npartition_samples(
            self.nsamples_per_model)
        reorder_allocation_mat = reorder_allocation_matrix_acvgmf(
            self.allocation_mat, self.nsamples_per_model, self.recursion_index)
        samples_per_model, subset_indices_per_model = generate_samples_acv(
            reorder_allocation_mat, self.nsamples_per_model,
            npartition_samples, self.generate_samples)
        values_per_model = functions.evaluate_models(samples_per_model)
        acv_values = separate_model_values_acv(
            reorder_allocation_mat, values_per_model, subset_indices_per_model)
        return None, acv_values  # TODO remove first return item

    def _get_approximate_control_variate_weights(self):
        return get_generalized_approximate_control_variate_weights(
            self.allocation_mat, self.nsamples_per_model,
            self._get_npartition_samples, self.cov, self.recursion_index)[0]


class ACVMFEstimator(AbstractACVEstimator):
    def _get_rsquared(self, cov, nsample_ratios):
        return get_rsquared_acv(
            cov, nsample_ratios, get_discrepancy_covariances_MF)

    def generate_data(self, functions):
        return generate_samples_and_values_mfmc(
            self.nsamples_per_model, functions, self.generate_samples,
            acv_modification=True)

    def _get_constraints(self, target_cost):
        cons = [
            {'type': 'ineq',
             'fun': acv_sample_allocation_nlf_gt_nhf_ratio_constraint,
             'jac': acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac,
             'args': (ii, target_cost, self.costs)}
            for ii in range(1, self.nmodels)]
        return cons

    def _get_approximate_control_variate_weights(self):
        nsample_ratios = self.nsamples_per_model[1:]/self.nsamples_per_model[0]
        CF, cf = get_discrepancy_covariances_MF(self.cov, nsample_ratios)
        return get_approximate_control_variate_weights(CF, cf)


class ACVISEstimator(AbstractACVEstimator):
    def _get_rsquared(self, cov, nsample_ratios):
        return get_rsquared_acv(
            cov, nsample_ratios, get_discrepancy_covariances_IS)

    def generate_data(self, functions):
        return generate_samples_and_values_acv_IS(
            self.nsamples_per_model, functions, self.generate_samples)

    def _get_approximate_control_variate_weights(self):
        nsample_ratios = self.nsamples_per_model[1:]/self.nsamples_per_model[0]
        CF, cf = get_discrepancy_covariances_IS(self.cov, nsample_ratios)
        return get_approximate_control_variate_weights(CF, cf)


monte_carlo_estimators = {"acvmf": ACVMFEstimator,
                          "acvis": ACVISEstimator,
                          "mfmc": MFMCEstimator,
                          "mlmc": MLMCEstimator,
                          "acvgmf": ACVGMFEstimator}


def get_estimator(estimator_type, cov, costs, variable, **kwargs):
    if estimator_type not in monte_carlo_estimators:
        msg = f"Estimator {estimator_type} not supported"
        msg += f"Must be one of {monte_carlo_estimators.keys()}"
        raise ValueError(msg)
    return monte_carlo_estimators[estimator_type](
        cov, costs, variable, **kwargs)
