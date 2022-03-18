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
    allocate_samples_acv,  get_control_variate_weights, get_rsquared_acv,
    get_discrepancy_covariances_MF, get_discrepancy_covariances_KL,
    get_rsquared_acv_KL_best, allocate_samples_acv_best_kl,
    generate_samples_and_values_acv_KL, round_nsample_ratios,
    check_mfmc_model_costs_and_correlations, pkg, use_torch, get_nhf_samples,
    generate_samples_and_values_acv_IS, get_discrepancy_covariances_IS
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

    def variance_reduction(self, nsample_ratios):
        """
        Get the variance reduction of the Monte Carlo estimator

        Parameters
        ----------
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
        return 1-self._get_rsquared(nsample_ratios)

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
        var_red = self.variance_reduction(nsample_ratios)
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
        return get_nsamples_per_model(target_cost, self.costs, nsample_ratios)

    @abstractmethod
    def _get_rsquared(self, nsample_ratios):
        r"""
        Compute r^2 used to compute the variance reduction of the Monte Carlo
        like estimator from a provided covariance. This is useful when
        optimizer is using cov as a torch.tensor

        Parameters
        ----------
        nsample_ratios : np.ndarray (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

        Returns
        -------
        rsquared : float
            The value r^2
        """
        raise NotImplementedError()


    @abstractmethod
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
        raise NotImplementedError

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
        return (nsample_ratios,
                self.get_variance(target_cost, nsample_ratios),
                rounded_target_cost)

    @abstractmethod
    def _generate_data(self, nsamples_per_model, functions):
        r"""
        Generate the samples and values needed to compute the Monte Carlo like
        estimator.

        Parameters
        ----------
        nsamples_per_model : np.ndarray (nsamples)
            The number of samples allocated to each model

        functions : list of callables
            The functions used to evaluate each model with signature

            `function(samples)->np.ndarray (nsamples, 1)`

            where samples : np.ndarray (nvars, nsamples)

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

    def generate_data(self, target_cost, nsample_ratios, functions):
        r"""
        Generate the samples and values needed to compute the Monte Carlo like
        estimator.

        Parameters
        ----------
        target_cost : float
            The total cost budget

        nsample_ratios : np.ndarray (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

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
        nsamples_per_model = self.get_nsamples_per_model(
            target_cost, nsample_ratios)
        return self._generate_data(nsamples_per_model, functions)


class MCEstimator(AbstractMonteCarloEstimator):
    def _get_rsquared(self, nsample_ratios):
        return 0

    def _estimate(self, values):
        return values[0].mean()

    def _allocate_samples(self, target_cost):
        nhf_samples = np.floor(target_cost/self.costs[0])
        nsample_ratios = np.zeros(0)
        log10_variance = np.log10(self.get_variance(
            nhf_samples, nsample_ratios))
        return nsample_ratios, log10_variance

    def _generate_data(self, nsamples_per_model, functions):
        samples = self.generate_samples(nsamples_per_model[0])
        if not callable(functions[0]):
            values = functions[0](samples)
        else:
            samples_with_id = np.vstack(
                [samples, np.zeros((1, nsamples_per_model[0]), dtype=float)])
            values = functions(samples_with_id)
        return samples, values


class MLMCEstimator(AbstractMonteCarloEstimator):
    def _get_rsquared(self, nsample_ratios):
        rsquared = get_rsquared_mlmc(self.cov, nsample_ratios)
        return rsquared

    def _allocate_samples(self, target_cost):
        return allocate_samples_mlmc(self.cov, self.costs, target_cost)

    def _estimate(self, values):
        eta = get_mlmc_control_variate_weights(self.get_covariance())
        return compute_approximate_control_variate_mean_estimate(eta, values)

    def _generate_data(self, nsamples_per_model, functions):
        return generate_samples_and_values_mlmc(
            nsamples_per_model, functions, self.generate_samples)


class MFMCEstimator(AbstractMonteCarloEstimator):
    def __init__(self, cov, costs, variable, sampling_method="random"):
        super().__init__(cov, costs, variable, sampling_method)
        corr_matrix = get_correlation_from_covariance(self.get_covariance())
        self.model_order = np.hstack(
            (0, np.argsort(corr_matrix[1:, 0])[::-1]+1))
        self.ordered_cov = self.cov[np.ix_(self.model_order, self.model_order)]
        self.ordered_costs = self.costs[self.model_order]
        self._check_model_costs_and_correlation(
            get_correlation_from_covariance(self.ordered_cov),
            self.ordered_costs)

    def _check_model_costs_and_correlation(self, corr, costs):
        models_accetable = check_mfmc_model_costs_and_correlations(costs, corr)
        if not models_accetable:
            msg = "Model correlations and costs cannot be used with MFMC"
            raise ValueError(msg)

    def _get_rsquared(self, ordered_nsample_ratios):
        rsquared = get_rsquared_mfmc(self.ordered_cov, ordered_nsample_ratios)
        return rsquared

    def _allocate_samples(self, target_cost):
        # nsample_ratios returned will be listed in according to
        # self.model_order which is what self.get_rsquared requires
        return allocate_samples_mfmc(
            self.ordered_cov, self.ordered_costs, target_cost)

    def _estimate(self, values):
        # Use self.cov (not self.ordered_cov) to ensure data corresponds to
        # user order of functions
        eta = get_mfmc_control_variate_weights(self.cov)
        return compute_approximate_control_variate_mean_estimate(eta, values)

    def _generate_data(self, nsamples_per_model, functions):
        return generate_samples_and_values_mfmc(
            nsamples_per_model, functions, self.generate_samples,
            acv_modification=False)


class AbstractACVEstimator(AbstractMonteCarloEstimator):
    def _estimate(self, values):
        eta = get_control_variate_weights(self.get_covariance())
        return compute_approximate_control_variate_mean_estimate(eta, values)


    def objective(self, target_cost, x):
        return acv_sample_allocation_objective_all(
            self, target_cost, x, use_torch)

    def _allocate_samples(self, target_cost, **kwargs):
        return allocate_samples_acv(
            self.cov_opt, self.costs_opt, target_cost, self,  **kwargs)


class ACVMFEstimator(AbstractACVEstimator):
    def _get_rsquared(self, nsample_ratios):
        return get_rsquared_acv(
            self.cov, nsample_ratios, get_discrepancy_covariances_MF)

    def _generate_data(self, nsamples_per_model, functions):
        return generate_samples_and_values_mfmc(
            nsamples_per_model, functions, self.generate_samples,
            acv_modification=True)


class ACVISEstimator(AbstractACVEstimator):
    def _get_rsquared(self, nsample_ratios):
        return get_rsquared_acv(
            self.cov, nsample_ratios, get_discrepancy_covariances_IS)

    def _generate_data(self, nsamples_per_model, functions):
        return generate_samples_and_values_acv_IS(
            nsamples_per_model, functions, self.generate_samples)


class ACVMFKLEstimator(ACVMFEstimator):
    def __init__(self, cov, costs, variable, K, L, sampling_method="random"):
        super().__init__(cov, costs, variable, sampling_method)
        self.K, self.L = K, L

    def _get_rsquared(self, nsample_ratios):
        return get_rsquared_acv(
            self.cov, nsample_ratios,
            partial(get_discrepancy_covariances_KL, K=self.K, L=self.L))

    def _generate_data(self, nsamples_per_model, functions):
        return generate_samples_and_values_acv_KL(
            nsamples_per_model, functions, self.generate_samples, K=self.K,
            L=self.L)


class ACVMFKLBestEstimator(ACVMFEstimator):
    def _get_rsquared(self, nsample_ratios):
        return get_rsquared_acv_KL_best(self.cov, nsample_ratios)

    def _allocate_samples(self, target_cost):
        return allocate_samples_acv_best_kl(
            self.cov, self.costs, target_cost,
            initial_guess=None, optim_options=None,
            optim_method='SLSQP')


monte_carlo_estimators = {"acvmf": ACVMFEstimator,
                          "acvmfkl": ACVMFKLEstimator,
                          "acvis": ACVISEstimator,
                          "mfmc": MFMCEstimator,
                          "mlmc": MLMCEstimator}


def get_estimator(estimator_type, cov, costs, variable, **kwargs):
    if estimator_type not in monte_carlo_estimators:
        msg = f"Estimator {estimator_type} not supported"
        msg += f"Must be one of {monte_carlo_estimators.keys()}"
        raise ValueError(msg)
    return monte_carlo_estimators[estimator_type](
        cov, costs, variable, **kwargs)
