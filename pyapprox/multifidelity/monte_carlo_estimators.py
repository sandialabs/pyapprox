from abc import ABC, abstractmethod
import os
import numpy as np
from functools import partial
from itertools import combinations
import copy

from pyapprox.util.utilities import get_correlation_from_covariance
from pyapprox.util.visualization import mathrm_label
from pyapprox.expdesign.low_discrepancy_sequences import (
    sobol_sequence, halton_sequence
)
from pyapprox.multifidelity.control_variate_monte_carlo import (
    compute_approximate_control_variate_mean_estimate,
    get_rsquared_mlmc, allocate_samples_mlmc, get_mlmc_control_variate_weights,
    get_rsquared_mfmc, allocate_samples_mfmc, get_mfmc_control_variate_weights,
    acv_sample_allocation_objective_all, get_nsamples_per_model,
    allocate_samples_acv, get_approximate_control_variate_weights,
    get_discrepancy_covariances_MF, round_nsample_ratios,
    check_mfmc_model_costs_and_correlations, pkg, use_torch, get_nhf_samples,
    get_discrepancy_covariances_IS,
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
    get_generalized_approximate_control_variate_weights,
    get_sample_allocation_matrix_mlmc, get_npartition_samples_mlmc,
    get_npartition_samples_mfmc,
    separate_samples_per_model_acv, get_sample_allocation_matrix_mfmc,
    get_npartition_samples_acvis, get_sample_allocation_matrix_acvis,
    bootstrap_acv_estimator, get_acv_recursion_indices,
)
from pyapprox.interface.wrappers import ModelEnsemble
from pyapprox.multifidelity.multilevelblue import (
    BLUE_variance, BLUE_Psi, BLUE_RHS, BLUE_evaluate_models,
    get_model_subsets, BLUE_cost_constraint, BLUE_hf_nsamples_constraint,
    BLUE_cost_constraint_jac, AETC_optimal_loss, BLUE_hf_nsamples_constraint_jac)
from scipy.optimize import minimize


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

        variable : :class:`pyapprox.variables.IndependentMarginalsVariable`
            The uncertain model parameters

        sampling_method : string
            Supported types are ["random", "sobol", "halton"]
        """
        self.cov, self.costs = self._check_cov(cov, costs)

        self.variable = variable
        self.sampling_method = sampling_method
        self.set_random_state(None)
        self.nmodels = len(costs)

        self.cov_opt = self.cov
        self.costs_opt = self.costs
        if use_torch:
            if not pkg.is_tensor(self.cov):
                self.cov_opt = pkg.tensor(self.cov, dtype=pkg.double)
            if not pkg.is_tensor(self.costs):
                self.costs_opt = pkg.tensor(self.costs, dtype=pkg.double)

    def _check_cov(self, cov, costs):
        if cov.shape[0] != len(costs):
            print(cov.shape, costs.shape)
            raise ValueError("cov and costs are inconsistent")
        return cov.copy(), np.array(costs)

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

    @abstractmethod
    def _estimate(self, values, *args):
        raise NotImplementedError()

    def __call__(self, values, *args):
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
        return self._estimate(values, *args)

    def __repr__(self):
        if self.optimized_variance is None:
            return "{0}".format(self.__class__.__name__)
        return "{0}(variance={1:.3g}, target_cost={2:.3g})".format(
            self.__class__.__name__, self.optimized_variance,
            self.rounded_target_cost)


class AbstractACVEstimator(AbstractMonteCarloEstimator):

    def __init__(self, cov, costs, variable, sampling_method="random"):
        super().__init__(cov, costs, variable, sampling_method)
        self.nsamples_per_model, self.optimized_variance = None, None
        self.rounded_target_cost = None
        self.model_labels = None

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
        # print(nsample_ratios, "nsample_ratios")
        nsample_ratios, rounded_target_cost = round_nsample_ratios(
            target_cost, self.costs, nsample_ratios)
        # print(nsample_ratios, "rounded nsample_ratios")
        variance = self.get_variance(rounded_target_cost, nsample_ratios)
        self.set_optimized_params(
            nsample_ratios, rounded_target_cost, variance)
        return nsample_ratios, variance, rounded_target_cost

    def set_optimized_params(self, nsample_ratios, rounded_target_cost,
                             optimized_variance):
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

        optimized_variance : float
            The variance of the estimator using the integer sample allocations
        """
        self.nsample_ratios = nsample_ratios
        self.rounded_target_cost = rounded_target_cost
        self.nsamples_per_model = get_nsamples_per_model(
            self.rounded_target_cost, self.costs, self.nsample_ratios,
            True).astype(int)
        self.optimized_variance = optimized_variance

    @abstractmethod
    def _get_npartition_samples(self, nsamples_per_model):
        r"""
        Get the size of the partitions combined to form
        :math:`z_i, i=0\ldots, M-1`.

        Parameters
        ----------
        nsamples_per_model : np.ndarray (nmodels)
             The number of total samples allocated to each model. I.e.
             :math:`|z_i\cup\z^\star_i|, i=0,\ldots,M-1`

        Returns
        -------
        npartition_samples : np.ndarray (nmodels)
            The size of the partitions that make up the subsets
            :math:`z_i, i=0\ldots, M-1`. These are represented by different
            color blocks in the ACV papers figures of sample allocation
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_reordered_sample_allocation_matrix(self):
        r"""
        Compute the reordered allocation matrix corresponding to
        self.nsamples_per_model set by set_optimized_params

        Returns
        -------
        mat : np.ndarray (nmodels, 2*nmodels)
            For columns :math:`2j, j=0,\ldots,M-1` the ith row contains a
            flag specifiying if :math:`z_i^\star\subseteq z_j^\star`
            For columns :math:`2j+1, j=0,\ldots,M-1` the ith row contains a
            flag specifiying if :math:`z_i\subseteq z_j`
        """
        raise NotImplementedError()

    def generate_sample_allocations(self):
        """
        Returns
        -------
        samples_per_model : list (nmodels)
                The ith entry contains the set of samples
                np.narray(nvars, nsamples_ii) used to evaluate the ith model.

        partition_indices_per_model : list (nmodels)
                The ith entry contains the indices np.narray(nsamples_ii)
                mapping each sample to a sample allocation partition
        """
        npartition_samples = self._get_npartition_samples(
            self.nsamples_per_model)
        reorder_allocation_mat = self._get_reordered_sample_allocation_matrix()
        samples_per_model, partition_indices_per_model = generate_samples_acv(
            reorder_allocation_mat, self.nsamples_per_model,
            npartition_samples, self.generate_samples)
        return samples_per_model, partition_indices_per_model

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
        acv_samples : list (nmodels)
            List containing the samples :math:`\mathcal{Z}_{i,1}` and
            :math:`\mathcal{Z}_{i,2}` for each model :math:`i=0,\ldots,M-1`.
            The list is [[:math:`\mathcal{Z}_{0,1}`,:math:`\mathcal{Z}_{0,2}`],
            ...,[:math:`\mathcal{Z}_{M-1,1}`,:math:`\mathcal{Z}_{M-1,2}`]],
            where :math:`M` is the number of models

        acv_values : list (nmodels)
            Each entry of the list contains

        values0 : np.ndarray (num_samples_i0,num_qoi)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

        values1: np.ndarray (num_samples_i1,num_qoi)
            Evaluations used compute the approximate
            mean :math:`\mu_{i,r_iN}` of the low fidelity models.
        """
        samples_per_model, partition_indices_per_model = \
            self.generate_sample_allocations()
        if type(functions) == list:
            functions = ModelEnsemble(functions)
        values_per_model = functions.evaluate_models(samples_per_model)
        reorder_allocation_mat = self._get_reordered_sample_allocation_matrix()
        acv_values = separate_model_values_acv(
            reorder_allocation_mat, values_per_model,
            partition_indices_per_model)
        acv_samples = separate_samples_per_model_acv(
            reorder_allocation_mat, samples_per_model,
            partition_indices_per_model)
        return acv_samples, acv_values

    def estimate_from_values_per_model(self, values_per_model,
                                       partition_indices_per_model):
        reorder_allocation_mat = self._get_reordered_sample_allocation_matrix()
        acv_values = separate_model_values_acv(
            reorder_allocation_mat, values_per_model,
            partition_indices_per_model)
        return self(acv_values)

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

    def bootstrap(self, values_per_model, partition_indices_per_model,
                  nbootstraps=1000):
        return bootstrap_acv_estimator(
            values_per_model, partition_indices_per_model,
            self._get_npartition_samples(self.nsamples_per_model),
            self._get_reordered_sample_allocation_matrix(),
            self._get_approximate_control_variate_weights(), nbootstraps)


class MCEstimator(AbstractMonteCarloEstimator):

    def _estimate(self, values):
        return values[0].mean()

    def allocate_samples(self, target_cost):
        nhf_samples = int(np.floor(target_cost/self.costs[0]))
        nsample_ratios = np.zeros(0)
        variance = self.get_variance(nhf_samples)
        self.nsamples_per_model = np.array([nhf_samples])
        self.rounded_target_cost = self.costs[0]*self.nsamples_per_model[0]
        self.optimized_variance = variance
        return nsample_ratios, variance, self.rounded_target_cost

    def get_variance(self, nhf_samples):
        return self.cov[0, 0]/nhf_samples

    def generate_data(self, functions):
        samples = self.generate_samples(self.nhf_samples)
        if not callable(functions):
            values = functions[0](samples)
        else:
            samples_with_id = np.vstack(
                [samples,
                 np.zeros((1, self.nhf_samples), dtype=np.double)])
            values = functions(samples_with_id)
        return samples, values


class MLMCEstimator(AbstractACVEstimator):
    def _get_rsquared(self, cov, nsample_ratios):
        rsquared = get_rsquared_mlmc(cov, nsample_ratios)
        return rsquared

    def _allocate_samples(self, target_cost):
        return allocate_samples_mlmc(self.cov, self.costs, target_cost)

    def _get_npartition_samples(self, nsamples_per_model):
        return get_npartition_samples_mlmc(nsamples_per_model)

    def _get_reordered_sample_allocation_matrix(self):
        return get_sample_allocation_matrix_mlmc(self.nmodels)

    def _get_approximate_control_variate_weights(self):
        return get_mlmc_control_variate_weights(self.get_covariance().shape[0])


class MFMCEstimator(AbstractACVEstimator):
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

    def _get_reordered_sample_allocation_matrix(self):
        return get_sample_allocation_matrix_mfmc(self.nmodels)

    # def generate_data(self, functions):
    #    return generate_samples_and_values_mfmc(
    #        self.nsamples_per_model, functions, self.generate_samples,
    #        acv_modification=False)

    def _get_npartition_samples(self, nsamples_per_model):
        return get_npartition_samples_mfmc(nsamples_per_model)


class AbstractNumericalACVEstimator(AbstractACVEstimator):
    def __init__(self, cov, costs, variable, sampling_method="random"):
        super().__init__(cov, costs, variable, sampling_method)
        self.cons = []
        self.set_initial_guess(None)

    def objective(self, target_cost, x, return_grad=True):
        # return_grad argument used for testing with finte difference
        out = acv_sample_allocation_objective_all(
            self, target_cost, x, (use_torch and return_grad))
        return out

    def set_initial_guess(self, initial_guess):
        self.initial_guess = initial_guess

    def _allocate_samples(self, target_cost, **kwargs):
        cons = self.get_constraints(target_cost)

        opt = allocate_samples_acv(
            self.cov_opt, self.costs, target_cost, self,  cons,
            initial_guess=self.initial_guess)

        if (check_mfmc_model_costs_and_correlations(
                self.costs, get_correlation_from_covariance(self.cov)) and
            len(self.cov_opt) == len(self.costs)):
            # second condition above  will not be true for multiple qoi
            mfmc_initial_guess = allocate_samples_mfmc(
                self.cov_opt, self.costs, target_cost)[0]
            opt_mfmc = allocate_samples_acv(
                self.cov_opt, self.costs, target_cost, self, cons,
                initial_guess=mfmc_initial_guess)
            # print(opt[1], opt_mfmc[1])
            if opt_mfmc[1] < opt[1]:
                # print("using mfmc initial guess")
                opt = opt_mfmc

        return opt

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


class ACVGMFEstimator(AbstractNumericalACVEstimator):
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
        self.allocation_mat = get_sample_allocation_matrix_acvmf(
            self.recursion_index)

    def _get_npartition_samples(self, nsamples_per_model):
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

    def _get_rsquared(self, cov, nsample_ratios, target_cost):
        nsamples_per_model = get_nsamples_per_model(
            target_cost, self.costs, nsample_ratios, False)
        weights, cf = get_generalized_approximate_control_variate_weights(
            self.allocation_mat, nsamples_per_model,
            self._get_npartition_samples, self.cov, self.recursion_index)
        rsquared = -cf.dot(weights)/cov[0, 0]
        return rsquared

    def _get_approximate_control_variate_weights(self):
        return get_generalized_approximate_control_variate_weights(
            self.allocation_mat, self.nsamples_per_model,
            self._get_npartition_samples, self.cov, self.recursion_index)[0]

    def _get_reordered_sample_allocation_matrix(self):
        return reorder_allocation_matrix_acvgmf(
            self.allocation_mat, self.nsamples_per_model, self.recursion_index)

    def __repr__(self):
        if self.optimized_variance is None:
            return "{0}".format(self.__class__.__name__)
        return (
            "{0}(variance={1:.3g}, target_cost={2:.3g}, recursion={3})".format(
                self.__class__.__name__, self.optimized_variance,
                self.rounded_target_cost, self.recursion_index))


class ACVGMFBEstimator(ACVGMFEstimator):
    def __init__(self, cov, costs, variable, tree_depth=None,
                 sampling_method="random"):
        super().__init__(cov, costs, variable, sampling_method)
        self._depth = tree_depth

    def allocate_samples(self, target_cost):
        best_variance = np.inf
        best_result = None
        for index in get_acv_recursion_indices(self.nmodels, self._depth):
            self.set_recursion_index(index)
            try:
                super().allocate_samples(target_cost)
            except RuntimeError:
                # typically solver failes because trying to use
                # uniformative model as a recursive control variate
                self.optimized_variance = np.inf
            if self.optimized_variance < best_variance:
                best_result = [self.nsample_ratios, self.rounded_target_cost,
                               self.optimized_variance, index]
                best_variance = self.optimized_variance
        if best_result is None:
            raise RuntimeError("No solutions were found")
        self.set_recursion_index(best_result[3])
        self.set_optimized_params(*best_result[:3])
        return best_result[:3]

    def _get_rsquared_from_nhf_samples(self, nhf_samples, cov, nsample_ratios):
        target_cost = nhf_samples*self.costs[0]+nsample_ratios.dot(
            self.costs[1:])*nhf_samples
        best_rsq = -np.inf
        for index in get_acv_recursion_indices(self.nmodels):
            self.set_recursion_index(index)
            rsq = super()._get_rsquared(cov, nsample_ratios, target_cost)
            best_rsq = max(best_rsq, rsq)
        return best_rsq


class ACVMFEstimator(AbstractNumericalACVEstimator):
    def _get_rsquared(self, cov, nsample_ratios):
        return get_rsquared_acv(
            cov, nsample_ratios, get_discrepancy_covariances_MF)

    # def generate_data(self, functions):
    #     return generate_samples_and_values_mfmc(
    #         self.nsamples_per_model, functions, self.generate_samples,
    #         acv_modification=True)

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

    def _get_npartition_samples(self, nsamples_per_model):
        return get_npartition_samples_acvmf(nsamples_per_model)

    def _get_reordered_sample_allocation_matrix(self):
        return get_sample_allocation_matrix_acvmf(
            np.zeros(self.nmodels-1, dtype=int))


class ACVISEstimator(AbstractNumericalACVEstimator):
    # recursion not currently supported

    def _get_rsquared(self, cov, nsample_ratios):
        # from pyapprox.multifidelity.control_variate_monte_carlo import (
        #     get_discrepancy_covariances_IS_new)
        return get_rsquared_acv(
            cov, nsample_ratios, get_discrepancy_covariances_IS)

    # def generate_data(self, functions):
    #     return generate_samples_and_values_acv_IS(
    #         self.nsamples_per_model, functions, self.generate_samples)

    def _get_approximate_control_variate_weights(self):
        nsample_ratios = self.nsamples_per_model[1:]/self.nsamples_per_model[0]
        CF, cf = get_discrepancy_covariances_IS(self.cov, nsample_ratios)
        return get_approximate_control_variate_weights(CF, cf)

    def _get_npartition_samples(self, nsamples_per_model):
        npartition_samples = get_npartition_samples_acvis(
            nsamples_per_model)
        return npartition_samples

    def _get_reordered_sample_allocation_matrix(self):
        return get_sample_allocation_matrix_acvis(
            np.zeros(self.nmodels-1, dtype=int))

    def _get_constraints(self, target_cost):
        cons = [
            {'type': 'ineq',
             'fun': acv_sample_allocation_nlf_gt_nhf_ratio_constraint,
             'jac': acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac,
             'args': (ii, target_cost, self.costs)}
            for ii in range(1, self.nmodels)]
        return cons


def get_best_models_for_acv_estimator(
        estimator_type, cov, costs, variable, target_cost,
        max_nmodels=None, init_kwargs={}, allocate_kwargs={}):
    nmodels = cov.shape[0]
    if max_nmodels is None:
        max_nmodels = nmodels
    lf_model_indices = np.arange(1, nmodels)
    best_variance = np.inf
    best_est, best_model_indices = None, None
    for nsubset_lfmodels in range(1, max_nmodels):
        for lf_model_subset_indices in combinations(
                lf_model_indices, nsubset_lfmodels):
            idx = np.hstack(([0], lf_model_subset_indices)).astype(int)
            subset_cov, subset_costs = cov[np.ix_(idx, idx)], costs[idx]
            # print('####', idx)
            # print(kwargs)
            if "tree_depth" in init_kwargs:
                init_kwargs["tree_depth"] = min(
                    init_kwargs["tree_depth"], nsubset_lfmodels)
            try:
                est = get_estimator(
                    estimator_type, subset_cov, subset_costs, variable,
                    **init_kwargs)
            except ValueError:
                # Some estiamtors e.g. MFMC fail when certain criteria are
                # not satisfied
                continue
            try:
                est.allocate_samples(target_cost, **allocate_kwargs)
                # print(idx, est.recursion_index, est.optimized_variance)
                if est.optimized_variance < best_variance:
                    best_est = est
                    best_model_indices = idx
                    best_variance = est.optimized_variance
            except (RuntimeError, ValueError):
                # raise e
                # print(e)
                continue
            # print(idx, est.optimized_variance)
    return best_est, best_model_indices


class BestModelSubsetEstimator():
    def __init__(self, estimator_type, cov, costs, variable,
                 max_nmodels, **kwargs):
        self.estimator_type = estimator_type
        self._cov, self._costs, self.variable = cov, costs, variable
        self.max_nmodels = max_nmodels
        self.kwargs = kwargs

        self.optimized_variance = None
        self.rounded_target_cost = None
        self.best_est = None
        self.best_model_indices = None
        self._all_model_labels = None

    @property
    def nmodels(self):
        return self.best_est.nmodels

    @property
    def nsamples_per_model(self):
        return self.best_est.nsamples_per_model

    @property
    def cov(self):
        return self.best_est.cov

    @property
    def costs(self):
        return self.best_est.costs

    @property
    def model_labels(self):
        return [self._all_model_labels[idx] for idx in self.best_model_indices]

    def get_variance(self, *args):
        return self.best_est.get_variance(*args)

    @model_labels.setter
    def model_labels(self, labels):
        self._all_model_labels = labels

    def allocate_samples(self, target_cost, **allocate_kwargs):
        if self.estimator_type == "mc":
            best_model_indices = np.array([0])
            best_est = get_estimator(
                self.estimator_type, self._cov[:1, :1],
                self._costs[:1], self.variable, **self.kwargs)
            best_est.allocate_samples(target_cost)

        else:
            best_est, best_model_indices = get_best_models_for_acv_estimator(
                self.estimator_type, self._cov, self._costs, self.variable,
                target_cost, self.max_nmodels, self.kwargs,
                allocate_kwargs)
        self.optimized_variance = best_est.optimized_variance
        self.rounded_target_cost = best_est.rounded_target_cost
        self.best_est = best_est
        self.best_model_indices = best_model_indices

    def __repr__(self):
        if self.optimized_variance is None:
            return "{0}".format(self.__class__.__name__)
        return "{0}(type={1}, variance={2:.3g}, target_cost={3:.3g}, subset={4})".format(
            self.__class__.__name__, self.best_est.__class__.__name__,
            self.optimized_variance, self.rounded_target_cost,
            self.best_model_indices)


def compute_single_fidelity_and_approximate_control_variate_mean_estimates(
        target_cost, nsample_ratios, estimator,
        model_ensemble, seed):
    r"""
    Compute the approximate control variate estimate of a high-fidelity
    model from using it and a set of lower fidelity models.
    Also compute the single fidelity Monte Carlo estimate of the mean from
    only the high-fidelity data.

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
    random_state = np.random.RandomState(seed)
    estimator.set_random_state(random_state)
    samples, values = estimator.generate_data(model_ensemble)
    # compute mean using only hf daa
    hf_mean = values[0][1].mean()
    # compute ACV mean
    acv_mean = estimator(values)
    return hf_mean, acv_mean


def estimate_variance(model_ensemble, estimator, target_cost,
                      ntrials=1e3, nsample_ratios=None, max_eval_concurrency=1):
    r"""
    Numerically estimate the variance of an approximate control variate
    estimator.

    Parameters
    ----------
    model_ensemble: :class:`pyapprox.interface.wrappers.ModelEnsemble`
        Model that takes random samples and model id as input

    estimator : :class:`pyapprox.multifidelity.monte_carlo_estimators.AbstractMonteCarloEstimator`
        A Monte Carlo like estimator for computing sample based statistics

    target_cost : float
        The total cost budget

    ntrials : integer
        The number of times to compute estimator using different randomly
        generated set of samples

    nsample_ratios : np.ndarray (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
            the number of samples in the two different discrepancies involving
            the ith model.

            If not provided nsample_ratios will be optimized based on target
            cost

    max_eval_concurrency : integer
        The number of processors used to compute realizations of the estimators
        which can be run independently and in parallel.

    Returns
    -------
    means : np.ndarray (ntrials, 2)
        The high-fidelity and estimator means for each trial

    numerical_var : float
        The variance computed numerically from the trials

    true_var : float
        The variance computed analytically
    """
    if nsample_ratios is None:
        nsample_ratios, variance, rounded_target_cost = \
            estimator.allocate_samples(target_cost)
    else:
        rounded_target_cost = target_cost
        estimator.set_optimized_params(
            nsample_ratios, rounded_target_cost, None)

    ntrials = int(ntrials)
    from multiprocessing import Pool
    func = partial(
        compute_single_fidelity_and_approximate_control_variate_mean_estimates,
        rounded_target_cost, nsample_ratios, estimator, model_ensemble)
    if max_eval_concurrency > 1:
        assert int(os.environ['OMP_NUM_THREADS']) == 1
        pool = Pool(max_eval_concurrency)
        means = np.asarray(pool.map(func, list(range(ntrials))))
        pool.close()
    else:
        means = np.empty((ntrials, 2))
        for ii in range(ntrials):
            means[ii, :] = func(ii)

    numerical_var = means[:, 1].var(axis=0)
    true_var = estimator.get_variance(
        estimator.rounded_target_cost, estimator.nsample_ratios)
    return means, numerical_var, true_var


def bootstrap_monte_carlo_estimator(values, nbootstraps=10, verbose=True):
    """
    Approximate the variance of the Monte Carlo estimate of the mean using
    bootstraping

    Parameters
    ----------
    values : np.ndarry (nsamples, 1)
        The values used to compute the mean

    nbootstraps : integer
        The number of boostraps used to compute estimator variance

    verbose:
        If True print the estimator mean and +/- 2 standard deviation interval

    Returns
    -------
    bootstrap_mean : float
        The bootstrap estimate of the estimator mean

    bootstrap_variance : float
        The bootstrap estimate of the estimator variance
    """
    values = values.squeeze()
    assert values.ndim == 1
    nsamples = values.shape[0]
    bootstrap_values = np.random.choice(
        values, size=(nsamples, nbootstraps), replace=True)
    bootstrap_means = bootstrap_values.mean(axis=0)
    bootstrap_mean = bootstrap_means.mean()
    bootstrap_variance = np.var(bootstrap_means)
    if verbose:
        print('No. samples', values.shape[0])
        print('Mean', bootstrap_mean)
        print('Mean +/- 2 sigma', [bootstrap_mean-2*np.sqrt(
            bootstrap_variance), bootstrap_mean+2*np.sqrt(bootstrap_variance)])

    return bootstrap_mean, bootstrap_variance


def compare_estimator_variances(target_costs, estimators):
    """
    Compute the variances of different Monte-Carlo like estimators.

    Parameters
    ----------
    target_costs : np.ndarray (ntarget_costs)
        Different total cost budgets

    estimators : list (nestimators)
        List of Monte Carlo estimator objects, e.g.
        :class:`pyapprox.multifidelity.control_variate_monte_carlo.MC`

    Returns
    -------
        optimized_estimators : list
         Each entry is a list of optimized estimators for a set of target costs
    """
    optimized_estimators = []
    for est in estimators:
        est_copies = []
        for target_cost in target_costs:
            est_copy = copy.deepcopy(est)
            est_copy.allocate_samples(target_cost)
            est_copies.append(est_copy)
        optimized_estimators.append(est_copies)
    return optimized_estimators


def plot_estimator_variances(optimized_estimators,
                             est_labels, ax, ylabel=None,
                             relative_id=0, cost_normalization=1):
    """
    Plot variance as a function of the total cost for a set of estimators.

    Parameters
    ----------
    optimized_estimators : list
         Each entry is a list of optimized estimators for a set of target costs

    est_labels : list (nestimators)
        String used to label each estimator

    relative_id the model id used to normalize variance
    """
    linestyles = ['-', '--', ':', '-.', (0, (5, 10)), '-']
    nestimators = len(est_labels)
    est_variances = []
    for ii in range(nestimators):
        est_total_costs = np.array(
            [est.rounded_target_cost for est in optimized_estimators[ii]])
        est_variances.append(np.array(
            [est.optimized_variance for est in optimized_estimators[ii]]))
    est_total_costs *= cost_normalization
    print(est_total_costs, cost_normalization)
    for ii in range(nestimators):
        # print(est_labels[ii], nestimators)
        ax.loglog(est_total_costs,
                  est_variances[ii]/est_variances[relative_id][0],
                  label=est_labels[ii], ls=linestyles[ii], marker='o')
    if ylabel is None:
        ylabel = mathrm_label("Estimator variance")
    ax.set_xlabel(mathrm_label("Target cost"))
    ax.set_ylabel(ylabel)
    ax.legend()


def _autolabel(ax, rects, model_labels):
    # Attach a text label in each bar in *rects*
    for rect, label in zip(rects, model_labels):
        try:
            rect = rect[0]
        except TypeError:
            pass
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width()/2,
                        rect.get_y() + rect.get_height()/2),
                    xytext=(0, -10),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_acv_sample_allocation_comparison(
        estimators, model_labels, ax, legendloc=[0.925, 0.25]):
    """
    Plot the number of samples allocated to each model for a set of estimators

    Parameters
    ----------
    estimators : list
       Each entry is a MonteCarlo like estimator

    model_labels : list (nestimators)
        String used to label each estimator
    """

    nestimators = len(estimators)
    xlocs = np.arange(nestimators)

    from matplotlib.pyplot import cm
    for jj, est in enumerate(estimators):
        cnt = 0
        # warning currently colors will not match if estimators use different
        # models
        colors = cm.rainbow(np.linspace(0, 1, est.nmodels))
        rects = []
        est.model_labels = model_labels
        for ii in range(est.nmodels):
            if jj == 0:
                label = est.model_labels[ii]
            else:
                label = None
            cost_ratio = (est.costs[ii]*est.nsamples_per_model[ii] /
                          est.rounded_target_cost)
            rect = ax.bar(
                xlocs[jj:jj+1], cost_ratio, bottom=cnt, edgecolor='white',
                label=label, color=colors[ii])
            rects.append(rect)
            cnt += cost_ratio
            # print(est.nsamples_per_model[ii], label)
        _autolabel(ax, rects, ['$%d$' % int(est.nsamples_per_model[ii])
                               for ii in range(est.nmodels)])
    ax.set_xticks(xlocs)
    # number of samples are rounded cost est_rounded cost,
    # but target cost is not rounded
    ax.set_xticklabels(
        ['$%1.2f$' % est.rounded_target_cost for est in estimators])
    ax.set_xlabel(mathrm_label("Target cost"))
    # / $N_\alpha$')
    ax.set_ylabel(
        mathrm_label("Precentage of target cost"))
    if legendloc is not None:
        ax.legend(loc=legendloc)


class MLBLUEstimator(AbstractMonteCarloEstimator):
    def __init__(self, cov, costs, variable, sampling_method="random",
                 reg_blue=1e-12):
        super().__init__(cov, costs, variable, sampling_method)
        self._reg_blue = reg_blue
        self.nsamples_per_subset, self.optimized_variance = None, None
        self.rounded_target_cost = None
        self.nsamples_per_model = None
        self.subsets = get_model_subsets(self.nmodels)
        self.nsubsets = len(self.subsets)
        self.subset_costs = self._get_model_subset_costs(
            self.subsets, self.costs)
        self.hf_subset_vec = self._get_nhf_subset_vec()

    def _get_nhf_subset_vec(self):
        hf_subset_vec = np.zeros(self.nsubsets)
        for ii, subset in enumerate(self.subsets):
            if 0 in subset:
                hf_subset_vec[ii] = 1
        return hf_subset_vec

    def _get_nsamples_per_model(self, subsets, nsamples_per_subset):
        nsamples_per_model = np.zeros(self.nmodels)
        for jj, subset in enumerate(subsets):
            nsamples_per_model[subset] += nsamples_per_subset[jj]
        return nsamples_per_model

    def _objective(self, asketch, nsamples_per_subset):
        obj, grad = BLUE_variance(
            asketch, self.cov, self.costs, self._reg_blue, self.subsets,
            nsamples_per_subset, return_grad=True)
        grad = 1/obj*grad
        obj = np.log(obj)
        return obj, grad

    def _get_bounds(self, target_cost):
        nsubsets = len(self.subsets)
        bounds = [(0, target_cost/self.subset_costs[ii]) for ii in range(
            nsubsets)]
        return bounds

    @staticmethod
    def _get_constraints(subsets, subset_costs, target_cost, constraint_reg):
        return [
            {'type': 'ineq',
             'fun': partial(BLUE_hf_nsamples_constraint, subsets),
             # use of anlytical gradient messes up minimize
             'jac': partial(BLUE_hf_nsamples_constraint_jac, subsets)
             },
            {'type': 'eq',
             'fun': partial(BLUE_cost_constraint, target_cost, subset_costs),
             # 'jac': partial(
             #     BLUE_cost_constraint_jac, target_cost, subset_costs)
             }
        ]

    def _set_optimization_result(
            self, nsamples_per_subset, subsets, round_nsamples,
            asketch):
        if round_nsamples:
            unrounded_nsamples_per_subset = nsamples_per_subset.copy()
            nsamples_per_subset = np.asarray(nsamples_per_subset).astype(int)
            maxn = 0
            for subset, ns in zip(subsets, nsamples_per_subset):
                if 0 in subset:
                    maxn = max(maxn, ns)
            if maxn == 0:
                msg = "No high-fidelity samples were used after rounding. "
                msg += "Likely the relaxed sum of samples of sets involving "
                msg += "the high-fidelity model sumed to >1 but no individual "
                msg += "sample size was >1. nsamples_per_subset was "
                msg += "\n{0}.\n".format(unrounded_nsamples_per_subset)
                msg += 'Increasing target_cost will fix this'
                raise RuntimeError(msg)
        rounded_target_cost = self._get_nsamples_per_model(
            subsets, nsamples_per_subset) @ self.costs

        # set attributes needed for self._estimate
        self.nsamples_per_subset = nsamples_per_subset
        # self.optimized_variance = BLUE_variance(
        #     asketch, self.cov, None, self._reg_blue,
        #     self.nsamples_per_subset)
        self.optimized_variance = self.get_variance(
            nsamples_per_subset, asketch)
        self.rounded_target_cost = rounded_target_cost
        self.nsamples_per_model = self._get_nsamples_per_model(
            self.subsets, self.nsamples_per_subset)
        return self.optimized_variance, self.rounded_target_cost

    def _extract_optimization_result(self, res, target_cost):
        variance = res["fun"]
        # This is normalized variance true variance is obtained by
        # variance/target_cost
        # nsamples_per_subset_frac = np.maximum(
        #     np.zeros_like(res["x"]), res["x"])
        # nsamples_per_subset_frac /= nsamples_per_subset_frac.sum()
        # # transform nsamples as fraction of unit budget to fraction of
        # # target_cost
        # nsamples_per_subset = target_cost*nsamples_per_subset_frac
        # # correct for normalization of nsamples by cost
        # nsamples_per_subset /= self.subset_costs
        nsamples_per_subset = res["x"]
        return variance, nsamples_per_subset, self.subsets

    def _init_guess(self, target_cost):
        nsubsets = len(self.subsets)
        # x0 = (target_cost/nsubsets)/self.subset_costs
        x0 = np.zeros(nsubsets)
        x0[-1] = target_cost/self.subset_costs[-1]
        return x0

    def _cvxpy_psi(self, nsps_cvxpy):
        import cvxpy
        Psi_blocks = BLUE_Psi(
            self.cov, self.costs, 0, self.subsets, np.ones(self.nsubsets))[1]
        Psi = (
            np.hstack([b.flatten()[:, None] for b in Psi_blocks])@nsps_cvxpy)
        Psi = cvxpy.reshape(Psi, (self.nmodels, self.nmodels))
        return Psi

    def _cvxpy_spd_constraint(self, asketch, nsps_cvxpy, t_cvxpy):
        import cvxpy
        Psi = self._cvxpy_psi(nsps_cvxpy)
        mat = cvxpy.bmat(
            [[Psi, asketch], [asketch.T, cvxpy.reshape(t_cvxpy, (1, 1))]])
        return mat

    def _minimize_cvxpy(self, target_cost, asketch):
        # use notation from https://www.cvxpy.org/examples/basic/sdp.html
        import cvxpy
        t_cvxpy = cvxpy.Variable(nonneg=True)
        nsps_cvxpy = cvxpy.Variable(self.nsubsets, nonneg=True)
        obj = cvxpy.Minimize(t_cvxpy)
        constraints = [self.subset_costs@nsps_cvxpy <= target_cost]
        constraints += [self.hf_subset_vec@nsps_cvxpy >= 1]
        constraints += [self._cvxpy_spd_constraint(
            asketch, nsps_cvxpy, t_cvxpy) >> 0]
        prob = cvxpy.Problem(obj, constraints)
        prob.solve(verbose=0, solver="CVXOPT")
        res = dict([("x",  nsps_cvxpy.value), ("fun", t_cvxpy.value)])
        return res

    def allocate_samples(self, target_cost, asketch=None,
                         constraint_reg=1e-12, round_nsamples=True,
                         options={}, init_guess=None):
        """
        Parameters
        ----------
        """
        nmodels = len(self.costs)
        if asketch is None:
            asketch = np.zeros((nmodels, 1))
            asketch[0] = 1.0
        assert asketch.shape[0] == self.costs.shape[0]

        obj = partial(self._objective, asketch)
        constraints = self._get_constraints(
            self.subsets, self.subset_costs, target_cost, constraint_reg)
        if init_guess is None:
            init_guess = self._init_guess(target_cost)
        jac = True
        method = options.pop("method", "trust-constr")
        if method == "cvxpy":
            res = self._minimize_cvxpy(target_cost, asketch)
        else:
            res = minimize(
                obj, init_guess, jac=jac,
                method=method, constraints=constraints, options=options,
                bounds=self._get_bounds(target_cost))
            if not res.success:
                msg = f"optimization not successful {res}"
                print(msg)
                raise RuntimeError(msg)

        variance, nsamples_per_subset, subsets = (
            self._extract_optimization_result(res, target_cost))
        variance, rounded_target_cost = self._set_optimization_result(
            nsamples_per_subset, subsets, round_nsamples,
            asketch)
        return nsamples_per_subset, variance, rounded_target_cost

    @staticmethod
    def _get_model_subset_costs(subsets, costs):
        subset_costs = np.array(
            [costs[subset].sum() for subset in subsets])
        return subset_costs

    def generate_data(self, models, variable, pilot_values=None):
        # todo consider removing self.variable from baseclass
        return BLUE_evaluate_models(
            variable.rvs, models, self.subsets, self.nsamples_per_subset,
            pilot_values)

    def _estimate(self, values, asketch=None):
        if asketch is None:
            asketch = np.zeros((self.nmodels, 1))
            asketch[0] = 1.0
        Psi, _ = BLUE_Psi(
            self.cov, None, self._reg_blue, self.subsets,
            self.nsamples_per_subset)
        rhs = BLUE_RHS(self.subsets, self.cov, values)
        return np.linalg.multi_dot(
            (asketch.T, np.linalg.lstsq(Psi, rhs, rcond=None)[0]))

    def get_variance(self, nsamples_per_subset, asketch):
        return BLUE_variance(
            asketch, self.cov, None, self._reg_blue, self.subsets,
            nsamples_per_subset)

    def bootstrap_estimator(self, values, asketch, nbootstraps=1000):
        means = np.empty(nbootstraps)
        for ii in range(nbootstraps):
            perturbed_vals = []
            for jj in range(len(values)):
                if len(values[jj]) == 0:
                    perturbed_vals.append([])
                    continue
                nsubset_samples = values[jj].shape[0]
                indices = np.random.choice(
                    np.arange(nsubset_samples, dtype=int), nsubset_samples,
                    p=None, replace=True)
                perturbed_vals.append(values[jj][indices])
            means[ii] = self._estimate(perturbed_vals, asketch)
        return means

    def _get_rsquared_from_nhf_samples(self, nhf_samples, cov, nsample_ratios):
        # WARNING
        # this only works for fully coupled BLUEs defined in Section 5.5
        # of the paper 'On multilevel best linear unbiased estimators'
        nmodels = cov.shape[0]

        # compute active subsets of fully coupled BLUEs
        active_subsets = [
            np.arange(nmodels)[ii:] for ii in range(nmodels)]
        active_subset_indices = []
        for subset in active_subsets:
            for ii, s in enumerate(self.subsets):
                if s.shape[0] == len(subset) and np.allclose(s, subset):
                    active_subset_indices.append(ii)
                    break
        active_subset_indices = np.asarray(active_subset_indices)

        # convert nsample_ratios into nsamples_per_subset
        nsamples_per_subset = np.zeros(len(self.subsets))
        nlf_model_samples = nsample_ratios*nhf_samples
        nsamples_per_subset[active_subset_indices] = np.hstack(
            (nhf_samples, nlf_model_samples[0]-nhf_samples,
             np.diff(nlf_model_samples)))

        # define asketch that targets estimation of only the high-fidelity
        # model
        asketch = np.zeros((nmodels, 1))
        asketch[0] = 1.0

        # estimate the variance reduction relative to single-fidelity MC
        mlblue_variance = BLUE_variance(
            asketch, cov, None, self._reg_blue, self.subsets,
            nsamples_per_subset)
        mc_variance = cov[0, 0]/nhf_samples
        gamma = mlblue_variance/mc_variance
        return 1-gamma


class AETCBLUE():
    def __init__(self, models, rvs, costs=None, oracle_stats=None,
                 reg_blue=1e-15, constraint_reg=0):
        r"""
        Parameters
        ----------
        models : list
            List of callable functions fun with signature

            ``fun(samples)-> np.ndarary (nsamples, nqoi)``

        where samples is np.ndarray (nvars, nsamples)

        rvs : callable
            Function used to generate random samples with signature

            ``fun(nsamples)-> np.ndarary (nvars, nsamples)``

        costs : iterable
            Iterable containing the time taken to evaluate a single sample
            with each model. If None then each model will be assumed to
            track the evaluation time.

        oracle_stats : list[np.ndarray (nmodels, nmodels), np.ndarray (nmodels, nmodels)]
            This is only used for testing.
            First element is the Oracle covariance between models.
            Second element is the Oracle Lambda_Sp
        """
        self.models = models
        self._nmodels = len(models)
        if not callable(rvs):
            raise ValueError("rvs must be callabe")
        self.rvs = rvs
        self._costs = self._validate_costs(costs)
        self._reg_blue = reg_blue
        self._constraint_reg = constraint_reg
        self._oracle_stats = oracle_stats

    def _validate_costs(self, costs):
        if costs is None:
            return
        if len(costs) != self._nmodels:
            raise ValueError("costs must be provided for each model")
        return np.asarray(costs)

    def _validate_subsets(self, subsets):
        # subsets are indexes of low fidelity models
        if subsets is None:
            subsets = get_model_subsets(self._nmodels-1)
        validated_subsets, max_ncovariates = [], -np.inf
        for subset in subsets:
            if ((np.unique(subset).shape[0] != len(subset)) or
                    (np.max(subset) >= self._nmodels-1)):
                msg = "subsets provided are not valid. First invalid subset"
                msg += f" {subset}"
                raise ValueError(msg)
            validated_subsets.append(np.asarray(subset))
            max_ncovariates = max(max_ncovariates, len(subset))
        return validated_subsets, max_ncovariates

    def _explore_step(self, total_budget, subsets, values, alpha,
                      reg_blue, constraint_reg):
        """
        Parameters
        ----------
        subsets : list[np.ndarray]
           Indices of the low fidelity models in a subset from 0,...,K-2
           e.g. (0) contains only the first low fidelity model and (0, 2)
           contains the first and third. 0 DOES NOT correspond to the
           high-fidelity model
        """
        explore_cost = np.sum(self._costs)
        results = []
        # print()
        for subset in subsets:
            result = AETC_optimal_loss(
                total_budget, values[:, :1], values[:, 1:], self._costs,
                subset, alpha, reg_blue, constraint_reg, self._oracle_stats)
            (loss, nsamples_per_subset_frac, explore_rate, beta_Sp,
             Sigma_S, k2, exploit_budget) = result
            results.append(result)
            # print(subset)
            # print(result)

        # compute optimal model
        best_subset_idx = np.argmin([result[0] for result in results])
        best_result = results[best_subset_idx]
        (best_loss, best_allocation_frac, best_rate, best_beta_Sp,
         best_Sigma_S, best_blue_variance, best_exploit_budget) = best_result
        best_cost = self._costs[subsets[best_subset_idx]+1].sum()

        nsamples = values.shape[0]

        # Incrementing one round at a time is the most optimal
        # but does not allow for parallelism
        # if best_rate <= nsamples:
        #     nexplore_samples = nsamples
        # else:
        #     nexplore_samples = nsamples + 1

        if best_rate > 2*nsamples:
            nexplore_samples = 2*nsamples
        elif best_rate > nsamples:
            nexplore_samples = int(np.ceil((nsamples+best_rate)/2))
        else:
            nexplore_samples = nsamples

        if (total_budget-nexplore_samples*explore_cost) < 0:
            nexplore_samples = int(total_budget/explore_cost)

        best_subset = subsets[best_subset_idx]
        # use +1 to accound for subset indexing only lf models
        best_subset_costs = self._costs[best_subset+1]
        best_subset_groups = get_model_subsets(best_subset.shape[0])
        best_subset_group_costs = [
            best_subset_costs[group].sum() for group in best_subset_groups]
        # transform nsamples as fraction of unit budget to fraction of
        # target_cost
        target_cost_fractions = (total_budget-nexplore_samples*explore_cost)*(
            best_allocation_frac)
        if (total_budget-nexplore_samples*explore_cost) < 0:
            raise RuntimeError("Exploitation budget is negative")
        # recorrect for normalization of nsamples by cost
        best_allocation = np.floor(
            target_cost_fractions/best_subset_group_costs).astype(int)

        # todo change subset to groups when reffereing to model groups
        # passed to multilevel blue. This requires changing notion of group
        # above which refers to subsets of a model group (using new definition)
        return (nexplore_samples, best_subset, best_cost, best_beta_Sp,
                best_Sigma_S, best_allocation, best_loss, best_blue_variance,
                best_exploit_budget, best_subset_group_costs)

    def explore_deprecated(self, total_budget, subsets, alpha=4,
                           constraint_reg=0):
        if self._costs is None:
            # todo extract costs from models
            # costs = ...
            raise NotImplementedError()
        subsets, max_ncovariates = self._validate_subsets(subsets)

        nexplore_samples = max_ncovariates+2
        samples = self.rvs(nexplore_samples)
        values = np.hstack([model(samples) for model in self.models])
        # will fail if model does not return ndarray (nsamples, nqoi=1)
        assert values.ndim == 2

        while True:
            nexplore_samples_prev = nexplore_samples
            result = self._explore_step(
                total_budget, subsets, values, alpha, self._reg_blue,
                self._constraint_reg)
            # (nexplore_samples, best_subset, best_cost, best_beta_Sp,
            # best_Sigma_S, best_allocation, best_loss,
            # best_blue_variance) = result
            nexplore_samples = result[0]
            if nexplore_samples - nexplore_samples_prev <= 0:
                break
            # TODO is using archive model then rvs must not select any
            # previously selected samples
            nnew_samples = nexplore_samples-nexplore_samples_prev
            new_samples = self.rvs(nnew_samples)
            new_values = [
                model(new_samples) for model in self.models]
            samples = np.hstack((samples, new_samples))
            values = np.vstack((values, np.hstack(new_values)))
            last_result = result
        return samples, values, last_result  # akil returns result

    def explore(self, total_budget, subsets, alpha=4):
        if self._costs is None:
            # todo extract costs from models
            # costs = ...
            raise NotImplementedError()
        subsets, max_ncovariates = self._validate_subsets(subsets)

        nexplore_samples = max_ncovariates+2
        nexplore_samples_prev = 0
        while ((nexplore_samples - nexplore_samples_prev > 0)):
            nnew_samples = nexplore_samples-nexplore_samples_prev
            new_samples = self.rvs(nnew_samples)
            new_values = [
                model(new_samples) for model in self.models]
            if nexplore_samples_prev == 0:
                samples = new_samples
                values = np.hstack(new_values)
                # will fail if model does not return ndarray (nsamples, nqoi=1)
                assert values.ndim == 2
            else:
                samples = np.hstack((samples, new_samples))
                values = np.vstack((values, np.hstack(new_values)))
            nexplore_samples_prev = nexplore_samples
            result = self._explore_step(
                total_budget, subsets, values, alpha, self._reg_blue,
                self._constraint_reg)
            nexplore_samples = result[0]
            last_result = result
        return samples, values, last_result  # akil returns result

    def exploit(self, result):
        best_subset = result[1]
        beta_Sp, Sigma_best_S, nsamples_per_subset = result[3:6]
        Psi, _ = BLUE_Psi(
            Sigma_best_S, None, self._reg_blue, self.subsets,
            nsamples_per_subset)
        # use +1 to accound for subset indexing only lf models
        values = BLUE_evaluate_models(
            self.rvs, [self.models[s+1] for s in best_subset],
            self.subsets, nsamples_per_subset)
        rhs = BLUE_RHS(self.subsets, Sigma_best_S, values)
        beta_S = beta_Sp[1:]
        return np.linalg.multi_dot(
            (beta_S.T, np.linalg.lstsq(Psi, rhs, rcond=None)[0])) + beta_Sp[0]

    @staticmethod
    def _explore_result_to_dict(result):
        result = {
            "nexplore_samples": result[0], "subset": result[1],
            "subset_cost": result[2], "beta_Sp": result[3],
            "sigma_S": result[4], "nsamples_per_subset": result[5],
            "loss": result[6], "BLUE_variance": result[7],
            "exploit_budget": result[8], "subset_costs": result[9]}
        return result

    def estimate(self, total_budget, subsets=None, return_dict=True):
        samples, values, result = self.explore(total_budget, subsets)
        mean = self.exploit(result)
        if not return_dict:
            return mean, values, result
        # package up result
        result = self._explore_result_to_dict(result)
        return mean, values, result

    def __repr__(self):
        if self.optimized_criteria is None:
            return "{0}".format(self.__class__.__name__)
        return "{0}(stat={1}, MSE={2:.3g}, target_cost={3:.5g}, ratios={4})".format(
            self.__class__.__name__, 
            self.optimized_varuabce, self.rounded_target_cost,
            self.nsample_ratios.numpy())


monte_carlo_estimators = {"acvmf": ACVMFEstimator,
                          "acvis": ACVISEstimator,
                          "mfmc": MFMCEstimator,
                          "mlmc": MLMCEstimator,
                          "acvgmf": ACVGMFEstimator,
                          "acvgmfb": ACVGMFBEstimator,
                          "mc": MCEstimator,
                          "mlblue": MLBLUEstimator}


def get_estimator(estimator_type, cov, costs, variable, max_nmodels=None,
                  **kwargs):
    """
    Initialize an monte-carlo estimator.
    """
    if estimator_type not in monte_carlo_estimators:
        msg = f"Estimator {estimator_type} not supported"
        msg += f"Must be one of {monte_carlo_estimators.keys()}"
        raise ValueError(msg)
    if max_nmodels is None:
        return monte_carlo_estimators[estimator_type](
            cov, costs, variable, **kwargs)
    return BestModelSubsetEstimator(
        estimator_type, cov, costs, variable, max_nmodels, **kwargs)
