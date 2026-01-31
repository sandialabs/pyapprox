"""Numerical verification utilities for multifidelity estimators.

This module provides functions to numerically verify that analytical
estimator covariance formulas match Monte Carlo estimates.
"""
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Generic, List, Tuple, Union

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


class EstimatorComponentExtractor(ABC, Generic[Array]):
    """Abstract base for extracting estimator components.

    Different estimator types (ACV, CV, MC) require different methods
    to extract the HF estimate and control variate discrepancies.
    """

    @abstractmethod
    def extract(
        self,
        est: Any,
        values_per_model: List[Array],
        mc_est: Callable[[Array], Array],
    ) -> Tuple[Array, Array, Array]:
        """Extract estimator value, HF estimate, and discrepancies.

        Parameters
        ----------
        est : EstimatorProtocol
            The estimator instance.
        values_per_model : List[Array]
            Model outputs for each model.
        mc_est : Callable[[Array], Array]
            Monte Carlo sample estimate function.

        Returns
        -------
        est_val : Array
            Full estimator value. Shape: (nstats,)
        Q : Array
            HF Monte Carlo estimate. Shape: (nstats,)
        delta : Array
            Control variate discrepancies. Shape: (nstats * (nmodels-1),)
        """
        pass


class ACVComponentExtractor(EstimatorComponentExtractor[Array]):
    """Extract components from ACV estimators."""

    def extract(
        self,
        est: Any,
        values_per_model: List[Array],
        mc_est: Callable[[Array], Array],
    ) -> Tuple[Array, Array, Array]:
        """Extract components using ACV sample separation."""
        bkd: Backend[Array] = est._bkd
        est_val: Array = est(values_per_model)
        acv_values: List[Array] = est._separate_values_per_model(values_per_model)
        Q: Array = mc_est(acv_values[1])  # Z_0 (HF unstarred samples)
        delta: Array = bkd.hstack([
            mc_est(acv_values[2 * ii]) - mc_est(acv_values[2 * ii + 1])
            for ii in range(1, est.nmodels())
        ])
        return est_val, Q, delta


class CVComponentExtractor(EstimatorComponentExtractor[Array]):
    """Extract components from CV estimators."""

    def extract(
        self,
        est: Any,
        values_per_model: List[Array],
        mc_est: Callable[[Array], Array],
    ) -> Tuple[Array, Array, Array]:
        """Extract components using CV structure."""
        bkd: Backend[Array] = est._bkd
        est_val: Array = est(values_per_model)
        Q: Array = mc_est(values_per_model[0])
        delta: Array = bkd.hstack([
            mc_est(values_per_model[ii]) - est._lowfi_stats[ii - 1]
            for ii in range(1, est.nmodels())
        ])
        return est_val, Q, delta


class MCComponentExtractor(EstimatorComponentExtractor[Array]):
    """Extract components from MC estimators."""

    def extract(
        self,
        est: Any,
        values_per_model: List[Array],
        mc_est: Callable[[Array], Array],
    ) -> Tuple[Array, Array, Array]:
        """Extract components - MC has no control variates."""
        est_val: Array = est(values_per_model[0])
        Q: Array = mc_est(values_per_model[0])
        delta: Array = Q * 0
        return est_val, Q, delta


def _get_component_extractor(est: Any) -> EstimatorComponentExtractor[Any]:
    """Get appropriate component extractor for estimator type.

    Uses isinstance checks to determine estimator type and return
    the corresponding extractor.

    Parameters
    ----------
    est : EstimatorProtocol
        The estimator instance.

    Returns
    -------
    EstimatorComponentExtractor
        Extractor appropriate for the estimator type.
    """
    from pyapprox.typing.stats.estimators.acv import ACVEstimator
    from pyapprox.typing.stats.estimators.cv import CVEstimator

    if isinstance(est, ACVEstimator):
        return ACVComponentExtractor()
    if isinstance(est, CVEstimator):
        return CVComponentExtractor()
    return MCComponentExtractor()


def _estimate_components(
    variable: Any,
    est: Any,
    funs: List[Callable[..., Any]],
    trial_idx: int,
) -> Tuple[Array, Array, Array]:
    """Compute estimator components for a single trial.

    Parameters
    ----------
    variable : RandomVariable
        Variable with rvs() and _rvs_given_random_states() methods.
    est : EstimatorProtocol
        Estimator with allocated samples.
    funs : List[Callable]
        Model functions.
    trial_idx : int
        Trial index for reproducible random seeding.

    Returns
    -------
    est_val : Array
        Estimator value for this trial. Shape: (nstats,)
    Q : Array
        HF Monte Carlo estimate. Shape: (nstats,)
    delta : Array
        Control variate discrepancies. Shape: (nstats * (nmodels-1),)
    """
    bkd: Backend[Array] = est._bkd

    # Use RandomState for reproducibility in parallel
    random_states = [
        np.random.RandomState(trial_idx * variable.nvars() + jj)
        for jj in range(variable.nvars())
    ]

    samples_per_model: List[Array] = est.generate_samples_per_model(
        partial(variable._rvs_given_random_states, random_states=random_states)
    )
    values_per_model: List[Array] = [
        bkd.asarray(fun(samples))
        for fun, samples in zip(funs, samples_per_model)
    ]

    extractor = _get_component_extractor(est)
    return extractor.extract(est, values_per_model, est._stat.sample_estimate)


def _estimate_components_loop(
    variable: Any,
    ntrials: int,
    est: Any,
    funs: List[Callable[..., Any]],
    max_eval_concurrency: int = 1,
) -> Tuple[Array, Array, Array]:
    """Run estimation loop for multiple trials.

    Parameters
    ----------
    variable : RandomVariable
        Variable with rvs() method.
    ntrials : int
        Number of MC trials.
    est : EstimatorProtocol
        Estimator with allocated samples.
    funs : List[Callable]
        Model functions.
    max_eval_concurrency : int
        Number of processors (currently only 1 supported).

    Returns
    -------
    estimator_vals : Array
        Shape: (ntrials, nstats)
    Q : Array
        Shape: (ntrials, nstats)
    delta : Array
        Shape: (ntrials, nstats * (nmodels-1))
    """
    bkd: Backend[Array] = est._bkd
    Q_list: List[Array] = []
    delta_list: List[Array] = []
    estimator_vals_list: List[Array] = []

    for ii in range(ntrials):
        est_val, Q_val, delta_val = _estimate_components(variable, est, funs, ii)
        estimator_vals_list.append(est_val)
        Q_list.append(Q_val)
        delta_list.append(delta_val)

    return (
        bkd.stack(estimator_vals_list),
        bkd.stack(Q_list),
        bkd.stack(delta_list),
    )


def numerically_compute_estimator_variance(
    funs: List[Callable[..., Any]],
    variable: Any,
    est: Any,
    ntrials: int = int(1e3),
    max_eval_concurrency: int = 1,
    return_all: bool = False,
) -> Union[
    Tuple[Array, Array, Array, Array],
    Tuple[Array, Array, Array, Array, Array, Array, Array],
]:
    """Numerically estimate the variance of a multifidelity estimator.

    This function runs many independent trials of the estimator and
    computes the empirical covariance, which can be compared against
    the analytical covariance from the estimator.

    Parameters
    ----------
    funs : List[Callable]
        Model functions with signature fun(samples) -> Array (nsamples, nqoi)
    variable : RandomVariable
        Variable with rvs() method.
    est : EstimatorProtocol
        Estimator with allocated samples.
    ntrials : int
        Number of MC trials.
    max_eval_concurrency : int
        Number of processors (currently only 1 supported).
    return_all : bool
        If True, also return individual trial values.

    Returns
    -------
    hf_covar_numer : Array
        MC estimate of HF estimator covariance.
    hf_covar : Array
        Analytical HF estimator covariance.
    covar_numer : Array
        MC estimate of full estimator covariance.
    covar : Array
        Analytical full estimator covariance.
    est_vals : Array (optional)
        Individual trial estimator values. Shape: (ntrials, nstats)
    Q0 : Array (optional)
        Individual trial HF estimates. Shape: (ntrials, nstats)
    delta : Array (optional)
        Individual trial discrepancies.
    """
    bkd: Backend[Array] = est._bkd
    est_vals, Q0, delta = _estimate_components_loop(
        variable, int(ntrials), est, funs, max_eval_concurrency
    )

    hf_covar_numer: Array = bkd.cov(Q0, ddof=1, rowvar=False)
    hf_covar: Array = est._stat.high_fidelity_estimator_covariance(
        est.npartition_samples()[0]
    )
    covar_numer: Array = bkd.cov(est_vals, ddof=1, rowvar=False)
    covar: Array = est._covariance_from_npartition_samples(est.npartition_samples())

    if not return_all:
        return hf_covar_numer, hf_covar, covar_numer, covar
    return hf_covar_numer, hf_covar, covar_numer, covar, est_vals, Q0, delta
