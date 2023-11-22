"""
Functions that can be re-used throughout the multi-fidelity tutorials
"""
import copy
from itertools import cycle

import torch
import numpy as np

from pyapprox.multifidelity.multioutput_monte_carlo import (
    compute_variance_reductions)
from pyapprox.util.visualization import mathrm_label


def plot_control_variate_variance_ratios(
        cv_variance_reductions, cv_labels, ax):
    for var_red, label in zip(cv_variance_reductions, cv_labels):
        ax.axhline(y=1/var_red, linestyle='--', c='k')
        ax.text(1, 1/var_red*1.1, label, fontsize=16)
    ax.axhline(y=1, linestyle='--', c='k')
    # text position is in data coordinates
    # to speficy coords in (0,0 is lower-left and 1,1 is upper-right)
    # add kwarg: transform=ax.transAxes
    ax.text(1, 1, mathrm_label("MC"), fontsize=16)
    ax.set_yscale('log')


def plot_estimator_variance_ratios_for_polynomial_ensemble(
        estimators, est_labels, ax):
    nhf_samples = 1
    factors = np.arange(22)
    npartition_ratio_base = np.array([2, 2, 2, 2])
    optimized_estimators = []
    lines = ["-", "--", ":", "-."]
    linecycler = cycle(lines)
    colors = ["k", "r", "b"]
    colorcycler = cycle(colors)
    for est in estimators:
        est_copies = []
        for factor in factors:
            npartition_samples = torch.as_tensor(
                nhf_samples*np.hstack((1, npartition_ratio_base*2**factor)),
                dtype=torch.double)
            if est._tree_depth is not None:
                ests_per_factor = []
                for ii, index in enumerate(est.get_all_recursion_indices()):
                    est_copy = copy.deepcopy(est)
                    est_copy._set_recursion_index(index)
                    est_copy._set_optimized_params_base(
                        npartition_samples,
                        est_copy._compute_nsamples_per_model(npartition_samples),
                        est_copy._estimator_cost(npartition_samples))
                    ests_per_factor.append(est_copy)
                best_idx = np.argmin(
                    [e._optimized_criteria for e in ests_per_factor])
                est_copy = ests_per_factor[best_idx]
            else:
                est_copy = copy.deepcopy(est)
                est_copy._set_optimized_params_base(
                    npartition_samples,
                    est_copy._compute_nsamples_per_model(npartition_samples),
                    est_copy._estimator_cost(npartition_samples))
            est_copies.append(est_copy)
        optimized_estimators.append(est_copies)

    for ii in range(len(optimized_estimators)):
        variance_reductions = compute_variance_reductions(
            optimized_estimators[ii], nhf_samples=nhf_samples)[0]
        est_costs = [est._estimator_cost(est._rounded_npartition_samples)
                     for est in optimized_estimators[ii]]
        ax.loglog(est_costs, 1/variance_reductions,
                  label=est_labels[ii], linestyle=next(linecycler),
                  c=next(colorcycler))
    ax.legend()


def plot_estimator_variance_ratios_for_polynomial_ensemble_nspm(
        estimators, est_labels, ax):
    """
    Reproduce plots from original ACV paper by Gorodetsky et al.
    This can only be used if the estimator has the function
    native_ratios_to_npartition_ratios, i.e. for MLMC and MFMC estimators
    It is hard to create such a function for general PACV estimators
    when optimizing the number of samples per independent partition
    which was not done in the original paper. This function is mainly 
    for posterity.
    """
    nhf_samples = 1
    factors = np.arange(10)
    nsamples_per_model_ratios_base = np.array([2, 4, 8, 16])
    optimized_estimators = []
    for est in estimators:
        est_copies = []
        for factor in factors:
            # set npartitions per sample based on the number of
            # samples_per_model
            nsample_per_model_ratios = np.array(
                [r*(2**factor) for r in nsamples_per_model_ratios_base])
            npartition_ratios = est._native_ratios_to_npartition_ratios(
                nsample_per_model_ratios)
            npartition_samples = torch.as_tensor(
                nhf_samples*np.hstack((1, npartition_ratios)),
                dtype=torch.double)
            est_copy = copy.deepcopy(est)
            est_copy._set_optimized_params_base(
                npartition_samples,
                est_copy._compute_nsamples_per_model(npartition_samples),
                est_copy._estimator_cost(npartition_samples))
            est_copies.append(est_copy)
        optimized_estimators.append(est_copies)

    for ii in range(len(optimized_estimators)):
        variance_reductions = compute_variance_reductions(
            optimized_estimators[ii], nhf_samples=nhf_samples)[0]
        ax.semilogy(np.arange(len(factors)), 1/variance_reductions,
                    label=est_labels[ii])
