"""
Functions that can be re-used throughout the multi-fidelity tutorials
"""
import copy

import torch
import numpy as np

from pyapprox.multifidelity.multioutput_monte_carlo import (
    compute_variance_reductions)
from pyapprox.util.visualization import mathrm_label


def plot_control_variate_variance_ratios(
        cv_variance_reductions, cv_labels, ax):
    for var_red, label in zip(cv_variance_reductions, cv_labels):
        ax.axhline(y=1/var_red, linestyle='--', c='k')
        ax.text(0.01, 1/var_red*1.1, label, fontsize=16)
    ax.axhline(y=1, linestyle='--', c='k')
    ax.text(.01, 1, mathrm_label("MC"), fontsize=16)
    ax.set_yscale('log')


def plot_estimator_variance_ratios_for_polynomial_ensemble(
        estimators, est_labels, ax):
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
