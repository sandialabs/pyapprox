import numpy as np
from functools import partial
from scipy import stats

from pyapprox.surrogates.interp.sparse_grid import get_sparse_grid_samples_and_weights
from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.surrogates.orthopoly.quadrature import leja_growth_rule
from pyapprox.surrogates.interp.manipulate_polynomials import get_indices_double_set
from pyapprox.variables.density import tensor_product_pdf, beta_pdf
from pyapprox.surrogates.orthopoly.leja_quadrature import (
    get_univariate_leja_quadrature_rule
)


def evaluate_mixture_model(mixtures, samples):
    values = np.zeros((samples.shape[1]))
    for ii in range(len(mixtures)):
        values += mixtures[ii](samples)
    return values/len(mixtures)


def sample_mixture(mixture_samplers, num_vars, num_samples):
    num_mixtures = len(mixture_samplers)
    samples = np.empty((num_vars, num_samples))
    mixture_choices = np.random.randint(0, num_mixtures, num_samples)
    for ii in range(num_mixtures):
        II = np.where(mixture_choices == ii)[0]
        samples[:, II] = mixture_samplers[ii](II.shape[0])
    return samples


def get_mixture_tensor_product_gauss_quadrature(
        mixture_univariate_quadrature_rules, nquad_samples_1d, num_vars):
    """
    Assumes a given mixture is tensor product of one univariate density
    """
    num_mixtures = len(mixture_univariate_quadrature_rules)

    samples = np.empty((num_vars, 0), dtype=float)
    weights = np.empty((0), dtype=float)
    for ii in range(num_mixtures):
        samples_ii, weights_ii = get_tensor_product_quadrature_rule(
            nquad_samples_1d, num_vars,
            mixture_univariate_quadrature_rules[ii])
        samples = np.hstack((samples, samples_ii))
        weights = np.hstack((weights, weights_ii))
    return samples, weights/num_mixtures


def get_mixture_sparse_grid_quadrature_rule(
        mixture_univariate_quadrature_rules,
        univariate_growth_rules,
        num_vars, level,
        sparse_grid_subspace_indices=None):
    """
    Assumes a given mixture is tensor product of one univariate density

    Warning This can return non-unique points. However this function
    is only intended to be used for cheaply evaluted functions
    """
    num_mixtures = len(mixture_univariate_quadrature_rules)
    assert num_mixtures == len(univariate_growth_rules)

    samples = np.empty((num_vars, 0), dtype=float)
    weights = np.empty((0), dtype=float)
    for ii in range(num_mixtures):
        samples_ii, weights_ii, data_structures =\
            get_sparse_grid_samples_and_weights(
                num_vars, level, mixture_univariate_quadrature_rules[ii],
                univariate_growth_rules[ii],
                sparse_grid_subspace_indices=sparse_grid_subspace_indices)
        samples = np.hstack((samples, samples_ii))
        weights = np.hstack((weights, weights_ii))
    return samples, weights/num_mixtures


def get_leja_univariate_quadrature_rules_of_beta_mixture(
        rv_params, growth_rule=leja_growth_rule, basename=None,
        return_weights_for_all_levels=True):
    mixtures = []
    mixture_univariate_quadrature_rules = []
    for ii in range(len(rv_params)):
        alpha_stat, beta_stat = rv_params[ii]
        univariate_beta_pdf = partial(beta_pdf, alpha_stat, beta_stat)
        random_variable_density = partial(
            tensor_product_pdf, univariate_pdfs=univariate_beta_pdf)
        mixtures.append(random_variable_density)

        if basename is not None:
            samples_filename = basename + '-%d.npz' % (ii)
        else:
            samples_filename = None
        univariate_quadrature_rule = get_univariate_leja_quadrature_rule(
            stats.beta(alpha_stat, beta_stat), leja_growth_rule,
            return_weights_for_all_levels=return_weights_for_all_levels)
        # univariate_quadrature_rule = partial(
        #    beta_leja_quadrature_rule, rv_params[ii][0],
        #    rv_params[ii][1], growth_rule=growth_rule,
        #    samples_filename=samples_filename,
        #    return_weights_for_all_levels=return_weights_for_all_levels)

        mixture_univariate_quadrature_rules.append(univariate_quadrature_rule)

    return mixtures, mixture_univariate_quadrature_rules


def compute_grammian_of_mixture_models_using_sparse_grid_quadrature(
        basis_matrix_func, indices, mixture_univariate_quadrature_rules,
        mixture_univariate_growth_rules, num_vars):

    double_set_indices = get_indices_double_set(indices)
    samples, weights = get_mixture_sparse_grid_quadrature_rule(
        mixture_univariate_quadrature_rules,
        mixture_univariate_growth_rules, num_vars, double_set_indices.max(),
        sparse_grid_subspace_indices=double_set_indices)

    basis_matrix = basis_matrix_func(samples)
    moment_matrix = np.dot(basis_matrix.T*weights, basis_matrix)

    # assert (np.min(weights)>=0)
    # moment_matrix = np.dot(np.diag(np.sqrt(weights)),basis_matrix)
    return moment_matrix
