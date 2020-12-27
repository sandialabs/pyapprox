from functools import partial

import numpy as np
from scipy import stats

import pyapprox as pya
from pyapprox.models.wrappers import (
    evaluate_1darray_function_on_2d_array, vectorize_1darray_on_2d_array)

from pyapprox.sparse_grid import plot_sparse_grid_2d
from pyapprox.configure_plots import *


def test_model_decorator_equivalency():
    univariate_variables = [stats.uniform(), ] * 8

    variable = pya.IndependentMultivariateRandomVariable(univariate_variables)

    nsamples = 100
    samples = pya.generate_independent_random_samples(variable, nsamples)

    def fun(sample):
        assert sample.ndim == 1
        return np.sum(sample**2)

    def pyapprox_fun(samples):
        values = evaluate_1darray_function_on_2d_array(fun, samples)
        return values

    @vectorize_1darray_on_2d_array
    def target_func(x):
        return np.sum(x**2)


    values = pyapprox_fun(samples)
    res = target_func(samples)

    assert np.allclose(np.array(values), np.array(
        res)), f"Results are different!\n{values}\n========\n{res}"
