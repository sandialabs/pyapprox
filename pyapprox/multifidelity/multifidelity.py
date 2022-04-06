import numpy as np
from pyapprox.surrogates.approximate import adaptive_approximate


def adaptive_approximate_multi_index_sparse_grid(fun, variable, options):
    """
    A light weight wrapper for building multi-index approximations. 
    Some checks are made to ensure certain required options have been provided.
    See :func:`pyapprox.surrogates.approximate.adaptive_approximate_sparse_grid` for more
    details.
    """
    assert 'config_variables_idx' in options
    assert 'config_var_trans' in options
    sparse_grid = adaptive_approximate(fun, variable, 'sparse_grid', options)
    return sparse_grid
