import sympy as sp
import numpy as np


def _evaluate_sp_lambda(sp_lambda, xx):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args)
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals, dtype=np.double)


def _evaluate_transient_sp_lambda(sp_lambda, xx, time):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args, time)
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals, dtype=np.double)


def _evaluate_list_of_sp_lambda(sp_lambdas, xx, as_list=False):
    # sp_lambda returns list of values from multiple functions
    vals = [_evaluate_sp_lambda(sp_lambda, xx)
            for sp_lambda in sp_lambdas]
    if as_list:
        return vals
    return np.hstack(vals)


def _evaluate_list_of_transient_sp_lambda(sp_lambdas, xx, time, as_list=False):
    # sp_lambda returns list of values from multiple functions
    vals = [_evaluate_transient_sp_lambda(sp_lambda, xx, time)
            for sp_lambda in sp_lambdas]
    if as_list:
        return vals
    return np.hstack(vals)
