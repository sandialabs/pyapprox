from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


def _evaluate_sp_lambda(sp_lambda, xx, bkd=NumpyLinAlgMixin):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args)
    if isinstance(vals, bkd.array_type()):
        return vals[:, None]
    return bkd.full((xx.shape[1], 1), vals)


def _evaluate_transient_sp_lambda(sp_lambda, xx, time, bkd=NumpyLinAlgMixin):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args, time)
    if isinstance(vals, bkd.array_type()):
        return vals[:, None]
    return bkd.full((xx.shape[1], 1), vals)


def _evaluate_list_of_sp_lambda(
        sp_lambdas, xx, as_list=False, bkd=NumpyLinAlgMixin
):
    # sp_lambda returns list of values from multiple functions
    vals = [_evaluate_sp_lambda(sp_lambda, xx, bkd)
            for sp_lambda in sp_lambdas]
    if as_list:
        return vals
    return bkd.hstack(vals)


def _evaluate_list_of_transient_sp_lambda(
        sp_lambdas, xx, time, as_list=False, bkd=NumpyLinAlgMixin
):
    # sp_lambda returns list of values from multiple functions
    vals = [_evaluate_transient_sp_lambda(sp_lambda, xx, time, bkd)
            for sp_lambda in sp_lambdas]
    if as_list:
        return vals
    return bkd.hstack(vals)
