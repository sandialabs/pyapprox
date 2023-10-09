import numpy as np
from functools import partial

from pyapprox.surrogates.interp.sparse_grid import (
    get_sparse_grid_samples_and_weights)
from pyapprox.surrogates.orthopoly.quadrature import (
    clenshaw_curtis_rule_growth, leja_growth_rule,
    one_point_growth_rule)
from pyapprox.surrogates.polychaos.gpc import (
    get_univariate_quadrature_rules_from_variable)
from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.variables.transforms import AffineTransform
from pyapprox.surrogates.interp.tensorprod import (
    get_univariate_leja_quadrature_rules_from_variable,
    get_tensor_product_piecewise_polynomial_quadrature_rule)
from pyapprox.expdesign.low_discrepancy_sequences import (
    sobol_sequence, halton_sequence)


def _nested_1D_quadrature_all_level_weights(quad_rule, level):
    weights_1d = []
    x_prev = None
    for ll in range(level+1):
        x, w = quad_rule(ll)
        weights_1d.append(w)
        # ordered samples for last x
        if ll > 0:
            assert np.allclose(x_prev, x[:x_prev.shape[0]])
        x_prev = x
    return x, weights_1d


def _quadrature_rule_growth_api(quad_rule, growth_rule, level):
    return quad_rule(growth_rule(level))


def _leja_tensorprod_integration(variable, growth_rules, levels):
    nvars = variable.num_vars()
    univariate_quadrature_rules = \
        get_univariate_leja_quadrature_rules_from_variable(
            variable, growth_rules, levels,
            return_weights_for_all_levels=False)
    nsamples = [g(l) for g, l in zip(growth_rules, levels)]
    canonical_samples, weights = \
        get_tensor_product_quadrature_rule(
            nsamples, nvars, univariate_quadrature_rules)
    var_trans = AffineTransform(variable)
    samples = var_trans.map_from_canonical(canonical_samples)
    return samples, weights[:, None]


def _gauss_tensorprod_integration(variable, growth_rules, levels):
    max_nsamples = np.asarray(
        [g(l+1) for g, l in zip(growth_rules, levels)])
    univariate_quadrature_rules = (
        get_univariate_quadrature_rules_from_variable(
            variable, max_nsamples, canonical=False))
    nsamples = [g(l) for g, l in zip(growth_rules, levels)]
    samples, weights = \
        get_tensor_product_quadrature_rule(
            nsamples, variable.num_vars(), univariate_quadrature_rules)
    return samples, weights[:, None]


def _piecewise_poly_tensorprod_integration(
        degree, variable, growth_rules, levels):
    nsamples = [g(l) for g, l in zip(growth_rules, levels)]
    if variable.is_bounded_continuous_variable():
        alpha = 1
    else:
        alpha = 1-1e-6
    # new_ranges = variable.get_statistics("interval", alpha).flatten()
    new_ranges = []
    from pyapprox.variables.marginals import is_bounded_continuous_variable
    for rv in variable.marginals():
        if is_bounded_continuous_variable(rv):
            alpha = 1
        else:
            alpha = 1-1e-6
        new_ranges.append(rv.interval(alpha))
    new_ranges = np.hstack(new_ranges)
    print(new_ranges)
    x_quad, w_quad = \
        get_tensor_product_piecewise_polynomial_quadrature_rule(
            nsamples, new_ranges, degree)
    w_quad *= variable.pdf(x_quad)[:, 0]
    return x_quad, w_quad[:, None]


def integrate(method, variable, *args, **kwargs):
    methods = [
        "sparsegrid", "tensorproduct", "quasimontecarlo", "montecarlo"]
    if method == "sparsegrid" or method == "tensorproduct":
        growth_rules = {
            "clenshaw_curtis": clenshaw_curtis_rule_growth,
            "two_point":  leja_growth_rule,
            "one_point": one_point_growth_rule}
        # TODO allow different growth rules per dimension
        nvars = variable.num_vars()
        growth = kwargs.get("growth", "one_point")
        growth_rules = [growth_rules[growth]]*nvars
        levels = kwargs.get("levels", [2]*nvars)
        levels = np.atleast_1d(levels)
        if levels.shape[0] == 1:
            levels = np.full((nvars), levels[0])
        assert len(levels) == nvars
        if method == "sparsegrid":
            if not np.allclose(levels, levels[0]):
                raise ValueError("only isotropic sparse grids supported")
            univariate_quadrature_rules = \
                get_univariate_leja_quadrature_rules_from_variable(
                    variable, growth_rules, levels, **kwargs.get("opts", {}))
            canonical_samples, weights, data_structures = \
                get_sparse_grid_samples_and_weights(
                    nvars, levels[0], univariate_quadrature_rules,
                    growth_rules)
            var_trans = AffineTransform(variable)
            samples = var_trans.map_from_canonical(canonical_samples)
            return samples, weights
        else:
            # TODO allow piecewise poly rules
            # TODO make rule type per dimension
            rule = kwargs.get("rule", "gauss")
            rules = {
                "gauss": _gauss_tensorprod_integration,
                "leja": _leja_tensorprod_integration,
                "linear": partial(_piecewise_poly_tensorprod_integration, 1),
                "quadratic": partial(
                    _piecewise_poly_tensorprod_integration, 2)}
            if rule not in rules:
                msg = f"rule: {rule} not supported. \n"
                msg += f"select from {rules.keys()}"
                raise ValueError(msg)
            # if rule == "linear" or rule == "quadratic":
            #     assert growth == "clenshaw_curtis"
            return rules[rule](variable, growth_rules, levels)

    if method == "quasimontecarlo":
        nsamples = int(kwargs["nsamples"])
        start_index = kwargs.get("startindex", 1)
        rule = kwargs.get("rule", "sobol")
        if rule == "sobol":
            samples = sobol_sequence(
                variable.num_vars(), nsamples, start_index, variable)
        elif rule == "halton":
            samples = halton_sequence(
                variable.num_vars(), nsamples, start_index, variable)
        else:
            raise NotImplementedError(f"QMC rule {rule} not implemented")
        return samples, np.full((nsamples, 1), 1/nsamples)

    if method == "montecarlo":
        nsamples = kwargs["nsamples"]
        return (variable.rvs(nsamples),
                np.full((int(nsamples), 1), 1/int(nsamples)))

    msg = f"Method: {method} not supported. Choose from {methods}"
    raise NotImplementedError(msg)
