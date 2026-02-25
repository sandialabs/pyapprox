"""Basis setup utilities for sparse grid tensor product subspaces.

This module provides helper functions for initializing univariate bases
in tensor product constructions, handling:
- Quadrature rule extraction from various basis types
- LagrangeBasis1D wrapper creation
- Growth rule application to multi-indices
"""

from typing import Callable, List, Tuple, Union, cast

from pyapprox.surrogates.affine.protocols import (
    Basis1DProtocol,
    IndexGrowthRuleProtocol,
)
from pyapprox.surrogates.affine.univariate.lagrange import LagrangeBasis1D
from pyapprox.util.backends.protocols import Array, Backend


def compute_npts_from_growth_rule(
    index: Array,
    growth_rules: Union[IndexGrowthRuleProtocol, List[IndexGrowthRuleProtocol]],
) -> List[int]:
    """Compute number of points per dimension from a multi-index and growth rule(s).

    Parameters
    ----------
    index : Array
        Multi-index specifying level in each dimension. Shape: (nvars,)
    growth_rules : IndexGrowthRuleProtocol or List[IndexGrowthRuleProtocol]
        Rule(s) mapping level to number of points. If a single rule, it is
        used for all dimensions. If a list, each element applies to the
        corresponding dimension.

    Returns
    -------
    List[int]
        Number of points in each dimension.

    Raises
    ------
    ValueError
        If growth_rules is a list with length not matching nvars.
    """
    nvars = len(index)

    # Handle single growth rule (apply to all dimensions)
    if not isinstance(growth_rules, list):
        return [growth_rules(int(index[dim])) for dim in range(nvars)]

    # Handle per-dimension growth rules
    if len(growth_rules) != nvars:
        raise ValueError(
            f"growth_rules list length ({len(growth_rules)}) must match nvars ({nvars})"
        )
    return [growth_rules[dim](int(index[dim])) for dim in range(nvars)]


def get_quadrature_rule(
    basis: Basis1DProtocol[Array],
) -> Callable[[int], Tuple[Array, Array]]:
    """Extract a quadrature rule callable from a basis.

    Supports multiple basis types:
    - OrthonormalPolynomial1D: uses gauss_quadrature_rule (with auto set_nterms)
    - LagrangeBasis1D: uses gauss_quadrature_rule (wraps its internal callable)
    - Bases with quadrature_rule(): wraps to ignore npoints argument

    Parameters
    ----------
    basis : Basis1DProtocol[Array]
        Any basis that provides samples/weights.

    Returns
    -------
    Callable[[int], Tuple[Array, Array]]
        Function that takes npoints and returns (samples, weights).
        For orthogonal polynomials, this automatically calls set_nterms
        before computing the quadrature rule.

    Raises
    ------
    TypeError
        If basis has no supported quadrature method.

    Notes
    -----
    For orthogonal polynomials, the returned callable ensures set_nterms
    is called with at least npoints before computing the quadrature.
    This allows the quadrature rule to be used without manual initialization.
    """
    # Check for gauss_quadrature_rule (orthogonal polys, LagrangeBasis1D)
    if hasattr(basis, "gauss_quadrature_rule"):
        gauss_rule = getattr(basis, "gauss_quadrature_rule")

        # For bases with set_nterms, wrap to ensure initialization
        if hasattr(basis, "set_nterms"):

            def wrapped_gauss_rule(npoints: int) -> Tuple[Array, Array]:
                # Ensure basis has enough terms for the quadrature
                if basis.nterms() < npoints:
                    basis.set_nterms(npoints)
                result: Tuple[Array, Array] = gauss_rule(npoints)
                return result

            return wrapped_gauss_rule
        else:
            # LagrangeBasis1D doesn't need set_nterms
            return cast(Callable[[int], Tuple[Array, Array]], gauss_rule)

    # Check for quadrature_rule() with no args (piecewise polys, DynamicPiecewiseBasis)
    if hasattr(basis, "quadrature_rule"):
        quad_method = getattr(basis, "quadrature_rule")

        # For bases with set_nterms (like DynamicPiecewiseBasis), ensure init
        if hasattr(basis, "set_nterms"):

            def wrapped_dynamic_quadrature(npoints: int) -> Tuple[Array, Array]:
                # Ensure basis has correct nterms
                if basis.nterms() != npoints:
                    basis.set_nterms(npoints)
                result: Tuple[Array, Array] = quad_method()
                return result

            return wrapped_dynamic_quadrature
        else:
            # Fixed-node piecewise poly, ignore npoints
            def wrapped_fixed_quadrature(npoints: int) -> Tuple[Array, Array]:
                result: Tuple[Array, Array] = quad_method()
                return result

            return wrapped_fixed_quadrature

    raise TypeError(
        f"Basis {type(basis).__name__} has no quadrature rule method. "
        "Expected gauss_quadrature_rule(npoints) or quadrature_rule()."
    )


def create_lagrange_from_quadrature(
    bkd: Backend[Array],
    quadrature_rule: Callable[[int], Tuple[Array, Array]],
) -> LagrangeBasis1D[Array]:
    """Create a Lagrange basis from a quadrature rule callable.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    quadrature_rule : Callable[[int], Tuple[Array, Array]]
        Function that takes npoints and returns (samples, weights).

    Returns
    -------
    LagrangeBasis1D[Array]
        Lagrange interpolation basis using quadrature points.
    """
    return LagrangeBasis1D(bkd, quadrature_rule)
