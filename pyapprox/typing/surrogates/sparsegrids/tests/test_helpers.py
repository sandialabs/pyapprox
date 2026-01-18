"""Test helpers for sparse grid parametrized tests.

This module provides reusable functions for creating test configurations
for sparse grid tests with different:
- Marginal distributions (uniform, gaussian, beta, gamma)
- Joint configurations (2D, 3D, 4D, mixed)
- Quadrature rules (Gauss, Leja)
- Growth rules (linear, Clenshaw-Curtis)

Example usage:
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> pce = create_test_pce(joint, level=2, nqoi=1, bkd=bkd)
    >>> grid = create_test_grid_gauss(joint, level=2, bkd=bkd)
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from pyapprox.typing.probability import (
    BetaMarginal,
    GammaMarginal,
    GaussianMarginal,
    IndependentJoint,
    UniformMarginal,
)
from pyapprox.typing.surrogates.affine.expansions.pce import (
    create_pce_from_marginals,
    PolynomialChaosExpansion,
)
from pyapprox.typing.surrogates.affine.indices import (
    DoublePlusOneGrowthRule,
    IndexGrowthRule,
    LinearGrowthRule,
)
from pyapprox.typing.surrogates.sparsegrids import (
    IsotropicCombinationSparseGrid,
    create_basis_factories,
)
from pyapprox.typing.util.backends.protocols import Array, Backend


# =============================================================================
# Marginal factory functions
# =============================================================================

MARGINAL_FACTORIES: Dict[str, Callable[[Backend[Array]], Any]] = {
    "uniform": lambda bkd: UniformMarginal(-1.0, 1.0, bkd),
    "gaussian": lambda bkd: GaussianMarginal(0.0, 1.0, bkd),
    "beta": lambda bkd: BetaMarginal(2.0, 5.0, bkd),
    "gamma": lambda bkd: GammaMarginal(3.0, 2.0, bkd),
}


# =============================================================================
# Joint distribution configurations
# =============================================================================

# (name, list of marginal names)
JOINT_CONFIGS: List[Tuple[str, List[str]]] = [
    ("2d_uniform", ["uniform", "uniform"]),
    ("2d_gaussian", ["gaussian", "gaussian"]),
    ("2d_beta", ["beta", "beta"]),
    ("2d_gamma", ["gamma", "gamma"]),
    ("2d_mixed_ug", ["uniform", "gaussian"]),
    ("2d_mixed_ub", ["uniform", "beta"]),
    ("3d_uniform", ["uniform", "uniform", "uniform"]),
    ("3d_gamma", ["gamma", "gamma", "gamma"]),
    ("3d_mixed", ["uniform", "gaussian", "beta"]),
    ("4d_uniform", ["uniform", "uniform", "uniform", "uniform"]),
]

# Bounded-only configs (for future piecewise tests)
BOUNDED_JOINT_CONFIGS = [
    c for c in JOINT_CONFIGS if all(m in ["uniform", "beta"] for m in c[1])
]


# =============================================================================
# Growth rule registry
# =============================================================================

GROWTH_RULES: Dict[str, IndexGrowthRule] = {
    "linear_1_1": LinearGrowthRule(scale=1, shift=1),  # d = l + 1
    "linear_2_1": LinearGrowthRule(scale=2, shift=1),  # d = 2*l + 1
    "clenshaw_curtis": DoublePlusOneGrowthRule(),  # d = 2^l + 1 (nested)
}


# =============================================================================
# Joint distribution creation
# =============================================================================


def create_test_joint(
    config_name: str, bkd: Backend[Array]
) -> IndependentJoint[Array]:
    """Create IndependentJoint from config name.

    Parameters
    ----------
    config_name : str
        Name of the joint configuration (e.g., "2d_uniform", "3d_mixed").
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    IndependentJoint[Array]
        Joint distribution with the specified marginals.

    Raises
    ------
    StopIteration
        If config_name is not found in JOINT_CONFIGS.

    Example
    -------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> joint.nvars()
    2
    """
    config = next(c for c in JOINT_CONFIGS if c[0] == config_name)
    marginals = [MARGINAL_FACTORIES[m](bkd) for m in config[1]]
    return IndependentJoint(marginals, bkd)


# =============================================================================
# PCE creation
# =============================================================================


def create_test_pce(
    joint: IndependentJoint[Array],
    level: int,
    nqoi: int,
    bkd: Backend[Array],
    seed: int = 42,
) -> PolynomialChaosExpansion[Array]:
    """Create PCE with hyperbolic index set from joint distribution.

    Creates a PCE with random coefficients that can be used as an
    exact target function for sparse grid interpolation tests.

    Parameters
    ----------
    joint : IndependentJoint[Array]
        Joint distribution defining the marginals.
    level : int
        Maximum total degree of the PCE index set.
    nqoi : int
        Number of quantities of interest.
    bkd : Backend[Array]
        Computational backend.
    seed : int, optional
        Random seed for coefficient generation. Default: 42.

    Returns
    -------
    PolynomialChaosExpansion[Array]
        PCE with random coefficients.

    Example
    -------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> pce = create_test_pce(joint, level=3, nqoi=1, bkd=bkd)
    >>> pce.nterms()  # Number of terms in the PCE
    10
    """
    pce = create_pce_from_marginals(
        joint.marginals(), max_level=level, bkd=bkd, nqoi=nqoi
    )
    # Set random coefficients
    nterms = pce.nterms()
    np.random.seed(seed)
    coeffs = bkd.asarray(np.random.randn(nterms, nqoi))  # type: ignore[arg-type]
    pce.set_coefficients(coeffs)
    return pce


# =============================================================================
# Sparse grid creation
# =============================================================================


def create_test_grid_gauss(
    joint: IndependentJoint[Array],
    level: int,
    bkd: Backend[Array],
    growth: IndexGrowthRule | None = None,
) -> IsotropicCombinationSparseGrid[Array]:
    """Create IsotropicCombinationSparseGrid with Gauss quadrature.

    Parameters
    ----------
    joint : IndependentJoint[Array]
        Joint distribution defining the marginals.
    level : int
        Sparse grid level (controls number of subspaces).
    bkd : Backend[Array]
        Computational backend.
    growth : IndexGrowthRule, optional
        Growth rule for mapping level to polynomial degree.
        Default: LinearGrowthRule(scale=1, shift=1).

    Returns
    -------
    IsotropicCombinationSparseGrid[Array]
        Sparse grid with Gauss quadrature points.

    Example
    -------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> grid = create_test_grid_gauss(joint, level=2, bkd=bkd)
    >>> grid.nsubspaces()
    6
    """
    factories = create_basis_factories(joint.marginals(), bkd, "gauss")
    if growth is None:
        growth = LinearGrowthRule(scale=1, shift=1)
    return IsotropicCombinationSparseGrid(bkd, factories, growth, level=level)


def create_test_grid_leja(
    joint: IndependentJoint[Array],
    level: int,
    bkd: Backend[Array],
    growth: IndexGrowthRule | None = None,
) -> IsotropicCombinationSparseGrid[Array]:
    """Create IsotropicCombinationSparseGrid with Leja quadrature.

    Parameters
    ----------
    joint : IndependentJoint[Array]
        Joint distribution defining the marginals.
    level : int
        Sparse grid level (controls number of subspaces).
    bkd : Backend[Array]
        Computational backend.
    growth : IndexGrowthRule, optional
        Growth rule for mapping level to polynomial degree.
        Default: LinearGrowthRule(scale=1, shift=1).

    Returns
    -------
    IsotropicCombinationSparseGrid[Array]
        Sparse grid with Leja quadrature points.

    Example
    -------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> grid = create_test_grid_leja(joint, level=2, bkd=bkd)
    >>> grid.nsubspaces()
    6
    """
    factories = create_basis_factories(joint.marginals(), bkd, "leja")
    if growth is None:
        growth = LinearGrowthRule(scale=1, shift=1)
    return IsotropicCombinationSparseGrid(bkd, factories, growth, level=level)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "MARGINAL_FACTORIES",
    "JOINT_CONFIGS",
    "BOUNDED_JOINT_CONFIGS",
    "GROWTH_RULES",
    "create_test_joint",
    "create_test_pce",
    "create_test_grid_gauss",
    "create_test_grid_leja",
]
