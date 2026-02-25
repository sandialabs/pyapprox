"""Test helpers for sparse grid parametrized tests.

This module provides reusable functions for creating test configurations
for sparse grid tests with different:
- Marginal distributions (uniform, gaussian, beta, gamma)
- Joint configurations (2D, 3D, 4D, mixed)
- PCE target functions (isotropic, anisotropic, additive)
- Growth rules (linear, Clenshaw-Curtis)

Example usage:
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> pce = create_test_pce(joint, level=2, nqoi=1, bkd=bkd)
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from pyapprox.probability import (
    BetaMarginal,
    GammaMarginal,
    GaussianMarginal,
    IndependentJoint,
    UniformMarginal,
)
from pyapprox.surrogates.affine.expansions.pce import (
    create_pce_from_marginals,
    get_basis_from_marginal,
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.indices import (
    CubicNestedGrowthRule,
    ClenshawCurtisGrowthRule,
    HyperbolicIndexGenerator,
    IndexGrowthRule,
    LinearGrowthRule,
)
from pyapprox.surrogates.sparsegrids import (
    create_basis_factories,
    TensorProductSubspace,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    ClenshawCurtisLagrangeFactory,
    GaussLagrangeFactory,
    LejaLagrangeFactory,
    PiecewiseFactory,
)
from pyapprox.surrogates.affine.protocols import (
    IndexGrowthRuleProtocol,
    PhysicalDomainBasis1DProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.cartesian import cartesian_product_indices


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
    "clenshaw_curtis": ClenshawCurtisGrowthRule(),  # d = 2^l + 1 (nested)
    "cubic_nested": CubicNestedGrowthRule(),  # d = 3 * 2^(l-1) + 1 (nested for cubic)
}


def _get_default_growth_rule(basis_type: str) -> IndexGrowthRuleProtocol:
    """Get the default growth rule for a basis type."""
    if basis_type == "piecewise_cubic":
        return CubicNestedGrowthRule()
    elif basis_type.startswith("piecewise_") or basis_type == "clenshaw_curtis":
        return ClenshawCurtisGrowthRule()
    else:
        return ClenshawCurtisGrowthRule()


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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
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
# Tensor product PCE creation
# =============================================================================


def create_tensor_product_pce(
    joint: IndependentJoint[Array],
    npts_1d: List[int],
    nqoi: int,
    bkd: Backend[Array],
    seed: int = 42,
) -> PolynomialChaosExpansion[Array]:
    """Create PCE with tensor product index set for testing tensor product interpolation.

    For Lagrange interpolation with n points per dimension, the interpolant
    is exact for polynomials of degree n-1 in each dimension. This creates
    a PCE with tensor product index set [0..n1-1] × [0..n2-1] × ... so that
    the sparse grid tensor product interpolation is exact.

    Parameters
    ----------
    joint : IndependentJoint[Array]
        Joint distribution defining the marginals.
    npts_1d : List[int]
        Number of points in each dimension.
    nqoi : int
        Number of quantities of interest.
    bkd : Backend[Array]
        Computational backend.
    seed : int, optional
        Random seed for coefficient generation. Default: 42.

    Returns
    -------
    PolynomialChaosExpansion[Array]
        PCE with tensor product index set and random coefficients.

    Example
    -------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> pce = create_tensor_product_pce(joint, [3, 4], nqoi=1, bkd=bkd)
    >>> pce.nterms()  # 3 * 4 = 12
    12
    """
    marginals = joint.marginals()

    # For Lagrange interpolation with n points: exact for degree n-1
    max_degrees = [n - 1 for n in npts_1d]
    dims = [d + 1 for d in max_degrees]  # Number of indices per dimension

    # Use existing cartesian_product_indices for tensor product index set
    indices = cartesian_product_indices(dims, bkd)

    # Create physical-domain polynomial bases for each marginal
    bases_1d: List[PhysicalDomainBasis1DProtocol[Any]] = [
        get_basis_from_marginal(m, bkd) for m in marginals
    ]

    # Create basis with explicit tensor product indices
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)

    # Create PCE
    pce = PolynomialChaosExpansion(basis, bkd, nqoi=nqoi)

    # Set random coefficients
    nterms = pce.nterms()
    np.random.seed(seed)
    coeffs = bkd.asarray(np.random.randn(nterms, nqoi))  # type: ignore[arg-type]
    pce.set_coefficients(coeffs)
    return pce


# =============================================================================
# Tensor product subspace creation
# =============================================================================


def create_test_tensor_product_subspace(
    joint: IndependentJoint[Array],
    npts_1d: List[int],
    bkd: Backend[Array],
    basis_type: str = "gauss",
) -> TensorProductSubspace[Array]:
    """Create single TensorProductSubspace (no Smolyak combination).

    This creates a pure tensor product grid without Smolyak sparse grid
    combination, useful for testing tensor product interpolation and
    quadrature separately from the combination technique.

    Parameters
    ----------
    joint : IndependentJoint[Array]
        Joint distribution defining the marginals.
    npts_1d : List[int]
        Number of points in each dimension.
    bkd : Backend[Array]
        Computational backend.
    basis_type : str, optional
        Type of basis factory. One of "gauss", "leja", "piecewise_linear",
        "piecewise_quadratic", "piecewise_cubic". Default: "gauss".

    Returns
    -------
    TensorProductSubspace[Array]
        Tensor product subspace with specified number of points.

    Example
    -------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> subspace = create_test_tensor_product_subspace(joint, [3, 4], bkd)
    >>> subspace.nsamples()  # 3 * 4 = 12
    12
    """
    factories = create_basis_factories(joint.marginals(), bkd, basis_type)

    # With LinearGrowthRule(1, 1): npts = level + 1, so level = npts - 1
    growth = LinearGrowthRule(scale=1, shift=1)
    levels = [n - 1 for n in npts_1d]
    index = bkd.asarray(levels, dtype=bkd.int64_dtype())

    return TensorProductSubspace(bkd, index, factories, growth)


def create_test_tensor_product_subspace_mixed(
    joint: IndependentJoint[Array],
    basis_types: List[str],
    npts_1d: List[int],
    bkd: Backend[Array],
) -> TensorProductSubspace[Array]:
    """Create tensor product subspace with mixed basis types per dimension.

    This allows testing tensor product interpolation with different basis
    types in different dimensions (e.g., Gauss in dim 1, Leja in dim 2).

    Parameters
    ----------
    joint : IndependentJoint[Array]
        Joint distribution defining the marginals.
    basis_types : List[str]
        Basis type for each dimension. One of "gauss", "leja",
        "piecewise_linear", "piecewise_quadratic", "piecewise_cubic".
    npts_1d : List[int]
        Number of points in each dimension.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    TensorProductSubspace[Array]
        Tensor product subspace with mixed basis types.

    Example
    -------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> subspace = create_test_tensor_product_subspace_mixed(
    ...     joint, ["gauss", "leja"], [3, 3], bkd
    ... )
    >>> subspace.nsamples()
    9
    """
    marginals = joint.marginals()
    factories: List[BasisFactoryProtocol[Array]] = []

    for marginal, btype in zip(marginals, basis_types):
        factory: BasisFactoryProtocol[Array]
        if btype == "gauss":
            factory = GaussLagrangeFactory(marginal, bkd)
        elif btype == "leja":
            factory = LejaLagrangeFactory(marginal, bkd)
        elif btype.startswith("piecewise_"):
            poly_type = btype.replace("piecewise_", "")
            factory = PiecewiseFactory(marginal, bkd, poly_type=poly_type)
        else:
            raise ValueError(f"Unknown basis_type: {btype}")
        factories.append(factory)

    growth = LinearGrowthRule(scale=1, shift=1)
    levels = [n - 1 for n in npts_1d]
    index = bkd.asarray(levels, dtype=bkd.int64_dtype())

    return TensorProductSubspace(bkd, index, factories, growth)


# =============================================================================
# Anisotropic and Additive PCE creation
# =============================================================================


def create_anisotropic_pce(
    joint: IndependentJoint[Array],
    max_levels_1d: List[int],
    total_degree: int | None,
    nqoi: int,
    bkd: Backend[Array],
    seed: int = 42,
) -> PolynomialChaosExpansion[Array]:
    """Create PCE with anisotropic index set (different max degree per dimension).

    Uses HyperbolicIndexGenerator with max_1d_levels parameter.
    If total_degree is specified, also applies total degree constraint.

    Parameters
    ----------
    joint : IndependentJoint[Array]
        Joint distribution defining the marginals.
    max_levels_1d : List[int]
        Maximum polynomial degree per dimension.
    total_degree : int | None
        Maximum total degree (sum of indices). If None, uses sum of max_levels_1d.
    nqoi : int
        Number of quantities of interest.
    bkd : Backend[Array]
        Computational backend.
    seed : int, optional
        Random seed for coefficient generation. Default: 42.

    Returns
    -------
    PolynomialChaosExpansion[Array]
        PCE with anisotropic index set and random coefficients.

    Example
    -------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> pce = create_anisotropic_pce(joint, [3, 1], None, nqoi=1, bkd=bkd)
    >>> pce.nterms()  # (3+1) * (1+1) = 8 tensor product indices
    8
    """
    nvars = joint.nvars()
    max_1d_arr = bkd.asarray(max_levels_1d, dtype=bkd.int64_dtype())

    # Use total_degree if specified, otherwise sum of max_levels_1d
    max_level = sum(max_levels_1d) if total_degree is None else total_degree

    gen = HyperbolicIndexGenerator(
        nvars=nvars,
        max_level=max_level,
        pnorm=1.0,
        bkd=bkd,
        max_1d_levels=max_1d_arr,
    )
    indices = gen.get_selected_indices()

    # Create bases and PCE
    bases_1d: List[PhysicalDomainBasis1DProtocol[Any]] = [
        get_basis_from_marginal(m, bkd) for m in joint.marginals()
    ]
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    pce = PolynomialChaosExpansion(basis, bkd, nqoi=nqoi)

    # Set random coefficients
    np.random.seed(seed)
    coeffs = bkd.asarray(np.random.randn(pce.nterms(), nqoi))  # type: ignore[arg-type]
    pce.set_coefficients(coeffs)
    return pce


def create_additive_indices(
    max_levels_1d: List[int],
    bkd: Backend[Array],
) -> Array:
    """Create indices for additive function: f(x) = g1(x1) + g2(x2) + ...

    Returns indices where only one dimension is non-zero at a time.

    Parameters
    ----------
    max_levels_1d : List[int]
        Maximum polynomial degree per dimension.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Multi-indices of shape (nvars, nterms).

    Example
    -------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> indices = create_additive_indices([3, 2], bkd)
    >>> indices.shape  # (2, 1 + 3 + 2) = (2, 6)
    (2, 6)
    """
    nvars = len(max_levels_1d)
    indices = [[0] * nvars]  # constant term
    for dim, max_level in enumerate(max_levels_1d):
        for level in range(1, max_level + 1):
            idx = [0] * nvars
            idx[dim] = level
            indices.append(idx)
    return bkd.asarray(indices, dtype=bkd.int64_dtype()).T  # Shape: (nvars, nterms)


def create_additive_pce(
    joint: IndependentJoint[Array],
    max_level_per_dim: List[int],
    nqoi: int,
    bkd: Backend[Array],
    seed: int = 42,
) -> PolynomialChaosExpansion[Array]:
    """Create additive PCE: f(x) = g1(x1) + g2(x2) + ... (only 1D terms).

    Parameters
    ----------
    joint : IndependentJoint[Array]
        Joint distribution defining the marginals.
    max_level_per_dim : List[int]
        Maximum polynomial degree per dimension.
    nqoi : int
        Number of quantities of interest.
    bkd : Backend[Array]
        Computational backend.
    seed : int, optional
        Random seed for coefficient generation. Default: 42.

    Returns
    -------
    PolynomialChaosExpansion[Array]
        PCE with additive index set and random coefficients.

    Example
    -------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> joint = create_test_joint("2d_uniform", bkd)
    >>> pce = create_additive_pce(joint, [3, 2], nqoi=1, bkd=bkd)
    >>> pce.nterms()  # 1 + 3 + 2 = 6
    6
    """
    indices = create_additive_indices(max_level_per_dim, bkd)

    bases_1d: List[PhysicalDomainBasis1DProtocol[Any]] = [
        get_basis_from_marginal(m, bkd) for m in joint.marginals()
    ]
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    pce = PolynomialChaosExpansion(basis, bkd, nqoi=nqoi)

    np.random.seed(seed)
    coeffs = bkd.asarray(np.random.randn(pce.nterms(), nqoi))  # type: ignore[arg-type]
    pce.set_coefficients(coeffs)
    return pce


def get_required_sg_levels(
    pce_max_levels: List[int],
    growth: IndexGrowthRule,
) -> List[int]:
    """Compute SG levels needed to exactly represent PCE given growth rule.

    For each dimension d, find minimum SG level l such that
    growth(l) > pce_max_levels[d] (number of points covers degree + 1).

    Note: This only computes per-dimension max levels. For PCEs with cross-terms,
    use `compute_required_sg_subspaces` to get the full required index set.

    Parameters
    ----------
    pce_max_levels : List[int]
        Maximum polynomial degree per dimension in the PCE.
    growth : IndexGrowthRule
        Growth rule mapping SG level to number of points.

    Returns
    -------
    List[int]
        Required SG level per dimension.

    Example
    -------
    >>> from pyapprox.surrogates.affine.indices import LinearGrowthRule
    >>> growth = LinearGrowthRule(scale=1, shift=1)  # npts = level + 1
    >>> get_required_sg_levels([3, 1], growth)  # Need levels [3, 1]
    [3, 1]
    """
    from pyapprox.surrogates.affine.indices import inverse_growth_rule

    return [inverse_growth_rule(max_deg, growth) for max_deg in pce_max_levels]


def compute_required_sg_subspaces(
    pce_indices: Array,
    growth: IndexGrowthRule,
    bkd: Backend[Array],
) -> Array:
    """Compute sparse grid subspace indices required to represent a PCE.

    For each PCE multi-index specifying polynomial degrees, computes the minimum
    sparse grid level in each dimension needed to represent that degree, then
    takes the downward closure to get a valid sparse grid index set.

    The relationship between PCE degree d and sparse grid level l is:
        l = min{k : growth(k) > d}

    Since Lagrange interpolation with n points can exactly represent polynomials
    of degree n-1, we need growth(l) > d to have enough points.

    Parameters
    ----------
    pce_indices : Array
        PCE multi-indices specifying polynomial degrees. Shape: (nvars, nterms)
    growth : IndexGrowthRule
        Growth rule mapping sparse grid level to number of points.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Sparse grid subspace indices. Shape: (nvars, nsubspaces)
        The result is guaranteed to be downward-closed and sorted.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.indices import LinearGrowthRule
    >>> bkd = NumpyBkd()
    >>> growth = LinearGrowthRule(scale=1, shift=1)  # n(l) = l + 1
    >>> # PCE with terms up to degree (3, 1) including cross-term (3, 1)
    >>> pce_indices = bkd.asarray([[3, 0], [1, 1]], dtype=bkd.int64_dtype())
    >>> sg_indices = compute_required_sg_subspaces(pce_indices, growth, bkd)
    >>> # The term (3, 1) requires SG level (3, 1), and downward closure gives
    >>> # all indices (i, j) with i <= 3 and j <= 1
    """
    from pyapprox.surrogates.affine.indices import (
        inverse_growth_rule,
        compute_downward_closure,
    )

    nvars = pce_indices.shape[0]
    nterms = pce_indices.shape[1]

    if nterms == 0:
        # Empty PCE: return just the zero index
        return bkd.zeros((nvars, 1), dtype=bkd.int64_dtype())

    # Compute minimum SG level for each PCE term
    sg_levels = bkd.zeros((nvars, nterms), dtype=bkd.int64_dtype())

    for j in range(nterms):
        for i in range(nvars):
            degree = int(bkd.to_numpy(pce_indices[i, j]))
            sg_levels[i, j] = inverse_growth_rule(degree, growth)

    # Compute downward closure
    return compute_downward_closure(sg_levels, bkd)


# =============================================================================
# Basis type configurations for extensibility
# =============================================================================

# (name, basis_type, default_growth_rule_name)
# Extensible: add "spline", "wavelet" entries when implemented
BASIS_TYPE_CONFIGS: List[Tuple[str, str, str]] = [
    ("gauss", "gauss", "linear_1_1"),
    ("leja", "leja", "linear_1_1"),
    ("clenshaw_curtis", "clenshaw_curtis", "clenshaw_curtis"),
    ("piecewise_linear", "piecewise_linear", "clenshaw_curtis"),
    ("piecewise_quadratic", "piecewise_quadratic", "clenshaw_curtis"),
    ("piecewise_cubic", "piecewise_cubic", "cubic_nested"),  # special growth rule for cubic
]

# Bounded-only basis types (require bounded domains)
BOUNDED_BASIS_TYPES: List[str] = [
    "piecewise_linear",
    "piecewise_quadratic",
    "piecewise_cubic",
]




# =============================================================================
# Smooth test function for piecewise interpolation
# =============================================================================


def _get_marginal_bounds(marginal: Any) -> Tuple[float, float]:
    """Get bounds from a marginal distribution."""
    if hasattr(marginal, "bounds"):
        bounds = marginal.bounds()
        return float(bounds[0]), float(bounds[1])
    elif hasattr(marginal, "_lb") and hasattr(marginal, "_ub"):
        return float(marginal._lb), float(marginal._ub)
    else:
        # Default for unbounded (should not happen for piecewise tests)
        return -1.0, 1.0


def create_smooth_test_function(
    joint: IndependentJoint[Array],
    bkd: Backend[Array],
) -> Callable[[Array], Array]:
    """Create smooth test function for piecewise interpolation tests.

    Returns f(x) = prod_i cos(pi * (x_i - a_i) / (b_i - a_i))
    where [a_i, b_i] are the bounds of marginal i.

    This function is smooth and has value 1 at domain center, oscillating
    toward the boundaries. It provides a good test for interpolation.

    Parameters
    ----------
    joint : IndependentJoint[Array]
        Joint distribution defining the domain bounds.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Callable[[Array], Array]
        Test function f: (nvars, nsamples) -> (1, nsamples)
    """
    import math

    marginals = joint.marginals()
    bounds_list = [_get_marginal_bounds(m) for m in marginals]

    def test_func(samples: Array) -> Array:
        # samples shape: (nvars, nsamples)
        result = bkd.ones((1, samples.shape[1]))
        for i, (a, b) in enumerate(bounds_list):
            # Normalize to [0, 1]
            normalized = (samples[i, :] - a) / (b - a)
            # Cosine oscillation
            result = result * bkd.cos(math.pi * normalized)
        return result

    return test_func


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "MARGINAL_FACTORIES",
    "JOINT_CONFIGS",
    "BOUNDED_JOINT_CONFIGS",
    "GROWTH_RULES",
    "_get_default_growth_rule",
    "BASIS_TYPE_CONFIGS",
    "BOUNDED_BASIS_TYPES",
    "create_test_joint",
    "create_test_pce",
    "create_tensor_product_pce",
    "create_test_tensor_product_subspace",
    "create_test_tensor_product_subspace_mixed",
    "create_anisotropic_pce",
    "create_additive_indices",
    "create_additive_pce",
    "get_required_sg_levels",
    "compute_required_sg_subspaces",
    "create_smooth_test_function",
]
