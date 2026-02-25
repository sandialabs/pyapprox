"""Basis factory system for sparse grids.

This module provides factories that create univariate bases from marginal
distributions. Factories handle domain transforms internally, returning
bases that operate in user domain (not canonical domain).

Key classes:
- BasisFactoryProtocol: Protocol for basis factories
- GaussLagrangeFactory: Creates Lagrange basis with Gauss quadrature
- LejaLagrangeFactory: Creates Lagrange basis with Leja quadrature (cached)
- ClenshawCurtisLagrangeFactory: Creates Lagrange basis with CC quadrature (nested)
- PiecewiseFactory: Creates piecewise polynomial basis (placeholder)
- PrebuiltBasisFactory: Wraps existing basis for migration

Key functions:
- create_basis_factories: Create factories from marginals
- create_bases_from_marginals: Create pre-built bases for tensor products
- get_transform_from_marginal: Get appropriate domain transform
- get_bounds_from_marginal: Get integration bounds from marginal
- register_basis_factory: Register custom factory types
- get_registered_basis_types: List available factory types

Design notes:
- Lagrange interpolation is numerically stable in user domain for affine
  transforms because ratios cancel in the Lagrange formula
- Quadrature weights are NOT adjusted when transforming domains (matches
  legacy behavior) - orthonormal polynomials are constructed for the
  probability measure, so weights integrate correctly as-is
- The factory system uses a registry pattern for extensibility
"""

from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.affine.protocols import (
    Basis1DProtocol,
    InterpolationBasis1DProtocol,
    OrthonormalPolynomial1DProtocol,
)
from pyapprox.surrogates.affine.univariate.lagrange import LagrangeBasis1D
from pyapprox.surrogates.affine.univariate.transforms import (
    Univariate1DTransformProtocol,
    get_transform_from_marginal,
)
from pyapprox.surrogates.affine.univariate.registry import (
    _lookup_analytical,
)
from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
    ClenshawCurtisQuadratureRule,
)
from pyapprox.surrogates.affine.leja.protocols import (
    LejaSequence1DProtocol,
    LejaWeightingProtocol,
)
from pyapprox.surrogates.affine.leja.univariate import LejaSequence1D
from pyapprox.surrogates.affine.leja.weighting import (
    ChristoffelWeighting,
    PDFWeighting,
)
from pyapprox.probability.protocols import MarginalProtocol


# Type alias for factory creators
# Using Any for return type since factories are generic over Array type
FactoryCreator = Callable[..., Any]

# Module-level registry for basis factory creators
# Maps basis_type name -> callable(marginal, bkd, **kwargs) -> BasisFactoryProtocol
_BASIS_FACTORY_REGISTRY: Dict[str, FactoryCreator] = {}


def register_basis_factory(name: str, factory_creator: FactoryCreator) -> None:
    """Register a basis factory creator by name.

    This allows extending the basis factory system without modifying
    existing code. New factory types can be registered at module load
    or dynamically at runtime.

    Parameters
    ----------
    name : str
        Name to use when calling create_basis_factories(basis_type=name).
    factory_creator : Callable
        Callable that takes (marginal, bkd, **kwargs) and returns a
        BasisFactoryProtocol instance.

    Examples
    --------
    >>> def my_factory_creator(marginal, bkd, **kwargs):
    ...     return MyCustomFactory(marginal, bkd)
    >>> register_basis_factory("my_custom", my_factory_creator)
    >>> # Now can use: create_basis_factories(marginals, bkd, "my_custom")
    """
    _BASIS_FACTORY_REGISTRY[name] = factory_creator


def get_registered_basis_types() -> List[str]:
    """Return list of registered basis factory type names.

    Returns
    -------
    List[str]
        Sorted list of registered basis type names.

    Examples
    --------
    >>> types = get_registered_basis_types()
    >>> "gauss" in types
    True
    >>> "leja" in types
    True
    """
    return sorted(_BASIS_FACTORY_REGISTRY.keys())


@runtime_checkable
class BasisFactoryProtocol(Protocol, Generic[Array]):
    """Protocol for factories that create univariate interpolation bases.

    Factories are used by sparse grids to create fresh basis instances
    for each subspace. This allows different subspaces to have different
    numbers of terms without sharing state.

    The factory handles domain transforms internally, so the basis
    operates in the user's domain (e.g., Uniform[0,1] returns samples
    in [0,1], not the canonical [-1,1]).

    The returned basis must satisfy InterpolationBasis1DProtocol, which
    requires: bkd(), set_nterms(), nterms(), __call__(), get_samples().
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def create_basis(self) -> InterpolationBasis1DProtocol[Array]:
        """Create a new interpolation basis instance.

        Returns a fresh basis that satisfies InterpolationBasis1DProtocol.
        For sparse grids, this is called once per dimension per subspace.
        """
        ...

    def is_nested(self) -> bool:
        """Return True if quadrature nodes are nested across levels.

        Nested quadrature rules (e.g., Clenshaw-Curtis, Leja) have the property
        that points at level l are a subset of points at level l+1. This allows
        for more efficient sample deduplication using basis indices rather than
        floating-point coordinate hashing.

        Returns
        -------
        bool
            True if the quadrature rule is nested, False otherwise.
        """
        ...


class GaussLagrangeFactory(Generic[Array]):
    """Factory that creates Lagrange basis with Gauss quadrature from marginal.

    Uses :func:`~pyapprox.surrogates.quadrature.gauss_quadrature_rule`
    to compute distribution-specific Gauss quadrature points in the
    marginal's physical domain with probability-measure weights.

    Parameters
    ----------
    marginal : MarginalProtocol
        Univariate marginal distribution.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
    >>> factory = GaussLagrangeFactory(marginal, bkd)
    >>> basis = factory.create_basis()
    >>> basis.set_nterms(5)
    >>> samples, weights = basis.quadrature_rule()
    >>> # samples are in [0, 1], not [-1, 1]
    """

    def __init__(self, marginal: MarginalProtocol[Array], bkd: Backend[Array]) -> None:
        self._marginal = marginal
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def create_basis(self) -> LagrangeBasis1D[Array]:
        """Create a Lagrange basis with user-domain Gauss quadrature.

        Returns
        -------
        LagrangeBasis1D[Array]
            Lagrange basis with quadrature points in user domain.
        """
        from pyapprox.surrogates.quadrature import (
            gauss_quadrature_rule,
        )

        marginal = self._marginal
        bkd = self._bkd

        def user_domain_quad_rule(npoints: int) -> Tuple[Array, Array]:
            """Quadrature rule that returns user-domain points."""
            return gauss_quadrature_rule(marginal, npoints, bkd)

        return LagrangeBasis1D(bkd, user_domain_quad_rule)

    def is_nested(self) -> bool:
        """Return False - Gauss quadrature points are not nested."""
        return False

    def __repr__(self) -> str:
        return f"GaussLagrangeFactory(marginal={self._marginal!r})"


class LejaLagrangeFactory(Generic[Array]):
    """Factory that creates Lagrange basis with Leja quadrature from marginal.

    Uses a cached Leja sequence so that points are nested and reused
    across different basis instances and subspace levels.

    Parameters
    ----------
    marginal : MarginalProtocol
        Univariate marginal distribution.
    bkd : Backend[Array]
        Computational backend.
    weighting : str, optional
        Weighting strategy for Leja optimization. One of "christoffel"
        (default) or "pdf".
    eps : float, optional
        Probability mass for bounds of unbounded distributions.
        Default: 1e-6 (captures 1-eps probability mass).

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
    >>> factory = LejaLagrangeFactory(marginal, bkd)
    >>> basis = factory.create_basis()
    >>> basis.set_nterms(5)
    >>> samples, weights = basis.quadrature_rule()
    """

    def __init__(
        self,
        marginal: MarginalProtocol[Array],
        bkd: Backend[Array],
        weighting: str = "christoffel",
        eps: float = 1e-6,
    ) -> None:
        self._marginal = marginal
        self._bkd = bkd
        self._weighting_type = weighting
        self._eps = eps
        self._leja_seq: Optional[LejaSequence1DProtocol[Array]] = None
        self._transform: Optional[Univariate1DTransformProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def _get_or_create_leja_sequence(self) -> LejaSequence1DProtocol[Array]:
        """Get or create the cached Leja sequence."""
        if self._leja_seq is not None:
            return self._leja_seq

        # Get polynomial using registry
        entry = _lookup_analytical(self._marginal)
        if entry is not None:
            poly = entry.polynomial_factory(self._marginal, self._bkd)
        else:
            # Fallback for custom marginals
            from pyapprox.surrogates.affine.univariate.globalpoly.continuous_numeric import (
                ContinuousNumericOrthonormalPolynomial1D,
            )
            poly = ContinuousNumericOrthonormalPolynomial1D(
                self._marginal, self._bkd
            )

        bounds = get_bounds_from_marginal(self._marginal, self._eps)
        self._transform = get_transform_from_marginal(self._marginal, self._bkd)

        # Create weighting
        weighting: LejaWeightingProtocol[Array]
        if self._weighting_type == "christoffel":
            weighting = ChristoffelWeighting(self._bkd)
        elif self._weighting_type == "pdf":
            # For PDF weighting we need the marginal's PDF
            weighting = PDFWeighting(self._bkd, self._marginal.pdf)
        else:
            raise ValueError(f"Unknown weighting: {self._weighting_type}")

        # Create Leja sequence in canonical domain
        self._leja_seq = LejaSequence1D(
            self._bkd, poly, weighting, bounds=bounds
        )

        return self._leja_seq

    def create_basis(self) -> LagrangeBasis1D[Array]:
        """Create a Lagrange basis with user-domain Leja quadrature.

        The Leja sequence is cached, so subsequent calls reuse
        the same sequence (nested points).

        Returns
        -------
        LagrangeBasis1D[Array]
            Lagrange basis with Leja quadrature points in user domain.
        """
        leja_seq = self._get_or_create_leja_sequence()
        transform = self._transform
        if transform is None:
            raise RuntimeError(
                "_get_or_create_leja_sequence() failed to initialize transform"
            )
        bkd = self._bkd

        def user_domain_quad_rule(npoints: int) -> Tuple[Array, Array]:
            """Quadrature rule that returns user-domain Leja points."""
            # Get Leja points/weights (extends sequence if needed)
            canonical_pts, weights = leja_seq.quadrature_rule(npoints)

            # Transform to user domain (weights unchanged per legacy behavior)
            user_pts = transform.map_from_canonical(canonical_pts)

            return user_pts, weights

        return LagrangeBasis1D(bkd, user_domain_quad_rule)

    def is_nested(self) -> bool:
        """Return True - Leja sequences are nested by construction."""
        return True

    def __repr__(self) -> str:
        return (
            f"LejaLagrangeFactory(marginal={self._marginal!r}, "
            f"weighting={self._weighting_type!r})"
        )


class ClenshawCurtisLagrangeFactory(Generic[Array]):
    """Factory that creates Lagrange basis with Clenshaw-Curtis quadrature.

    Clenshaw-Curtis points are nested: points at level l are a subset of
    points at level l+1. This makes them ideal for adaptive sparse grids
    where the hierarchical surplus should be stable.

    Important: Requires a compatible growth rule for proper nesting.
    Use ClenshawCurtisGrowthRule which gives n = 2^l + 1 points at level l
    (with level 0 giving 1 point).

    Parameters
    ----------
    marginal : MarginalProtocol
        Univariate marginal distribution.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
    >>> factory = ClenshawCurtisLagrangeFactory(marginal, bkd)
    >>> basis = factory.create_basis()
    >>> basis.set_nterms(5)  # Must be 1 or 2^l + 1
    >>> samples, weights = basis.quadrature_rule()
    >>> # samples are in [0, 1], not [-1, 1]
    """

    def __init__(self, marginal: MarginalProtocol[Array], bkd: Backend[Array]) -> None:
        self._marginal = marginal
        self._bkd = bkd
        self._cc_rule: Optional[ClenshawCurtisQuadratureRule[Array]] = None
        self._transform: Optional[Univariate1DTransformProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def _setup(self) -> None:
        """Initialize Clenshaw-Curtis quadrature and transform (lazy)."""
        if self._cc_rule is None:
            # Create CC rule with caching enabled and probability measure
            self._cc_rule = ClenshawCurtisQuadratureRule(
                self._bkd, store=True, prob_measure=True
            )
            self._transform = get_transform_from_marginal(
                self._marginal, self._bkd
            )

    def create_basis(self) -> LagrangeBasis1D[Array]:
        """Create a Lagrange basis with user-domain Clenshaw-Curtis quadrature.

        Returns
        -------
        LagrangeBasis1D[Array]
            Lagrange basis with CC quadrature points in user domain.
        """
        self._setup()

        # Capture references for closure (raise if _setup failed)
        cc_rule = self._cc_rule
        transform = self._transform
        if cc_rule is None:
            raise RuntimeError("_setup() failed to initialize CC rule")
        if transform is None:
            raise RuntimeError("_setup() failed to initialize transform")
        bkd = self._bkd

        def user_domain_quad_rule(npoints: int) -> Tuple[Array, Array]:
            """Quadrature rule that returns user-domain CC points."""
            # Get canonical domain CC quadrature
            canonical_pts, weights = cc_rule(npoints)

            # Transform to user domain (weights unchanged per legacy behavior)
            user_pts = transform.map_from_canonical(canonical_pts)

            return user_pts, weights

        return LagrangeBasis1D(bkd, user_domain_quad_rule)

    def is_nested(self) -> bool:
        """Return True - Clenshaw-Curtis points are nested."""
        return True

    def __repr__(self) -> str:
        return f"ClenshawCurtisLagrangeFactory(marginal={self._marginal!r})"


class PiecewiseFactory(Generic[Array]):
    """Factory that creates piecewise polynomial basis from marginal.

    This is a placeholder for future implementation of piecewise
    polynomial bases (linear, quadratic, cubic) for local adaptivity.

    Parameters
    ----------
    marginal : MarginalProtocol
        Univariate marginal distribution.
    bkd : Backend[Array]
        Computational backend.
    poly_type : str, optional
        Type of piecewise polynomial. One of "linear", "quadratic",
        "cubic". Default: "quadratic".
    eps : float, optional
        Probability mass for bounds of unbounded distributions.
        Default: 1e-6.
    """

    def __init__(
        self,
        marginal: MarginalProtocol[Array],
        bkd: Backend[Array],
        poly_type: str = "quadratic",
        eps: float = 1e-6,
    ) -> None:
        self._marginal = marginal
        self._bkd = bkd
        self._poly_type = poly_type
        self._eps = eps

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def create_basis(self) -> InterpolationBasis1DProtocol[Array]:
        """Create a piecewise polynomial basis.

        Returns
        -------
        InterpolationBasis1DProtocol[Array]
            Piecewise polynomial basis with dynamic node count support.
            Satisfies InterpolationBasis1DProtocol.

        Raises
        ------
        ValueError
            If poly_type is not one of "linear", "quadratic", "cubic".
        """
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            DynamicPiecewiseBasis,
            EquidistantNodeGenerator,
            PiecewiseLinear,
            PiecewiseQuadratic,
            PiecewiseCubic,
        )

        bounds = get_bounds_from_marginal(self._marginal, self._eps)
        node_gen = EquidistantNodeGenerator(self._bkd, bounds)

        basis_classes = {
            "linear": PiecewiseLinear,
            "quadratic": PiecewiseQuadratic,
            "cubic": PiecewiseCubic,
        }
        if self._poly_type not in basis_classes:
            raise ValueError(
                f"Unknown poly_type: {self._poly_type}. "
                f"Expected one of: {list(basis_classes.keys())}"
            )

        return DynamicPiecewiseBasis(
            self._bkd, basis_classes[self._poly_type], node_gen
        )

    def is_nested(self) -> bool:
        """Return False - piecewise polynomial nodes are not necessarily nested."""
        return False

    def __repr__(self) -> str:
        return (
            f"PiecewiseFactory(marginal={self._marginal!r}, "
            f"poly_type={self._poly_type!r})"
        )


class PrebuiltBasisFactory(Generic[Array]):
    """Factory that wraps an existing basis for migration.

    This factory allows existing code that uses pre-built bases to
    work with the new factory-based API. It extracts the quadrature rule
    from the prebuilt basis at construction time, then creates a fresh
    LagrangeBasis1D instance on each create_basis() call.

    This ensures independent state for each subspace in a sparse grid,
    matching the behavior of other factories (GaussLagrangeFactory, etc.).

    Parameters
    ----------
    basis : Basis1DProtocol[Array]
        Pre-built basis to wrap. Must have a quadrature rule method
        (gauss_quadrature_rule or quadrature_rule).

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.surrogates.affine.univariate import LagrangeBasis1D
    >>> from pyapprox.surrogates.affine.univariate.globalpoly import (
    ...     ClenshawCurtisQuadratureRule,
    ... )
    >>> bkd = NumpyBkd()
    >>> cc_quad = ClenshawCurtisQuadratureRule(bkd, store=True)
    >>> basis = LagrangeBasis1D(bkd, cc_quad)
    >>> factory = PrebuiltBasisFactory(basis)
    >>> # Each create_basis() returns independent instance:
    >>> b1 = factory.create_basis()
    >>> b2 = factory.create_basis()
    >>> b1.set_nterms(3)
    >>> b2.set_nterms(5)
    >>> assert b1.nterms() == 3 and b2.nterms() == 5
    """

    def __init__(self, basis: Basis1DProtocol[Array]) -> None:
        from pyapprox.surrogates.sparsegrids.basis_setup import (
            get_quadrature_rule,
        )

        self._bkd = basis.bkd()
        self._quad_rule = get_quadrature_rule(basis)  # Extract once at construction

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def create_basis(self) -> InterpolationBasis1DProtocol[Array]:
        """Create a fresh LagrangeBasis1D with the wrapped basis's quadrature rule.

        Returns a new LagrangeBasis1D instance each time, ensuring independent
        state for each subspace in a sparse grid.

        Returns
        -------
        InterpolationBasis1DProtocol[Array]
            Fresh Lagrange basis using the prebuilt basis's quadrature rule.
        """
        return LagrangeBasis1D(self._bkd, self._quad_rule)

    def is_nested(self) -> bool:
        """Return False - cannot determine nestedness for prebuilt basis.

        Conservative default: assume non-nested unless the wrapped basis
        explicitly indicates otherwise.
        """
        return False

    def __repr__(self) -> str:
        return f"PrebuiltBasisFactory(quad_rule={self._quad_rule!r})"


def get_bounds_from_marginal(
    marginal: MarginalProtocol[Array], eps: float = 1e-6
) -> Tuple[float, float]:
    """Get integration bounds from marginal.

    For bounded distributions, returns the full support.
    For unbounded distributions, returns the interval capturing
    (1-eps)*100% of the probability mass.

    Parameters
    ----------
    marginal : MarginalProtocol
        Univariate marginal distribution.
    eps : float, optional
        For unbounded distributions, the probability mass outside
        the returned interval. Default: 1e-6.

    Returns
    -------
    Tuple[float, float]
        Lower and upper bounds.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
    >>> get_bounds_from_marginal(marginal)
    (0.0, 1.0)
    """
    if marginal.is_bounded():
        bounds = marginal.interval(1.0)
    else:
        bounds = marginal.interval(1.0 - eps)

    return float(bounds[0, 0]), float(bounds[0, 1])


def create_basis_factories(
    marginals: List[MarginalProtocol[Array]],
    bkd: Backend[Array],
    basis_type: str = "gauss",
    **kwargs: Any,
) -> List[BasisFactoryProtocol[Array]]:
    """Create list of basis factories from marginals.

    Dimensions with identical marginal configurations share the same
    factory instance. For Leja factories, this means they share the
    cached Leja sequence, avoiding redundant computation.

    Parameters
    ----------
    marginals : List[MarginalProtocol]
        Univariate marginal distributions for each dimension.
    bkd : Backend[Array]
        Computational backend.
    basis_type : str, optional
        Type of basis factory to create. One of:
        - "gauss": GaussLagrangeFactory (default)
        - "leja": LejaLagrangeFactory
        - "clenshaw_curtis": ClenshawCurtisFactory
        - "piecewise_linear": PiecewiseFactory with linear polynomials
        - "piecewise_quadratic": PiecewiseFactory with quadratic polynomials
        - "piecewise_cubic": PiecewiseFactory with cubic polynomials
    **kwargs
        Additional arguments passed to factory constructors.
        For "leja": weighting (str), eps (float)
        For "piecewise_*": eps (float)

    Returns
    -------
    List[BasisFactoryProtocol[Array]]
        List of basis factories, one per marginal.
        Identical marginals share the same factory instance.

    Notes
    -----
    **Growth Rule Compatibility**

    When using these factories with sparse grids, certain basis types have
    constraints on the growth rule:

    - gauss, leja, piecewise_linear: Any growth rule works
    - clenshaw_curtis: Any growth rule works (ClenshawCurtisGrowthRule for nested)
    - piecewise_quadratic: Requires odd number of nodes.
      Use ClenshawCurtisGrowthRule() which produces 1, 3, 5, 9, 17, ...
    - piecewise_cubic: Requires (n - 4) % 3 == 0.
      Use CubicNestedGrowthRule() which produces 1, 4, 7, 13, 25, ...

    The sparse grid constructor validates growth rule compatibility and raises
    a ValueError with guidance if an incompatible combination is detected.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
    >>> factories = create_basis_factories(marginals, bkd, basis_type="gauss")
    >>> len(factories)
    2
    >>> factories[0] is factories[1]  # Same marginal, same factory
    True
    """
    if basis_type not in _BASIS_FACTORY_REGISTRY:
        raise ValueError(
            f"Unknown basis_type: {basis_type}. "
            f"Available: {get_registered_basis_types()}"
        )

    factory_creator = _BASIS_FACTORY_REGISTRY[basis_type]

    # Track seen marginals and their factories
    # Use list since marginals aren't hashable
    seen_marginals: List[MarginalProtocol[Array]] = []
    seen_factories: List[BasisFactoryProtocol[Array]] = []
    factories: List[BasisFactoryProtocol[Array]] = []

    for marginal in marginals:
        # Check if we've seen an equal marginal
        found_idx = None
        for idx, seen in enumerate(seen_marginals):
            if marginal == seen:
                found_idx = idx
                break

        if found_idx is not None:
            # Reuse existing factory
            factories.append(seen_factories[found_idx])
        else:
            # Create new factory
            factory = factory_creator(marginal, bkd, **kwargs)
            seen_marginals.append(marginal)
            seen_factories.append(factory)
            factories.append(factory)

    return factories


def create_bases_from_marginals(
    marginals: List[MarginalProtocol[Array]],
    bkd: Backend[Array],
    basis_type: str = "gauss",
    **kwargs: Any,
) -> List[Basis1DProtocol[Array]]:
    """Create list of pre-built bases from marginals.

    This is a convenience function for tensor product grids where
    you want pre-built bases rather than factories.

    Parameters
    ----------
    marginals : List[MarginalProtocol]
        Univariate marginal distributions for each dimension.
    bkd : Backend[Array]
        Computational backend.
    basis_type : str, optional
        Type of basis to create. See create_basis_factories for options.
    **kwargs
        Additional arguments passed to factory constructors.

    Returns
    -------
    List[Basis1DProtocol[Array]]
        List of pre-built bases, one per marginal.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
    >>> bases = create_bases_from_marginals(marginals, bkd)
    >>> for basis in bases:
    ...     basis.set_nterms(5)
    """
    factories = create_basis_factories(marginals, bkd, basis_type, **kwargs)
    return [factory.create_basis() for factory in factories]


# ---------------------------------------------------------------------
# Built-in factory creators and registration
# ---------------------------------------------------------------------


def _create_gauss_factory(
    marginal: Any, bkd: Backend[Array], **kwargs: Any
) -> GaussLagrangeFactory[Array]:
    """Factory creator for Gauss-Lagrange basis."""
    return GaussLagrangeFactory(marginal, bkd)


def _create_leja_factory(
    marginal: Any, bkd: Backend[Array], **kwargs: Any
) -> LejaLagrangeFactory[Array]:
    """Factory creator for Leja-Lagrange basis."""
    return LejaLagrangeFactory(
        marginal,
        bkd,
        weighting=kwargs.get("weighting", "christoffel"),
        eps=kwargs.get("eps", 1e-6),
    )


def _create_clenshaw_curtis_factory(
    marginal: Any, bkd: Backend[Array], **kwargs: Any
) -> ClenshawCurtisLagrangeFactory[Array]:
    """Factory creator for Clenshaw-Curtis Lagrange basis."""
    return ClenshawCurtisLagrangeFactory(marginal, bkd)


def _create_piecewise_factory(
    marginal: Any, bkd: Backend[Array], poly_type: str, **kwargs: Any
) -> PiecewiseFactory[Array]:
    """Factory creator for piecewise polynomial basis."""
    return PiecewiseFactory(
        marginal,
        bkd,
        poly_type=poly_type,
        eps=kwargs.get("eps", 1e-6),
    )


# Register built-in factories at module load
register_basis_factory("gauss", _create_gauss_factory)
register_basis_factory("leja", _create_leja_factory)
register_basis_factory("clenshaw_curtis", _create_clenshaw_curtis_factory)
register_basis_factory(
    "piecewise_linear", partial(_create_piecewise_factory, poly_type="linear")
)
register_basis_factory(
    "piecewise_quadratic",
    partial(_create_piecewise_factory, poly_type="quadratic"),
)
register_basis_factory(
    "piecewise_cubic", partial(_create_piecewise_factory, poly_type="cubic")
)


__all__ = [
    "BasisFactoryProtocol",
    "GaussLagrangeFactory",
    "LejaLagrangeFactory",
    "ClenshawCurtisLagrangeFactory",
    "PiecewiseFactory",
    "PrebuiltBasisFactory",
    "get_bounds_from_marginal",
    "get_transform_from_marginal",  # Re-exported from transforms module
    "create_basis_factories",
    "create_bases_from_marginals",
    "register_basis_factory",
    "get_registered_basis_types",
]
