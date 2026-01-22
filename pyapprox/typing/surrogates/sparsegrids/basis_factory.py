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

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import Basis1DProtocol
from pyapprox.typing.surrogates.affine.univariate.lagrange import LagrangeBasis1D
from pyapprox.typing.surrogates.affine.univariate.transforms import (
    IdentityTransform1D,
    Univariate1DTransformProtocol,
    get_transform_from_marginal,
)
from pyapprox.typing.surrogates.affine.univariate.registry import (
    _lookup_analytical,
)


# Type alias for factory creators: (marginal, bkd, **kwargs) -> BasisFactoryProtocol
# Using Any for marginal type since marginals have different types
FactoryCreator = Callable[..., "BasisFactoryProtocol[Array]"]

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
    """Protocol for factories that create univariate bases.

    Factories are used by sparse grids to create fresh basis instances
    for each subspace. This allows different subspaces to have different
    numbers of terms without sharing state.

    The factory handles domain transforms internally, so the basis
    operates in the user's domain (e.g., Uniform[0,1] returns samples
    in [0,1], not the canonical [-1,1]).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def create_basis(self) -> Basis1DProtocol[Array]:
        """Create a new basis instance.

        Returns a fresh basis that supports set_nterms() and quadrature_rule().
        For sparse grids, this is called once per dimension per subspace.
        """
        ...


class GaussLagrangeFactory(Generic[Array]):
    """Factory that creates Lagrange basis with Gauss quadrature from marginal.

    Uses the orthonormal polynomial for the marginal's measure to compute
    Gauss quadrature points, then transforms them to the user's domain.

    Parameters
    ----------
    marginal : MarginalProtocol
        Univariate marginal distribution.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
    >>> factory = GaussLagrangeFactory(marginal, bkd)
    >>> basis = factory.create_basis()
    >>> basis.set_nterms(5)
    >>> samples, weights = basis.quadrature_rule()
    >>> # samples are in [0, 1], not [-1, 1]
    """

    def __init__(self, marginal, bkd: Backend[Array]) -> None:
        self._marginal = marginal
        self._bkd = bkd
        self._poly = None
        self._transform: Optional[Univariate1DTransformProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def _setup(self) -> None:
        """Initialize polynomial and transform (lazy initialization)."""
        if self._poly is None:
            # Use registry to get polynomial for the marginal
            entry = _lookup_analytical(self._marginal)
            if entry is not None:
                self._poly = entry.polynomial_factory(self._marginal, self._bkd)
            else:
                # Fallback for custom marginals
                from pyapprox.typing.surrogates.affine.univariate.globalpoly.continuous_numeric import (
                    BoundedNumericOrthonormalPolynomial1D,
                    UnboundedNumericOrthonormalPolynomial1D,
                )
                if hasattr(self._marginal, "is_bounded") and self._marginal.is_bounded():
                    self._poly = BoundedNumericOrthonormalPolynomial1D(
                        self._bkd, self._marginal
                    )
                else:
                    self._poly = UnboundedNumericOrthonormalPolynomial1D(
                        self._bkd, self._marginal
                    )
            self._transform = get_transform_from_marginal(
                self._marginal, self._bkd
            )

    def create_basis(self) -> LagrangeBasis1D[Array]:
        """Create a Lagrange basis with user-domain Gauss quadrature.

        Returns
        -------
        LagrangeBasis1D[Array]
            Lagrange basis with quadrature points in user domain.
        """
        self._setup()

        # Capture references for closure
        poly = self._poly
        transform = self._transform
        bkd = self._bkd

        def user_domain_quad_rule(npoints: int) -> Tuple[Array, Array]:
            """Quadrature rule that returns user-domain points."""
            # Ensure polynomial has enough terms
            if poly.nterms() < npoints:
                poly.set_nterms(npoints)

            # Get canonical domain quadrature
            canonical_pts, weights = poly.gauss_quadrature_rule(npoints)

            # Transform to user domain (weights unchanged per legacy behavior)
            user_pts = transform.map_from_canonical(canonical_pts)

            return user_pts, weights

        return LagrangeBasis1D(bkd, user_domain_quad_rule)

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
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
    >>> factory = LejaLagrangeFactory(marginal, bkd)
    >>> basis = factory.create_basis()
    >>> basis.set_nterms(5)
    >>> samples, weights = basis.quadrature_rule()
    """

    def __init__(
        self,
        marginal,
        bkd: Backend[Array],
        weighting: str = "christoffel",
        eps: float = 1e-6,
    ) -> None:
        self._marginal = marginal
        self._bkd = bkd
        self._weighting = weighting
        self._eps = eps
        self._leja_seq = None  # Cached across create_basis() calls
        self._transform: Optional[Univariate1DTransformProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def _get_or_create_leja_sequence(self):
        """Get or create the cached Leja sequence."""
        if self._leja_seq is not None:
            return self._leja_seq

        from pyapprox.typing.surrogates.affine.leja.univariate import LejaSequence1D
        from pyapprox.typing.surrogates.affine.leja.weighting import (
            ChristoffelWeighting,
            PDFWeighting,
        )

        # Get polynomial using registry
        entry = _lookup_analytical(self._marginal)
        if entry is not None:
            poly = entry.polynomial_factory(self._marginal, self._bkd)
        else:
            # Fallback for custom marginals
            from pyapprox.typing.surrogates.affine.univariate.globalpoly.continuous_numeric import (
                BoundedNumericOrthonormalPolynomial1D,
                UnboundedNumericOrthonormalPolynomial1D,
            )
            if hasattr(self._marginal, "is_bounded") and self._marginal.is_bounded():
                poly = BoundedNumericOrthonormalPolynomial1D(
                    self._bkd, self._marginal
                )
            else:
                poly = UnboundedNumericOrthonormalPolynomial1D(
                    self._bkd, self._marginal
                )

        bounds = get_bounds_from_marginal(self._marginal, self._eps)
        self._transform = get_transform_from_marginal(self._marginal, self._bkd)

        # Create weighting
        if self._weighting == "christoffel":
            weighting = ChristoffelWeighting(self._bkd)
        elif self._weighting == "pdf":
            # For PDF weighting we need the marginal's PDF
            weighting = PDFWeighting(self._marginal, self._bkd)
        else:
            raise ValueError(f"Unknown weighting: {self._weighting}")

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
        bkd = self._bkd

        def user_domain_quad_rule(npoints: int) -> Tuple[Array, Array]:
            """Quadrature rule that returns user-domain Leja points."""
            # Get Leja points/weights (extends sequence if needed)
            canonical_pts, weights = leja_seq.quadrature_rule(npoints)

            # Transform to user domain (weights unchanged per legacy behavior)
            user_pts = transform.map_from_canonical(canonical_pts)

            return user_pts, weights

        return LagrangeBasis1D(bkd, user_domain_quad_rule)

    def __repr__(self) -> str:
        return (
            f"LejaLagrangeFactory(marginal={self._marginal!r}, "
            f"weighting={self._weighting!r})"
        )


class ClenshawCurtisLagrangeFactory(Generic[Array]):
    """Factory that creates Lagrange basis with Clenshaw-Curtis quadrature.

    Clenshaw-Curtis points are nested: points at level l are a subset of
    points at level l+1. This makes them ideal for adaptive sparse grids
    where the hierarchical surplus should be stable.

    Important: Requires a compatible growth rule for proper nesting.
    Use DoublePlusOneGrowthRule which gives n = 2^l + 1 points at level l
    (with level 0 giving 1 point).

    Parameters
    ----------
    marginal : MarginalProtocol
        Univariate marginal distribution.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = UniformMarginal(lower=0.0, upper=1.0, bkd=bkd)
    >>> factory = ClenshawCurtisLagrangeFactory(marginal, bkd)
    >>> basis = factory.create_basis()
    >>> basis.set_nterms(5)  # Must be 1 or 2^l + 1
    >>> samples, weights = basis.quadrature_rule()
    >>> # samples are in [0, 1], not [-1, 1]
    """

    def __init__(self, marginal, bkd: Backend[Array]) -> None:
        self._marginal = marginal
        self._bkd = bkd
        self._cc_rule = None
        self._transform: Optional[Univariate1DTransformProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def _setup(self) -> None:
        """Initialize Clenshaw-Curtis quadrature and transform (lazy)."""
        if self._cc_rule is None:
            from pyapprox.typing.surrogates.affine.univariate.globalpoly.quadrature import (
                ClenshawCurtisQuadratureRule,
            )

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

        # Capture references for closure (assertions satisfy mypy after _setup)
        cc_rule = self._cc_rule
        transform = self._transform
        assert cc_rule is not None
        assert transform is not None
        bkd = self._bkd

        def user_domain_quad_rule(npoints: int) -> Tuple[Array, Array]:
            """Quadrature rule that returns user-domain CC points."""
            # Get canonical domain CC quadrature
            canonical_pts, weights = cc_rule(npoints)

            # Transform to user domain (weights unchanged per legacy behavior)
            user_pts = transform.map_from_canonical(canonical_pts)

            return user_pts, weights

        return LagrangeBasis1D(bkd, user_domain_quad_rule)

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
        marginal,
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

    def create_basis(self) -> Basis1DProtocol[Array]:
        """Create a piecewise polynomial basis.

        Returns
        -------
        Basis1DProtocol[Array]
            Piecewise polynomial basis with dynamic node count support.

        Raises
        ------
        ValueError
            If poly_type is not one of "linear", "quadratic", "cubic".
        """
        from pyapprox.typing.surrogates.affine.univariate.piecewisepoly import (
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

    def __repr__(self) -> str:
        return (
            f"PiecewiseFactory(marginal={self._marginal!r}, "
            f"poly_type={self._poly_type!r})"
        )


class PrebuiltBasisFactory(Generic[Array]):
    """Factory that wraps an existing basis for migration.

    This factory allows existing code that uses pre-built bases to
    work with the new factory-based API. It simply returns a copy
    of the wrapped basis on each create_basis() call.

    Note: For true sparse grid usage where each subspace needs
    independent state, the wrapped basis must support being reused
    (i.e., set_nterms should work correctly when called multiple times).

    Parameters
    ----------
    basis : Basis1DProtocol[Array]
        Pre-built basis to wrap.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
    >>> bkd = NumpyBkd()
    >>> basis = LegendrePolynomial1D(bkd)
    >>> factory = PrebuiltBasisFactory(basis)
    >>> # Use with sparse grid:
    >>> # grid = IsotropicCombinationSparseGrid(bkd, [factory, factory], growth, level)
    """

    def __init__(self, basis: Basis1DProtocol[Array]) -> None:
        self._basis = basis

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._basis.bkd()

    def create_basis(self) -> Basis1DProtocol[Array]:
        """Return the wrapped basis.

        Note: This returns the same basis instance each time.
        For sparse grids, the basis's set_nterms() will be called
        with different values for different subspaces.

        Returns
        -------
        Basis1DProtocol[Array]
            The wrapped basis.
        """
        return self._basis

    def __repr__(self) -> str:
        return f"PrebuiltBasisFactory(basis={self._basis!r})"


def get_bounds_from_marginal(marginal, eps: float = 1e-6) -> Tuple[float, float]:
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
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.univariate import UniformMarginal
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
    marginals: List,
    bkd: Backend[Array],
    basis_type: str = "gauss",
    **kwargs,
) -> List[BasisFactoryProtocol[Array]]:
    """Create list of basis factories from marginals.

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
        - "piecewise_linear", "piecewise_quadratic", "piecewise_cubic":
          PiecewiseFactory (not yet implemented)
    **kwargs
        Additional arguments passed to factory constructors.
        For "leja": weighting (str), eps (float)
        For "piecewise_*": eps (float)

    Returns
    -------
    List[BasisFactoryProtocol[Array]]
        List of basis factories, one per marginal.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.univariate import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(2)]
    >>> factories = create_basis_factories(marginals, bkd, basis_type="gauss")
    >>> len(factories)
    2
    """
    if basis_type not in _BASIS_FACTORY_REGISTRY:
        raise ValueError(
            f"Unknown basis_type: {basis_type}. "
            f"Available: {get_registered_basis_types()}"
        )

    factory_creator = _BASIS_FACTORY_REGISTRY[basis_type]
    return [factory_creator(marginal, bkd, **kwargs) for marginal in marginals]


def create_bases_from_marginals(
    marginals: List,
    bkd: Backend[Array],
    basis_type: str = "gauss",
    **kwargs,
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
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.univariate import UniformMarginal
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
