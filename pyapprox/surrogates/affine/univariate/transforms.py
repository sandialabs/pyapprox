"""Univariate domain transforms for basis functions.

This module provides transforms for mapping between user domains and
canonical (reference) domains used by polynomial bases:

- Legendre/Jacobi: Canonical domain is [-1, 1]
- Hermite: Canonical domain is (-∞, ∞) with standard normal weighting
- Laguerre: Canonical domain is [0, ∞) with exponential weighting

Transform Types
---------------
- IdentityTransform1D: No transformation (identity mapping)
- BoundedAffineTransform1D: Map [lb, ub] ↔ [-1, 1] (for Legendre/Jacobi)
- UnboundedAffineTransform1D: Map N(loc, scale²) ↔ N(0, 1) (for Hermite)

Usage with Sparse Grids
-----------------------
When building sparse grids on non-canonical domains, pass the transform
to the basis constructor (if supported) or apply transforms to samples
before/after basis evaluation.

Example::

    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.surrogates.affine.univariate.transforms import (
        BoundedAffineTransform1D,
    )

    bkd = NumpyBkd()
    transform = BoundedAffineTransform1D(bkd, lb=0.0, ub=1.0)

    # Map samples from [0, 1] to [-1, 1] for Legendre basis
    canonical = transform.map_to_canonical(samples)

    # Evaluate basis in canonical domain
    values = legendre_basis(canonical)

    # Map quadrature points from [-1, 1] back to [0, 1]
    quad_pts = transform.map_from_canonical(canonical_quad_pts)
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class Univariate1DTransformProtocol(Protocol, Generic[Array]):
    """Protocol for 1D domain transformations.

    All transforms provide mappings between user domain and canonical domain.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def map_to_canonical(self, samples: Array) -> Array:
        """Map from user domain to canonical domain.

        Parameters
        ----------
        samples : Array
            Samples in user domain. Shape: (1, nsamples) or (nsamples,)

        Returns
        -------
        Array
            Samples in canonical domain. Same shape as input.
        """
        ...

    def map_from_canonical(self, canonical: Array) -> Array:
        """Map from canonical domain to user domain.

        Parameters
        ----------
        canonical : Array
            Samples in canonical domain. Shape: (1, nsamples) or (nsamples,)

        Returns
        -------
        Array
            Samples in user domain. Same shape as input.
        """
        ...

    def jacobian_factor(self) -> float:
        """Return Jacobian d(canonical)/d(user).

        For affine transforms this is constant: 1/scale.
        Used for derivative chain rule calculations.
        """
        ...


class IdentityTransform1D(Generic[Array]):
    """Identity transformation (no-op).

    Used when the basis is already on its canonical domain.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> transform = IdentityTransform1D(bkd)
    >>> samples = bkd.asarray([[0.0, 0.5, 1.0]])
    >>> transform.map_to_canonical(samples)  # Returns same array
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def map_to_canonical(self, samples: Array) -> Array:
        """Return samples unchanged."""
        return samples

    def map_from_canonical(self, canonical: Array) -> Array:
        """Return samples unchanged."""
        return canonical

    def jacobian_factor(self) -> float:
        """Return Jacobian factor (1.0 for identity)."""
        return 1.0

    def __repr__(self) -> str:
        return "IdentityTransform1D()"


class BoundedAffineTransform1D(Generic[Array]):
    """Affine transformation from [lb, ub] to [-1, 1].

    Maps user domain [lb, ub] to canonical domain [-1, 1]:
        canonical = 2 * (user - lb) / (ub - lb) - 1

    Inverse:
        user = lb + (canonical + 1) * (ub - lb) / 2

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    lb : float
        Lower bound of user domain.
    ub : float
        Upper bound of user domain.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> transform = BoundedAffineTransform1D(bkd, lb=0.0, ub=1.0)
    >>> samples = bkd.asarray([[0.0, 0.5, 1.0]])
    >>> transform.map_to_canonical(samples)  # Returns [[-1, 0, 1]]
    """

    def __init__(self, bkd: Backend[Array], lb: float, ub: float) -> None:
        if lb >= ub:
            raise ValueError(f"lb must be < ub, got lb={lb}, ub={ub}")
        self._bkd = bkd
        self._lb = lb
        self._ub = ub
        # Precompute constants
        self._scale = (ub - lb) / 2.0  # half-width
        self._shift = (ub + lb) / 2.0  # midpoint
        self._inv_scale = 1.0 / self._scale

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def lb(self) -> float:
        """Return lower bound."""
        return self._lb

    def ub(self) -> float:
        """Return upper bound."""
        return self._ub

    def map_to_canonical(self, samples: Array) -> Array:
        """Map from [lb, ub] to [-1, 1].

        canonical = (samples - midpoint) / half_width
        """
        return (samples - self._shift) * self._inv_scale

    def map_from_canonical(self, canonical: Array) -> Array:
        """Map from [-1, 1] to [lb, ub].

        user = midpoint + half_width * canonical
        """
        return self._shift + self._scale * canonical

    def jacobian_factor(self) -> float:
        """Return d(canonical)/d(user) = 1 / half_width."""
        return self._inv_scale

    def __repr__(self) -> str:
        return f"BoundedAffineTransform1D(lb={self._lb}, ub={self._ub})"


class UnboundedAffineTransform1D(Generic[Array]):
    """Affine transformation for unbounded domains.

    Maps user domain with location/scale to canonical domain:
        canonical = (user - loc) / scale

    Inverse:
        user = loc + scale * canonical

    This is used for Hermite polynomials to map from N(loc, scale²)
    to the standard normal N(0, 1).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    loc : float
        Location (mean) of user domain.
    scale : float
        Scale (standard deviation) of user domain. Must be positive.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Transform for N(5, 4) = N(mean=5, std=2)
    >>> transform = UnboundedAffineTransform1D(bkd, loc=5.0, scale=2.0)
    >>> samples = bkd.asarray([[5.0, 7.0, 3.0]])  # mean, mean+std, mean-std
    >>> transform.map_to_canonical(samples)  # Returns [[0, 1, -1]]
    """

    def __init__(self, bkd: Backend[Array], loc: float, scale: float) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self._bkd = bkd
        self._loc = loc
        self._scale = scale
        self._inv_scale = 1.0 / scale

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def loc(self) -> float:
        """Return location parameter."""
        return self._loc

    def scale(self) -> float:
        """Return scale parameter."""
        return self._scale

    def map_to_canonical(self, samples: Array) -> Array:
        """Map from user domain to canonical domain.

        canonical = (samples - loc) / scale
        """
        return (samples - self._loc) * self._inv_scale

    def map_from_canonical(self, canonical: Array) -> Array:
        """Map from canonical domain to user domain.

        user = loc + scale * canonical
        """
        return self._loc + self._scale * canonical

    def jacobian_factor(self) -> float:
        """Return d(canonical)/d(user) = 1 / scale."""
        return self._inv_scale

    def __repr__(self) -> str:
        return f"UnboundedAffineTransform1D(loc={self._loc}, scale={self._scale})"


def get_transform_from_marginal(
    marginal, bkd: Backend[Array]
) -> Univariate1DTransformProtocol:
    """Get the appropriate transform for a marginal distribution.

    This function uses the marginal registry to determine the correct
    transform for a given marginal type. For unregistered marginals,
    falls back to bounded/unbounded transforms based on `is_bounded()`.

    Parameters
    ----------
    marginal : Any
        Marginal distribution instance (e.g., UniformMarginal, GaussianMarginal).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Univariate1DTransformProtocol
        Transform mapping between marginal's physical domain and canonical domain.

    Raises
    ------
    ValueError
        If no transform can be determined for the marginal.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability import UniformMarginal, GaussianMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = UniformMarginal(0.0, 1.0, bkd)
    >>> transform = get_transform_from_marginal(marginal, bkd)
    >>> # Returns BoundedAffineTransform1D(lb=0.0, ub=1.0)
    """
    # Import registry here to avoid circular imports
    from pyapprox.surrogates.affine.univariate.registry import (
        _lookup_analytical,
        _lookup_discrete,
    )

    # Check analytical registry first
    entry = _lookup_analytical(marginal)
    if entry is not None:
        return entry.transform_factory(marginal, bkd)

    # Discrete marginals don't use transforms (physical domain polynomials)
    entry_discrete = _lookup_discrete(marginal)
    if entry_discrete is not None:
        return IdentityTransform1D(bkd)

    # Fallback for custom marginals based on is_bounded()
    if hasattr(marginal, "is_bounded") and marginal.is_bounded():
        lb, ub = marginal.bounds()
        return BoundedAffineTransform1D(bkd, lb, ub)

    raise ValueError(
        f"Cannot determine transform for marginal type {type(marginal).__name__}. "
        "Register it with register_analytical_marginal() or ensure is_bounded() is defined."
    )


__all__ = [
    "Univariate1DTransformProtocol",
    "IdentityTransform1D",
    "BoundedAffineTransform1D",
    "UnboundedAffineTransform1D",
    "get_transform_from_marginal",
]
