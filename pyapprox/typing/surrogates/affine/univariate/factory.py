"""Factory functions for creating physical-domain 1D bases.

This module provides the primary API for creating univariate bases that
accept samples in physical domain (user domain).

Functions
---------
create_basis_1d(marginal, bkd) -> PhysicalDomainBasis1DProtocol
    Create a physical-domain basis for a marginal distribution.

create_bases_1d(marginals, bkd) -> List[PhysicalDomainBasis1DProtocol]
    Create physical-domain bases for multiple marginals.

The factory uses the marginal registry to determine:
1. Which polynomial to use (analytical or numeric)
2. What transform to apply (if canonical-domain polynomial)
3. Whether to wrap with TransformedBasis1D or NativeBasis1D

Example
-------
>>> from pyapprox.typing.util.backends.numpy import NumpyBkd
>>> from pyapprox.typing.probability import UniformMarginal, BetaMarginal
>>> bkd = NumpyBkd()
>>> # Uniform marginal on [0, 2]
>>> marginal = UniformMarginal(0.0, 2.0, bkd)
>>> basis = create_basis_1d(marginal, bkd)
>>> basis.set_nterms(5)
>>> samples = bkd.asarray([[0.0, 1.0, 2.0]])  # Physical domain
>>> values = basis(samples)  # Shape: (3, 5)
"""

from typing import Any, List, Union

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.univariate.transformed import (
    TransformedBasis1D,
    NativeBasis1D,
)


def create_basis_1d(
    marginal: Any, bkd: Backend[Array]
) -> Union[TransformedBasis1D[Array], NativeBasis1D[Array]]:
    """Create a physical-domain basis for a marginal distribution.

    This factory creates the appropriate basis wrapper for a marginal:
    - For registered analytical marginals: uses TransformedBasis1D with
      the registered polynomial and transform
    - For registered discrete marginals: uses NativeBasis1D with the
      registered physical-domain polynomial
    - For custom marginals: falls back based on is_bounded()

    Parameters
    ----------
    marginal : Any
        Marginal distribution instance (e.g., UniformMarginal, BetaMarginal).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    TransformedBasis1D or NativeBasis1D
        Physical-domain basis wrapper.

    Raises
    ------
    ValueError
        If no polynomial/transform can be determined for the marginal.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability import UniformMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = UniformMarginal(0.0, 1.0, bkd)
    >>> basis = create_basis_1d(marginal, bkd)
    >>> basis.set_nterms(5)
    >>> samples = bkd.asarray([[0.0, 0.5, 1.0]])  # Physical domain [0, 1]
    >>> values = basis(samples)
    """
    # Import registry here to avoid circular imports
    from pyapprox.typing.surrogates.affine.univariate.registry import (
        _lookup_analytical,
        _lookup_discrete,
    )
    from pyapprox.typing.surrogates.affine.univariate.transforms import (
        BoundedAffineTransform1D,
        IdentityTransform1D,
    )
    from pyapprox.typing.surrogates.affine.univariate.globalpoly.continuous_numeric import (
        BoundedNumericOrthonormalPolynomial1D,
        UnboundedNumericOrthonormalPolynomial1D,
    )

    # Check analytical registry first
    entry = _lookup_analytical(marginal)
    if entry is not None:
        polynomial = entry.polynomial_factory(marginal, bkd)
        transform = entry.transform_factory(marginal, bkd)
        return TransformedBasis1D(polynomial, transform)

    # Check discrete registry
    entry_discrete = _lookup_discrete(marginal)
    if entry_discrete is not None:
        polynomial = entry_discrete.polynomial_factory(marginal, bkd)
        return NativeBasis1D(polynomial)

    # Fallback for custom marginals based on is_bounded()
    if hasattr(marginal, "is_bounded"):
        if marginal.is_bounded():
            # Bounded custom marginal: use BoundedNumericOrthonormalPolynomial1D
            polynomial = BoundedNumericOrthonormalPolynomial1D(bkd, marginal)
            lb, ub = marginal.bounds()
            transform = BoundedAffineTransform1D(bkd, lb, ub)
            return TransformedBasis1D(polynomial, transform)
        else:
            # Unbounded custom marginal: use UnboundedNumericOrthonormalPolynomial1D
            polynomial = UnboundedNumericOrthonormalPolynomial1D(bkd, marginal)
            # Physical domain polynomial, use identity transform wrapper
            return NativeBasis1D(polynomial)

    raise ValueError(
        f"Cannot determine basis for marginal type {type(marginal).__name__}. "
        "Register it with register_analytical_marginal() or ensure is_bounded() "
        "is defined."
    )


def create_bases_1d(
    marginals: List[Any], bkd: Backend[Array]
) -> List[Union[TransformedBasis1D[Array], NativeBasis1D[Array]]]:
    """Create physical-domain bases for multiple marginals.

    Parameters
    ----------
    marginals : List[Any]
        List of marginal distribution instances.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    List[TransformedBasis1D or NativeBasis1D]
        List of physical-domain basis wrappers, one per marginal.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability import UniformMarginal, GaussianMarginal
    >>> bkd = NumpyBkd()
    >>> marginals = [
    ...     UniformMarginal(0.0, 1.0, bkd),
    ...     GaussianMarginal(0.0, 1.0, bkd),
    ... ]
    >>> bases = create_bases_1d(marginals, bkd)
    >>> len(bases)
    2
    """
    return [create_basis_1d(m, bkd) for m in marginals]


__all__ = [
    "create_basis_1d",
    "create_bases_1d",
]
