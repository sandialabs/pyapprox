"""Registry for marginal → polynomial/transform mappings.

This module provides a single source of truth for mapping marginal distributions
to their corresponding orthonormal polynomials and domain transforms.

The registry pattern ensures that polynomial and transform logic stay in sync,
preventing silent bugs when adding new marginal types.

Extension Points
----------------
- For new analytical marginals: use register_analytical_marginal()
- For new discrete marginals: use register_discrete_marginal()
- For custom marginals: no registration needed (uses numeric polynomials)

Example
-------
>>> from pyapprox.surrogates.affine.univariate.registry import (
...     _lookup_analytical,
...     _lookup_discrete,
... )
>>> from pyapprox.probability import UniformMarginal
>>> from pyapprox.util.backends.numpy import NumpyBkd
>>> bkd = NumpyBkd()
>>> marginal = UniformMarginal(-1.0, 1.0, bkd)
>>> entry = _lookup_analytical(marginal)
>>> poly = entry.polynomial_factory(marginal, bkd)
>>> transform = entry.transform_factory(marginal, bkd)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from pyapprox.util.backends.protocols import Array, Backend


@dataclass
class AnalyticalMarginalEntry:
    """Registry entry for analytical marginal types.

    Attributes
    ----------
    polynomial_factory : Callable
        Factory function (marginal, bkd) -> OrthonormalPolynomial1D
    transform_factory : Callable
        Factory function (marginal, bkd) -> Univariate1DTransformProtocol
    """

    polynomial_factory: Callable[[Any, Backend[Array]], Any]
    transform_factory: Callable[[Any, Backend[Array]], Any]


@dataclass
class DiscreteMarginalEntry:
    """Registry entry for discrete marginal types.

    Discrete marginals use physical-domain polynomials and don't need transforms.

    Attributes
    ----------
    polynomial_factory : Callable
        Factory function (marginal, bkd) -> OrthonormalPolynomial1D
    """

    polynomial_factory: Callable[[Any, Backend[Array]], Any]


# Single source of truth for analytical marginal mappings
ANALYTICAL_MARGINAL_REGISTRY: Dict[Type[Any], AnalyticalMarginalEntry] = {}

# Single source of truth for discrete marginal mappings
DISCRETE_MARGINAL_REGISTRY: Dict[Type[Any], DiscreteMarginalEntry] = {}


def register_analytical_marginal(
    marginal_type: Type[Any],
    polynomial_factory: Callable[[Any, Backend[Array]], Any],
    transform_factory: Callable[[Any, Backend[Array]], Any],
) -> None:
    """Register an analytical marginal with its polynomial and transform factories.

    Parameters
    ----------
    marginal_type : Type
        The marginal class to register.
    polynomial_factory : Callable[[marginal, bkd], OrthonormalPolynomial1D]
        Factory function to create polynomial from marginal.
    transform_factory : Callable[[marginal, bkd], Univariate1DTransformProtocol]
        Factory function to create transform from marginal.

    Example
    -------
    >>> from pyapprox.probability import UniformMarginal
    >>> from pyapprox.surrogates.affine.univariate.globalpoly import (
    ...     LegendrePolynomial1D,
    ... )
    >>> from pyapprox.surrogates.affine.univariate.transforms import (
    ...     BoundedAffineTransform1D,
    ... )
    >>> register_analytical_marginal(
    ...     UniformMarginal,
    ...     lambda m, bkd: LegendrePolynomial1D(bkd),
    ...     lambda m, bkd: BoundedAffineTransform1D(bkd, m.lower(), m.upper()),
    ... )
    """
    ANALYTICAL_MARGINAL_REGISTRY[marginal_type] = AnalyticalMarginalEntry(
        polynomial_factory=polynomial_factory,
        transform_factory=transform_factory,
    )


def register_discrete_marginal(
    marginal_type: Type[Any],
    polynomial_factory: Callable[[Any, Backend[Array]], Any],
) -> None:
    """Register a discrete marginal with its polynomial factory.

    Discrete marginals use physical-domain polynomials and don't need transforms.

    Parameters
    ----------
    marginal_type : Type
        The discrete marginal class to register.
    polynomial_factory : Callable[[marginal, bkd], OrthonormalPolynomial1D]
        Factory function to create polynomial from marginal.
    """
    DISCRETE_MARGINAL_REGISTRY[marginal_type] = DiscreteMarginalEntry(
        polynomial_factory=polynomial_factory,
    )


def _lookup_analytical(marginal: Any) -> Optional[AnalyticalMarginalEntry]:
    """Look up registry entry for analytical marginal type.

    Parameters
    ----------
    marginal : Any
        Marginal distribution instance.

    Returns
    -------
    AnalyticalMarginalEntry or None
        Registry entry if found, None otherwise.
    """
    for marginal_type, entry in ANALYTICAL_MARGINAL_REGISTRY.items():
        if isinstance(marginal, marginal_type):
            return entry
    return None


def _lookup_discrete(marginal: Any) -> Optional[DiscreteMarginalEntry]:
    """Look up registry entry for discrete marginal type.

    Parameters
    ----------
    marginal : Any
        Marginal distribution instance.

    Returns
    -------
    DiscreteMarginalEntry or None
        Registry entry if found, None otherwise.
    """
    for marginal_type, entry in DISCRETE_MARGINAL_REGISTRY.items():
        if isinstance(marginal, marginal_type):
            return entry
    return None


def _register_builtins() -> None:
    """Register built-in marginal types.

    This is called on module import to register standard distributions.
    """
    # Import here to avoid circular imports
    from pyapprox.probability import (
        BetaMarginal,
        CustomDiscreteMarginal,
        GammaMarginal,
        GaussianMarginal,
        ScipyDiscreteMarginal,
        UniformMarginal,
    )
    from pyapprox.surrogates.affine.univariate.globalpoly import (
        CharlierPolynomial1D,
        HermitePolynomial1D,
        JacobiPolynomial1D,
        LaguerrePolynomial1D,
        LegendrePolynomial1D,
    )
    from pyapprox.surrogates.affine.univariate.globalpoly.numeric import (
        DiscreteNumericOrthonormalPolynomial1D,
    )
    from pyapprox.surrogates.affine.univariate.transforms import (
        BoundedAffineTransform1D,
        UnboundedAffineTransform1D,
    )

    # Analytical marginals (use polynomial + transform)
    register_analytical_marginal(
        UniformMarginal,
        lambda m, bkd: LegendrePolynomial1D(bkd),
        lambda m, bkd: BoundedAffineTransform1D(bkd, m.lower(), m.upper()),
    )
    register_analytical_marginal(
        GaussianMarginal,
        lambda m, bkd: HermitePolynomial1D(bkd, rho=0.0, prob_meas=True),
        lambda m, bkd: UnboundedAffineTransform1D(
            bkd, loc=m.mean_value(), scale=m.std()
        ),
    )
    register_analytical_marginal(
        BetaMarginal,
        lambda m, bkd: JacobiPolynomial1D(m.beta() - 1, m.alpha() - 1, bkd),
        lambda m, bkd: BoundedAffineTransform1D(bkd, *m.bounds()),
    )
    register_analytical_marginal(
        GammaMarginal,
        lambda m, bkd: LaguerrePolynomial1D(bkd, rho=m.shape() - 1),
        lambda m, bkd: UnboundedAffineTransform1D(bkd, loc=0.0, scale=m.scale()),
    )

    # Discrete marginals (physical domain polynomials, no transform)
    def _scipy_discrete_poly_factory(m: ScipyDiscreteMarginal, bkd):
        if m.name == "poisson":
            return CharlierPolynomial1D(bkd, mu=m.shapes["mu"])
        else:
            xk, pk = _get_scipy_discrete_probability_masses(m, bkd)
            return DiscreteNumericOrthonormalPolynomial1D(bkd, xk, pk)

    register_discrete_marginal(
        ScipyDiscreteMarginal,
        _scipy_discrete_poly_factory,
    )

    register_discrete_marginal(
        CustomDiscreteMarginal,
        lambda m, bkd: DiscreteNumericOrthonormalPolynomial1D(
            bkd, *m.probability_masses()
        ),
    )


def _get_scipy_discrete_probability_masses(marginal, bkd):
    """Get probability masses for scipy discrete marginal.

    Parameters
    ----------
    marginal : ScipyDiscreteMarginal
        The discrete marginal.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    xk : Array
        Support points.
    pk : Array
        Probability masses.
    """
    # Get support from marginal
    xk, pk = marginal.probability_masses()
    return xk, pk


# Register builtins on module import
_register_builtins()


__all__ = [
    "AnalyticalMarginalEntry",
    "DiscreteMarginalEntry",
    "ANALYTICAL_MARGINAL_REGISTRY",
    "DISCRETE_MARGINAL_REGISTRY",
    "register_analytical_marginal",
    "register_discrete_marginal",
    "_lookup_analytical",
    "_lookup_discrete",
]
