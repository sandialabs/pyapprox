"""Protocols for quadrature rules in sparse grid and interpolation contexts.

This module defines protocols for classes that provide quadrature points
and weights. There are two main patterns:

1. **Generator-style** (`QuadratureRuleGeneratorProtocol`):
   - `quadrature_rule(npoints: int) -> Tuple[Array, Array]`
   - Used by: `LejaSequence1D`, orthonormal polynomial `gauss_quadrature_rule`
   - The caller specifies the number of points

2. **Stateful-style** (`QuadratureRuleStatefulProtocol`):
   - `quadrature_rule() -> Tuple[Array, Array]`
   - Used by: `LagrangeBasis1D`, piecewise polynomial bases
   - Requires `set_nterms()` to be called first

Note: `Basis1DHasQuadratureProtocol` in `basis1d.py` uses the generator-style
signature. The protocols here provide additional options for sparse grid and
interpolation contexts where stateful bases are used.
"""

from typing import Protocol, Generic, Tuple, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class QuadratureRuleGeneratorProtocol(Protocol, Generic[Array]):
    """Protocol for quadrature rule generators.

    Classes implementing this protocol can generate quadrature rules
    for a specified number of points. The points may be computed on-demand
    (e.g., Leja sequences) or from closed-form expressions (e.g., Gauss).

    This is the "generator" pattern where the caller specifies npoints.

    Examples
    --------
    `LejaSequence1D` satisfies this protocol:

    >>> isinstance(leja_sequence, QuadratureRuleGeneratorProtocol)
    True

    Orthonormal polynomials' `gauss_quadrature_rule` method also satisfies
    this pattern (though the class itself has additional methods).
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Return quadrature points and weights.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints, 1)
        """
        ...


@runtime_checkable
class QuadratureRuleStatefulProtocol(Protocol, Generic[Array]):
    """Protocol for stateful quadrature rules.

    Classes implementing this protocol provide quadrature rules based on
    their current state (typically set via `set_nterms()`). The number
    of quadrature points is determined by the current `nterms` setting.

    This is the "stateful" pattern used by interpolation bases like
    `LagrangeBasis1D` and piecewise polynomial bases.

    Examples
    --------
    `LagrangeBasis1D` satisfies this protocol:

    >>> basis = LagrangeBasis1D(bkd, quad_rule_func)
    >>> basis.set_nterms(5)
    >>> pts, wts = basis.quadrature_rule()  # Returns 5-point rule
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms.

        Parameters
        ----------
        nterms : int
            Number of basis terms (and quadrature points).
        """
        ...

    def nterms(self) -> int:
        """Return the number of basis terms."""
        ...

    def quadrature_rule(self) -> Tuple[Array, Array]:
        """Return quadrature points and weights for current nterms.

        Must call `set_nterms()` before using this method.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, nterms)
        weights : Array
            Quadrature weights. Shape: (nterms, 1)

        Raises
        ------
        ValueError
            If `set_nterms()` has not been called.
        """
        ...


__all__ = [
    "QuadratureRuleGeneratorProtocol",
    "QuadratureRuleStatefulProtocol",
]
