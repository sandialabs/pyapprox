r"""Exact values of quantities of interest, from a problem's exact solution.

The truth a stack's computed quantity is compared with. Each value is a
point value or a definite integral of the problem's own sympy
expressions, so it does not depend on the stack under test:

- :class:`~pyapprox.pde.verification.quantities.PointValue`:
  :math:`u^*(x_0)`;
- :class:`~pyapprox.pde.verification.quantities.Integral`:
  :math:`\int_\Omega u^* \,dx`;
- :class:`~pyapprox.pde.verification.quantities.BoundaryFlux`:
  :math:`\int_\Gamma q(u^*)\, n \,ds`.

Integrals are taken symbolically; one that sympy cannot do in closed form
is evaluated by mpmath quadrature, in up to three dimensions. Values are
scalars: no array library is involved.

Limits: steady problems only (a transient problem's quantity needs a
time); integrals and fluxes need a :class:`~pyapprox.pde.verification.
domains.Box` (a disk would need polar coordinates, not built).
"""

from typing import List, Tuple

import sympy as sp

from pyapprox.pde.expressions.symbols import coordinates, normal
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.quantities import (
    BoundaryFlux,
    Integral,
    PointValue,
    QuantityProtocol,
)

_QUADRATURE_DIGITS = 17
_MAX_QUADRATURE_DIMENSIONS = 3


def _definite_integral(
    integrand: sp.Expr, limits: List[Tuple[sp.Symbol, float, float]]
) -> sp.Expr:
    """The integral of ``integrand`` over ``limits``: symbolic, else by
    mpmath quadrature."""
    if not limits:
        return integrand
    value = sp.integrate(integrand, *limits)
    if not value.has(sp.Integral):
        return sp.sympify(value)
    if len(limits) > _MAX_QUADRATURE_DIMENSIONS:
        raise ValueError(
            f"no closed form, and quadrature is built for at most "
            f"{_MAX_QUADRATURE_DIMENSIONS} dimensions, not {len(limits)}"
        )
    import mpmath

    symbols = [symbol for symbol, _, _ in limits]
    function = sp.lambdify(symbols, integrand, "mpmath")
    with mpmath.workdps(_QUADRATURE_DIGITS):
        result = mpmath.quad(function, *[[lower, upper] for _, lower, upper in limits])
    return sp.Float(result, _QUADRATURE_DIGITS)


class _ExactValues:
    """Each quantity at the exact solution, as a sympy number."""

    def __init__(self, problem: ManufacturedProblem) -> None:
        self._problem = problem
        self._symbols = dict(
            zip(problem.coordinates(), coordinates(problem.coordinates()))
        )

    def _field(self, name: str) -> sp.Expr:
        if name not in self._problem.fields():
            raise ValueError(
                f"no field {name!r}; the fields are {list(self._problem.fields())}"
            )
        return self._problem.solution(name)

    def _box(self, quantity: QuantityProtocol) -> Box:
        domain = self._problem.domain()
        if not isinstance(domain, Box):
            raise ValueError(
                f"the exact value of {quantity!r} needs a Box domain, got {domain!r}"
            )
        return domain

    def point_value(self, quantity: PointValue) -> sp.Expr:
        r""":math:`u^*(x_0)`."""
        solution = self._field(quantity.field())
        point = quantity.point()
        if len(point) != self._problem.ndim():
            raise ValueError(
                f"{quantity!r} needs one value per coordinate "
                f"{list(self._problem.coordinates())}"
            )
        return solution.subs(dict(zip(self._symbols.values(), point)))

    def integral(self, quantity: Integral) -> sp.Expr:
        r""":math:`\int_\Omega u^* \,dx`."""
        solution = self._field(quantity.field())
        box = self._box(quantity)
        limits = [
            (self._symbols[name], lower, upper)
            for name, (lower, upper) in zip(box.coordinates(), box.bounds())
        ]
        return _definite_integral(solution, limits)

    def boundary_flux(self, quantity: BoundaryFlux) -> sp.Expr:
        r""":math:`\int_\Gamma q(u^*)\, n \,ds` over one side of a box."""
        box = self._box(quantity)
        flux = self._problem.normal_flux(quantity.row())
        across, sign = box.side(quantity.part())
        names = box.coordinates()
        normals = normal(names)
        bounds = dict(zip(names, box.bounds()))
        lower, upper = bounds[across]
        on_side = {
            self._symbols[across]: lower if sign < 0 else upper,
            **{n: (sign if name == across else 0.0) for name, n in zip(names, normals)},
        }
        limits = [
            (self._symbols[name], *bounds[name]) for name in names if name != across
        ]
        return _definite_integral(flux.subs(on_side), limits)


def exact_value(problem: ManufacturedProblem, quantity: QuantityProtocol) -> float:
    """The value of ``quantity`` at ``problem``'s exact solution.

    Raises
    ------
    TypeError
        If ``quantity`` does not satisfy ``QuantityProtocol``.
    ValueError
        If the problem is transient, the field, row or boundary part is
        not the problem's, an integral or flux is asked on a domain other
        than a box, a point has the wrong length, or an integral has no
        closed form in more than three dimensions.
    """
    if not isinstance(quantity, QuantityProtocol):
        raise TypeError(
            f"quantity must satisfy QuantityProtocol, got {type(quantity).__name__}"
        )
    if problem.is_transient():
        raise ValueError(
            f"{quantity!r} of a transient problem needs a time; exact values "
            "are built for steady problems only"
        )
    value = quantity.accept(_ExactValues(problem))
    if not value.is_number:
        raise ValueError(f"{quantity!r} did not reduce to a number: {value}")
    return float(value)
