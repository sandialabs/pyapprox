r"""Advection formulations: the non-conservative and conservative forms.

The advection term of an advection-diffusion-reaction equation can be
written two ways, which agree for divergence-free velocity:

- non-conservative, :math:`v\cdot\nabla u`, kept as it is in the weak form,
  so it adds no boundary term;
- conservative, :math:`\nabla\cdot(v u)`, integrated by parts in the weak
  form, which moves the derivative onto the test function and adds the
  boundary flux :math:`(v\cdot n) u`.

A form states its strong term, its weak-form integrand and the advective
flux it integrates by parts, each written once with plain arithmetic, so
the same definition serves the symbolic side (manufactured solutions,
on sympy expressions) and the numeric side (Galerkin assembly, on arrays
or skfem fields). A consumer depends only on ``AdvectionFormProtocol``;
another formulation is added by implementing it and injecting the
instance.
"""

from typing import (
    Callable,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

_F = TypeVar("_F", bound="FieldExpression")


class FieldExpression(Protocol):
    """Values that support the arithmetic a formulation is written in.

    Sympy expressions, NumPy arrays and skfem fields all qualify.
    """

    def __add__(self: _F, other: _F, /) -> _F: ...

    def __mul__(self: _F, other: _F, /) -> _F: ...

    def __neg__(self: _F) -> _F: ...


def _sum(terms: Sequence[_F]) -> _F:
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


@runtime_checkable
class AdvectionFormProtocol(Protocol):
    """One way of writing the advection term.

    Velocities are sequences of components, one per spatial dimension.
    """

    def strong_term(
        self,
        velocity: Sequence[_F],
        u: _F,
        derivative: Callable[[_F, int], _F],
    ) -> _F:
        """Return the strong-form term, :math:`v\\cdot\\nabla u` or
        :math:`\\nabla\\cdot(v u)`.

        ``derivative(expr, i)`` differentiates along coordinate ``i``.
        """
        ...

    def weak_integrand(
        self,
        velocity: Sequence[_F],
        u: _F,
        grad_u: Sequence[_F],
        test: _F,
        grad_test: Sequence[_F],
    ) -> _F:
        """Return the bilinear integrand ``a(u, w)`` of the advection term,
        with the sign it has in the stiffness ``K``."""
        ...

    def flux(self, velocity: Sequence[_F], u: _F) -> Optional[List[_F]]:
        """Return the advective flux the weak form integrates by parts,
        one component per dimension, or ``None`` when the form integrates
        nothing by parts (which differs from a flux that is zero)."""
        ...


class NonConservativeAdvection:
    r"""Advection as :math:`v\cdot\nabla u`, kept as written in the weak form.

    Weak integrand :math:`(v\cdot\nabla u)\, w`. Nothing is integrated by
    parts, so the form adds no boundary term and has no flux.
    """

    def strong_term(
        self,
        velocity: Sequence[_F],
        u: _F,
        derivative: Callable[[_F, int], _F],
    ) -> _F:
        return _sum(
            [vel * derivative(u, ii) for ii, vel in enumerate(velocity)]
        )

    def weak_integrand(
        self,
        velocity: Sequence[_F],
        u: _F,
        grad_u: Sequence[_F],
        test: _F,
        grad_test: Sequence[_F],
    ) -> _F:
        return _sum([vel * du for vel, du in zip(velocity, grad_u)]) * test

    def flux(self, velocity: Sequence[_F], u: _F) -> Optional[List[_F]]:
        return None

    def __repr__(self) -> str:
        return "NonConservativeAdvection()"


class ConservativeAdvection:
    r"""Advection as :math:`\nabla\cdot(v u)`, integrated by parts.

    Weak integrand :math:`-u\, v\cdot\nabla w`; the boundary flux is
    :math:`v u`.
    """

    def strong_term(
        self,
        velocity: Sequence[_F],
        u: _F,
        derivative: Callable[[_F, int], _F],
    ) -> _F:
        return _sum(
            [derivative(vel * u, ii) for ii, vel in enumerate(velocity)]
        )

    def weak_integrand(
        self,
        velocity: Sequence[_F],
        u: _F,
        grad_u: Sequence[_F],
        test: _F,
        grad_test: Sequence[_F],
    ) -> _F:
        return -(
            u * _sum([vel * dw for vel, dw in zip(velocity, grad_test)])
        )

    def flux(self, velocity: Sequence[_F], u: _F) -> Optional[List[_F]]:
        return [vel * u for vel in velocity]

    def __repr__(self) -> str:
        return "ConservativeAdvection()"
