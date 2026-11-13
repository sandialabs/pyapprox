r"""Manufactured verification problems, as sympy expressions.

A problem is an exact solution :math:`u^*`, the physics' operator and
coefficients, a domain and a condition on each named boundary part. It
gives the solution, the forcing that makes :math:`u^*` exact, the flux
:math:`q(u^*)` in the convention the problem states, and each boundary
part's data, all as sympy expressions in the symbols of
:mod:`pyapprox.pde.verification.symbols`. It evaluates nothing: a stack
binds the symbols to its own points, so the same problem verifies any
discretization, serial or distributed.

A problem is transient when its expressions depend on time ``t``; its
forcing then includes :math:`\partial_t u^*`.
"""

from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import sympy as sp

from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
)
from pyapprox.pde.manufactured.advection_diffusion import (
    ManufacturedAdvectionDiffusionReaction,
)
from pyapprox.pde.verification.conditions import ConditionProtocol
from pyapprox.pde.verification.domains import DomainProtocol
from pyapprox.pde.verification.symbols import coordinates, normal, state, time
from pyapprox.util.backends.numpy import NumpyBkd

_Exprs = Tuple[sp.Expr, ...]
_Rows = Tuple[Tuple[sp.Expr, ...], ...]


def _exprs(values: Iterable[Union[sp.Expr, str, float]]) -> _Exprs:
    return tuple(sp.sympify(value) for value in values)


def _check_symbols(label: str, expression: sp.Expr, allowed: "set[sp.Symbol]") -> None:
    extra = expression.free_symbols - allowed
    if extra:
        raise ValueError(
            f"{label} may depend only on {sorted(map(str, allowed))}; got "
            f"{sorted(map(str, extra))}"
        )


class ManufacturedProblem:
    r"""An exact solution, its forcing and flux, a domain and conditions.

    Parameters
    ----------
    domain : DomainProtocol
        Where the problem is posed.
    conditions : Mapping[str, ConditionProtocol]
        A condition on every boundary part of ``domain``, by name.
    solution : Sequence
        :math:`u^*`, one expression per component.
    forcing : Sequence
        The forcing for which :math:`u^*` solves the problem, per
        component.
    flux : Sequence[Sequence]
        :math:`q(u^*)`: rows by component, columns by direction.

    Raises
    ------
    TypeError
        If ``domain`` or a condition is of the wrong type.
    ValueError
        If the conditions do not name exactly the domain's boundary parts,
        the shapes disagree, or an expression has symbols other than the
        domain's coordinates and time.
    """

    def __init__(
        self,
        domain: DomainProtocol,
        conditions: Mapping[str, ConditionProtocol],
        solution: Sequence[sp.Expr],
        forcing: Sequence[sp.Expr],
        flux: Sequence[Sequence[sp.Expr]],
    ) -> None:
        if not isinstance(domain, DomainProtocol):
            raise TypeError(
                f"domain must satisfy DomainProtocol, got {type(domain).__name__}"
            )
        names = set(domain.boundaries())
        missing, unknown = names - set(conditions), set(conditions) - names
        if missing or unknown:
            raise ValueError(
                "conditions must name every boundary part of the domain and no "
                f"other: missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        for name, condition in conditions.items():
            if not isinstance(condition, ConditionProtocol):
                raise TypeError(
                    f"condition {name!r} must satisfy ConditionProtocol, got "
                    f"{type(condition).__name__}"
                )
        ndim = domain.ndim()
        self._domain = domain
        self._conditions = MappingProxyType(dict(conditions))
        self._solution = _exprs(solution)
        self._forcing = _exprs(forcing)
        self._flux: _Rows = tuple(_exprs(row) for row in flux)
        ncomponents = len(self._solution)
        if (
            len(self._forcing) != ncomponents
            or len(self._flux) != ncomponents
            or any(len(row) != ndim for row in self._flux)
        ):
            raise ValueError(
                f"with {ncomponents} components in {ndim} dimensions, forcing "
                f"needs {ncomponents} entries and flux {ncomponents} rows of {ndim}"
            )
        allowed = set(coordinates(ndim)) | {time()}
        for expression in self._solution + self._forcing + sum(self._flux, ()):
            _check_symbols("expressions", expression, allowed)

    def domain(self) -> DomainProtocol:
        """Where the problem is posed."""
        return self._domain

    def conditions(self) -> Mapping[str, ConditionProtocol]:
        """The condition on each boundary part, by name (read only)."""
        return self._conditions

    def ndim(self) -> int:
        """The spatial dimension."""
        return self._domain.ndim()

    def ncomponents(self) -> int:
        """The number of solution components."""
        return len(self._solution)

    def is_transient(self) -> bool:
        """Whether any expression, boundary data included, depends on
        ``t``."""
        expressions = self._solution + self._forcing + sum(self._flux, ())
        expressions += sum((self.boundary_data(n) for n in self._conditions), ())
        return any(time() in e.free_symbols for e in expressions)

    def solution(self) -> _Exprs:
        r""":math:`u^*`, per component."""
        return self._solution

    def forcing(self) -> _Exprs:
        """The forcing, per component."""
        return self._forcing

    def flux(self) -> _Rows:
        r""":math:`q(u^*)`: rows by component, columns by direction."""
        return self._flux

    def normal_flux(self) -> _Exprs:
        r""":math:`q(u^*)\, n` per component, in the coordinate, time and
        normal symbols."""
        n = normal(self.ndim())
        return tuple(sp.Add(*(q * nj for q, nj in zip(row, n))) for row in self._flux)

    def boundary_data(self, name: str) -> _Exprs:
        """The data of the condition on boundary part ``name``.

        Raises
        ------
        ValueError
            If the domain has no boundary part ``name``.
        """
        if name not in self._conditions:
            raise ValueError(
                f"no boundary part {name!r}; the domain has "
                f"{list(self._domain.boundaries())}"
            )
        return self._conditions[name].data(self)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(solution={list(self._solution)}, "
            f"domain={self._domain!r}, conditions={dict(self._conditions)!r})"
        )


class AdvectionDiffusionReactionProblem(ManufacturedProblem):
    r"""An advection-diffusion-reaction problem.

    Strong form, for the scalar :math:`u`:

    .. math::

        \partial_t u - \nabla\cdot(D \nabla u) + a(v, u) - R(u) = f,

    without :math:`\partial_t u` when the solution is steady, and with
    :math:`a(v, u) = v\cdot\nabla u` (:class:`~pyapprox.pde.constitutive.
    advection.NonConservativeAdvection`) or :math:`\nabla\cdot(v u)`
    (:class:`~pyapprox.pde.constitutive.advection.ConservativeAdvection`).
    The flux, which natural conditions prescribe along the outward normal,
    is :math:`q = -D \nabla u + v u` for the conservative form and
    :math:`q = -D \nabla u` for the non-conservative one: the term the weak
    form integrates by parts.

    Parameters
    ----------
    solution : str
        :math:`u^*`, in ``x``, ``y``, ``z`` and ``t``.
    domain : DomainProtocol
        Where the problem is posed.
    conditions : Mapping[str, ConditionProtocol]
        A condition on every boundary part, by name.
    diffusion : str
        :math:`D`, in the coordinates and time.
    reaction : str
        :math:`R(u)`, in ``u``, the coordinates and time.
    velocity : Sequence[str], optional
        :math:`v`, one entry per direction. Default: zero.
    advection_form : AdvectionFormProtocol, optional
        How :math:`a` is written. Default: conservative.

    Raises
    ------
    TypeError
        If ``domain`` or ``advection_form`` is of the wrong type (and as
        :class:`ManufacturedProblem`).
    ValueError
        If ``velocity`` does not have one entry per direction, or a
        coefficient depends on other symbols (``diffusion`` on ``u``
        included) (and as :class:`ManufacturedProblem`).
    """

    def __init__(
        self,
        solution: str,
        domain: DomainProtocol,
        conditions: Mapping[str, ConditionProtocol],
        diffusion: str = "1",
        reaction: str = "0",
        velocity: Optional[Sequence[str]] = None,
        advection_form: Optional[AdvectionFormProtocol] = None,
    ) -> None:
        if not isinstance(domain, DomainProtocol):
            raise TypeError(
                f"domain must satisfy DomainProtocol, got {type(domain).__name__}"
            )
        ndim = domain.ndim()
        velocity = ["0"] * ndim if velocity is None else list(velocity)
        if len(velocity) != ndim:
            raise ValueError(
                f"velocity needs {ndim} entries, one per direction; got {len(velocity)}"
            )
        if advection_form is None:
            advection_form = ConservativeAdvection()
        if not isinstance(advection_form, AdvectionFormProtocol):
            raise TypeError(
                "advection_form must satisfy AdvectionFormProtocol, got "
                f"{type(advection_form).__name__}"
            )
        allowed = set(coordinates(ndim)) | {time()}
        self._diffusion: sp.Expr = sp.sympify(diffusion)
        self._reaction: sp.Expr = sp.sympify(reaction)
        self._velocity = _exprs(velocity)
        _check_symbols("diffusion", self._diffusion, allowed)
        _check_symbols("reaction", self._reaction, allowed | {state()})
        for v in self._velocity:
            _check_symbols("velocity", v, allowed)
        self._advection_form = advection_form
        # The generator writes time as T; here it is t. It also builds numpy
        # callables, which need a backend; only its expressions are read.
        to_generator = {time(): sp.Symbol("T")}
        expressions = ManufacturedAdvectionDiffusionReaction(
            str(sp.sympify(solution).subs(to_generator)),
            ndim,
            str(self._diffusion.subs(to_generator)),
            str(self._reaction.subs(to_generator)),
            [str(v.subs(to_generator)) for v in self._velocity],
            NumpyBkd(),
            advection_form=advection_form,
        ).expressions()
        from_generator = {sp.Symbol("T"): time()}
        super().__init__(
            domain,
            conditions,
            (sp.sympify(expressions["solution"]).subs(from_generator),),
            (sp.sympify(expressions["forcing"]).subs(from_generator),),
            (tuple(sp.sympify(q).subs(from_generator) for q in expressions["flux"]),),
        )

    def diffusion(self) -> sp.Expr:
        """:math:`D`, in the coordinates and time."""
        return self._diffusion

    def reaction(self) -> sp.Expr:
        """:math:`R(u)`, in ``u``, the coordinates and time."""
        return self._reaction

    def velocity(self) -> _Exprs:
        """:math:`v`, per direction."""
        return self._velocity

    def advection_form(self) -> AdvectionFormProtocol:
        """How the advection term is written."""
        return self._advection_form
