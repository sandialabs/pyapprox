r"""The specification of a pointwise physics.

For fields :math:`u_a`, each with test function :math:`v_a`, the weak
residual is

.. math::

    R_a(u; v_a) = \int_\Omega v_a\, f_{0,a}(\chi) + \nabla v_a \cdot
    f_{1,a}(\chi)\, dx,
    \qquad \chi = (x, t, u_1, \ldots, u_m, \nabla u_1, \ldots,
    \nabla u_m, w, p),

with auxiliary data fields :math:`w` (values and gradients) and scalar
parameters :math:`p`. A :class:`PointwisePhysics` holds the
:math:`f_{0,a}` and :math:`f_{1,a}` as sympy expressions in the
:mod:`~pyapprox.pde.expressions.symbols` names: coordinates ``x, y, z``,
time ``t``, and for a field or auxiliary field ``u`` its value ``u`` and
gradient ``u_x, u_y, u_z``. Higher derivatives and outward normals are not
available to volume integrands and are refused.

Poisson :math:`-\Delta u = f` with ``f`` an auxiliary field is
``f0 = {"u": "-f"}``, ``f1 = {"u": ("u_x", "u_y")}``.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import sympy as sp

from pyapprox.pde.expressions.symbols import (
    check_coordinates,
    check_field_name,
    coordinates,
    gradient,
    normal,
    pde_names,
    time,
)
from pyapprox.util.expressions.names import CustomFunction, Names
from pyapprox.util.expressions.parsing import Value, parse


@dataclass(frozen=True)
class Field:
    """A solution field.

    Parameters
    ----------
    name : str
        Letters and digits starting with a letter; its gradient symbols are
        ``name_x, name_y, ...``.
    shape : tuple of int
        The shape of the field's value. Only scalar fields (``()``) are
        supported.

    Raises
    ------
    ValueError
        If ``name`` cannot name a field.
    NotImplementedError
        If ``shape`` is not ``()``.
    """

    name: str
    shape: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        check_field_name(self.name)
        if self.shape != ():
            raise NotImplementedError(
                f"only scalar fields are supported, got shape {self.shape} "
                f"for {self.name!r}"
            )


@dataclass(frozen=True)
class AuxField:
    """A data field known at quadrature points, by value and gradient.

    Parameters
    ----------
    name : str
        Letters and digits starting with a letter.

    Raises
    ------
    ValueError
        If ``name`` cannot name a field.
    """

    name: str

    def __post_init__(self) -> None:
        check_field_name(self.name)


class PointwisePhysics:
    """The pointwise integrands of a weak residual.

    Parameters
    ----------
    coords : Sequence[str]
        The coordinate names, e.g. ``standard(2)``.
    fields : Sequence[Field]
        The solution fields, in the order of the residual's blocks.
    f0 : Mapping[str, Value]
        Per field name, the integrand multiplying the test function, as
        text or sympy. A field left out has ``f0 = 0``.
    f1 : Mapping[str, Sequence[Value]]
        Per field name, the ``ndim`` components of the integrand
        multiplying the test function's gradient. A field left out has
        ``f1 = 0``.
    params : Sequence[str]
        Names of the scalar parameters.
    aux : Sequence[AuxField]
        The auxiliary data fields.
    constants : Mapping[str, float], optional
        Named constants, substituted when kernels are generated.
    functions : Mapping[str, CustomFunction], optional
        Functions beyond the built-in ones. Derivatives through one sympy
        cannot differentiate are refused when they are generated.

    Raises
    ------
    TypeError
        If a field is not a :class:`Field` or an auxiliary field not an
        :class:`AuxField`.
    ValueError
        If there are no coordinates or no fields; names repeat or clash
        with coordinates, time, normals or jets; ``f0`` or ``f1`` names an
        unknown field; an ``f1`` entry does not have ``ndim`` components;
        or an integrand fails to parse or uses an outward normal.
    """

    def __init__(
        self,
        coords: Sequence[str],
        fields: Sequence[Field],
        f0: Mapping[str, Value],
        f1: Mapping[str, Sequence[Value]],
        params: Sequence[str] = (),
        aux: Sequence[AuxField] = (),
        constants: Optional[Mapping[str, float]] = None,
        functions: Optional[Mapping[str, CustomFunction]] = None,
    ) -> None:
        check_coordinates(coords)
        if len(coords) == 0:
            raise ValueError("a pointwise physics needs at least one coordinate")
        if len(fields) == 0:
            raise ValueError("a pointwise physics needs at least one field")
        for field in fields:
            if not isinstance(field, Field):
                raise TypeError(f"fields must be Field, got {type(field).__name__}")
        for data in aux:
            if not isinstance(data, AuxField):
                raise TypeError(
                    f"aux fields must be AuxField, got {type(data).__name__}"
                )
        self._coords = tuple(coords)
        self._fields = tuple(fields)
        self._aux = tuple(aux)
        self._params = tuple(params)
        field_names = [field.name for field in self._fields]
        all_names = field_names + [data.name for data in self._aux]
        repeated = sorted({n for n in all_names if all_names.count(n) > 1})
        if repeated:
            raise ValueError(f"field and aux names declared more than once: {repeated}")
        # Values and gradients only: a quadrature point carries no higher
        # derivatives of a finite-element function.
        geometry = pde_names(coords, all_names, constants, functions, jet_order=1)
        self._names = Names(
            [s.name for s in geometry.variables()] + list(self._params),
            {s.name: v for s, v in geometry.constants().items()},
            geometry.functions(),
        )
        for name in list(f0) + list(f1):
            if name not in field_names:
                raise ValueError(
                    f"integrand given for {name!r}, which is not a field of "
                    f"{field_names}"
                )
        ndim = len(coords)
        normals = set(normal(coords))
        self._f0: Dict[str, sp.Expr] = {}
        self._f1: Dict[str, Tuple[sp.Expr, ...]] = {}
        for name in field_names:
            self._f0[name] = self._volume(f0.get(name, 0), normals, f"f0[{name!r}]")
            components = f1.get(name, (0,) * ndim)
            if isinstance(components, str) or len(components) != ndim:
                raise ValueError(
                    f"f1[{name!r}] must have {ndim} components, got {components!r}"
                )
            self._f1[name] = tuple(
                self._volume(c, normals, f"f1[{name!r}]") for c in components
            )

    def _volume(self, value: Value, normals: "set[sp.Symbol]", where: str) -> sp.Expr:
        expression = parse(value, self._names)
        used = sorted(str(s) for s in expression.free_symbols & normals)
        if used:
            raise ValueError(f"{where} uses the outward normal {used}")
        return expression

    def coords(self) -> Tuple[str, ...]:
        """The coordinate names."""
        return self._coords

    def ndim(self) -> int:
        """The number of coordinates."""
        return len(self._coords)

    def fields(self) -> Tuple[Field, ...]:
        """The solution fields, in block order."""
        return self._fields

    def field_names(self) -> Tuple[str, ...]:
        """The solution field names, in block order."""
        return tuple(field.name for field in self._fields)

    def aux(self) -> Tuple[AuxField, ...]:
        """The auxiliary data fields."""
        return self._aux

    def params(self) -> Tuple[str, ...]:
        """The parameter names, in the order of the parameter array."""
        return self._params

    def nparams(self) -> int:
        """The number of parameters."""
        return len(self._params)

    def names(self) -> Names:
        """Every name an integrand may use."""
        return self._names

    def coordinate_symbols(self) -> Tuple[sp.Symbol, ...]:
        """The coordinate symbols."""
        return coordinates(self._coords)

    def time_symbol(self) -> sp.Symbol:
        """The time symbol."""
        return time()

    def value_symbol(self, name: str) -> sp.Symbol:
        """The value symbol of field or aux field ``name``."""
        return sp.Symbol(name)

    def gradient_symbols(self, name: str) -> Tuple[sp.Symbol, ...]:
        """The gradient symbols of field or aux field ``name``."""
        return gradient(name, self._coords)

    def param_symbols(self) -> Tuple[sp.Symbol, ...]:
        """The parameter symbols, in order."""
        return tuple(sp.Symbol(name) for name in self._params)

    def f0(self, name: str) -> sp.Expr:
        """The integrand multiplying field ``name``'s test function."""
        return self._f0[name]

    def f1(self, name: str) -> Tuple[sp.Expr, ...]:
        """The ``ndim`` components multiplying field ``name``'s test
        function gradient."""
        return self._f1[name]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(coords={list(self._coords)}, "
            f"fields={list(self.field_names())}, params={list(self._params)}, "
            f"aux={[data.name for data in self._aux]})"
        )
