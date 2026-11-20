"""A pointwise physics bound to backend functions of quadrature points.

:class:`BackendPointwiseKernel` is the backend target of the generated
sets (:mod:`~pyapprox.pde.pointwise.generated`): it lambdifies each set
(residual, Jacobian, parameter derivatives) with the backend's own
functions, computing the set's common subexpressions as generated, and
reads the results back by the set's entries. Quadrature points are the
last axis of every input and output. The kernel runs on backend arrays
throughout, so it follows the backend's dtype and device, and a
differentiable backend differentiates through it.

It serves stacks that assemble in Python (skfem) and the torch and JAX
checks; compiled stacks print the same sets instead.

Symbolically zero derivatives are ``None`` in the results and never
evaluated.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Mapping, Optional, Sequence, Tuple

import sympy as sp

from pyapprox.pde.pointwise.derivatives import PointwiseDerivatives
from pyapprox.pde.pointwise.generated import GeneratedSet, GeneratedSets, generate
from pyapprox.pde.pointwise.spec import PointwisePhysics
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.expressions.evaluation import (
    backend_namespace,
    check_backend_functions,
)
from pyapprox.util.expressions.names import Names


@dataclass(frozen=True)
class PointwiseInputs(Generic[Array]):
    """The inputs of a kernel at ``nq`` quadrature points.

    Only what the evaluated expressions use must be given.

    Attributes
    ----------
    coords : Array
        The points, shape ``(ndim, nq)``.
    values : Mapping[str, Array]
        Field and aux-field values by name, each shape ``(nq,)``.
    gradients : Mapping[str, Array]
        Field and aux-field gradients by name, each shape ``(ndim, nq)``.
    params : Array, optional
        The parameters in the physics' order, shape ``(nparams,)``.
    t : float
        The time.
    """

    coords: Array
    values: Mapping[str, Array] = field(default_factory=dict)
    gradients: Mapping[str, Array] = field(default_factory=dict)
    params: Optional[Array] = None
    t: float = 0.0


@dataclass(frozen=True)
class PointwiseResidual(Generic[Array]):
    """One field's residual integrands.

    Attributes
    ----------
    f0 : Array
        Shape ``(nq,)``.
    f1 : Array
        Shape ``(ndim, nq)``.
    """

    f0: Array
    f1: Array


@dataclass(frozen=True)
class PointwiseTerms(Generic[Array]):
    """One field's integrands differentiated by one scalar input.

    Attributes
    ----------
    f0 : Array, optional
        Shape ``(nq,)``; ``None`` when symbolically zero.
    f1 : Array, optional
        Shape ``(ndim, nq)``; ``None`` when symbolically zero.
    """

    f0: Optional[Array]
    f1: Optional[Array]


@dataclass(frozen=True)
class PointwiseJacobianBlock(Generic[Array]):
    """The Jacobian blocks of one (test, trial) field pair at the points.

    Each is ``None`` when symbolically zero; see
    :class:`~pyapprox.pde.pointwise.derivatives.SymbolicJacobianBlock`.

    Attributes
    ----------
    g0 : Array, optional
        Shape ``(nq,)``.
    g1 : Array, optional
        Shape ``(ndim, nq)``.
    g2 : Array, optional
        Shape ``(ndim, nq)``.
    g3 : Array, optional
        Shape ``(ndim, ndim, nq)``.
    """

    g0: Optional[Array]
    g1: Optional[Array]
    g2: Optional[Array]
    g3: Optional[Array]


# Where a variable's values come from: ("coord", "", i), ("time", "", 0),
# ("value", name, 0), ("gradient", name, i) or ("param", name, k).
_Source = Tuple[str, str, int]
# An entry of a generated set: the term's name followed by field and
# parameter names, e.g. ("g3", "u", "w").
_Key = Tuple[str, ...]


class _BoundSet(Generic[Array]):
    """A generated set lambdified with backend functions, its common
    subexpressions taken as generated rather than recomputed."""

    def __init__(
        self, generated: GeneratedSet, names: Names, bkd: Backend[Array]
    ) -> None:
        for _, expression in generated.replacements:
            check_backend_functions(expression, names)
        for expression in generated.outputs:
            check_backend_functions(expression, names)
        self._generated, self._bkd = generated, bkd
        reduction = (list(generated.replacements), list(generated.outputs))
        self._function: Optional[Callable[..., Sequence[Array]]] = None
        if generated.noutputs() > 0:
            self._function = sp.lambdify(
                generated.variables,
                list(generated.outputs),
                modules=[backend_namespace(bkd, names)],
                cse=lambda _: reduction,
            )

    def variables(self) -> Tuple[sp.Symbol, ...]:
        return self._generated.variables

    def __call__(self, rows: Mapping[sp.Symbol, Array], npts: int) -> Dict[_Key, Array]:
        if self._function is None:
            return {}
        zeros = self._bkd.zeros((npts,))
        # A constant output is a scalar: broadcast every output to the points.
        results = [
            r + zeros for r in self._function(*(rows[s] for s in self.variables()))
        ]
        out: Dict[_Key, Array] = {}
        for entry in self._generated.entries:
            if not entry.shape:
                out[entry.key] = results[entry.start]
                continue
            stacked = self._bkd.stack(
                results[entry.start : entry.start + entry.size()], axis=0
            )
            out[entry.key] = self._bkd.reshape(stacked, entry.shape + (npts,))
        return out


class BackendPointwiseKernel(Generic[Array]):
    """A pointwise physics evaluated at quadrature points with ``bkd``.

    Parameters
    ----------
    physics : PointwisePhysics
        The physics.
    bkd : Backend
        The backend of the inputs and outputs.

    Raises
    ------
    TypeError
        If ``physics`` is not a :class:`PointwisePhysics`.
    ValueError
        If an integrand uses a function sympy cannot differentiate or the
        backend cannot evaluate.
    """

    def __init__(self, physics: PointwisePhysics, bkd: Backend[Array]) -> None:
        self._derivatives = PointwiseDerivatives(physics)
        self._generated = generate(self._derivatives)
        self._physics, self._bkd = physics, bkd
        self._sources = self._variable_sources(physics)
        names = physics.names()
        self._residual = _BoundSet(self._generated.residual, names, bkd)
        self._jacobian = _BoundSet(self._generated.jacobian, names, bkd)
        self._param_derivatives = _BoundSet(self._generated.parameters, names, bkd)

    @staticmethod
    def _variable_sources(physics: PointwisePhysics) -> Dict[sp.Symbol, _Source]:
        sources: Dict[sp.Symbol, _Source] = {}
        for ii, symbol in enumerate(physics.coordinate_symbols()):
            sources[symbol] = ("coord", "", ii)
        sources[physics.time_symbol()] = ("time", "", 0)
        named = list(physics.field_names()) + [data.name for data in physics.aux()]
        for name in named:
            sources[physics.value_symbol(name)] = ("value", name, 0)
            for ii, symbol in enumerate(physics.gradient_symbols(name)):
                sources[symbol] = ("gradient", name, ii)
        for kk, symbol in enumerate(physics.param_symbols()):
            sources[symbol] = ("param", symbol.name, kk)
        return sources

    def bkd(self) -> Backend[Array]:
        """The backend of the inputs and outputs."""
        return self._bkd

    def physics(self) -> PointwisePhysics:
        """The physics evaluated."""
        return self._physics

    def derivatives(self) -> PointwiseDerivatives:
        """The symbolic derivatives, including which are zero."""
        return self._derivatives

    def generated(self) -> GeneratedSets:
        """The expression sets this kernel evaluates."""
        return self._generated

    def _rows(
        self, inputs: PointwiseInputs[Array], variables: Sequence[sp.Symbol]
    ) -> Tuple[Dict[sp.Symbol, Array], int]:
        ndim = self._physics.ndim()
        coords = inputs.coords
        if coords.ndim != 2 or coords.shape[0] != ndim:
            raise ValueError(
                f"coords must have shape ({ndim}, nq), got {tuple(coords.shape)}"
            )
        npts = coords.shape[1]
        zeros = self._bkd.zeros((npts,))
        rows: Dict[sp.Symbol, Array] = {}
        for symbol in variables:
            kind, name, index = self._sources[symbol]
            if kind == "coord":
                rows[symbol] = coords[index]
            elif kind == "time":
                # Time as an array too, so backend methods never see a float.
                rows[symbol] = zeros + inputs.t
            elif kind == "value":
                value = self._lookup(inputs.values, name, "value", (npts,))
                rows[symbol] = value
            elif kind == "gradient":
                grad = self._lookup(inputs.gradients, name, "gradient", (ndim, npts))
                rows[symbol] = grad[index]
            else:
                params = inputs.params
                nparams = self._physics.nparams()
                if params is None:
                    raise ValueError(f"these expressions need the parameter {name}")
                if tuple(params.shape) != (nparams,):
                    raise ValueError(
                        f"params must have shape ({nparams},), got "
                        f"{tuple(params.shape)}"
                    )
                rows[symbol] = zeros + params[index]
        return rows, npts

    @staticmethod
    def _lookup(
        table: Mapping[str, Array], name: str, what: str, shape: Tuple[int, ...]
    ) -> Array:
        if name not in table:
            raise ValueError(f"these expressions need the {what} of {name!r}")
        value = table[name]
        if tuple(value.shape) != shape:
            raise ValueError(
                f"the {what} of {name!r} must have shape {shape}, got "
                f"{tuple(value.shape)}"
            )
        return value

    def residual(
        self, inputs: PointwiseInputs[Array]
    ) -> Dict[str, PointwiseResidual[Array]]:
        """Each field's ``f0`` and ``f1`` at the points.

        Raises
        ------
        ValueError
            If an input the integrands use is missing or of the wrong
            shape.
        """
        rows, npts = self._rows(inputs, self._residual.variables())
        out = self._residual(rows, npts)
        return {
            name: PointwiseResidual(out[("f0", name)], out[("f1", name)])
            for name in self._physics.field_names()
        }

    def jacobian(
        self, inputs: PointwiseInputs[Array]
    ) -> Dict[Tuple[str, str], Optional[PointwiseJacobianBlock[Array]]]:
        """The Jacobian blocks of every (test, trial) field pair at the
        points; ``None`` for a symbolically zero pair.

        Raises
        ------
        ValueError
            If an input the derivatives use is missing or of the wrong
            shape.
        """
        rows, npts = self._rows(inputs, self._jacobian.variables())
        out = self._jacobian(rows, npts)
        blocks: Dict[Tuple[str, str], Optional[PointwiseJacobianBlock[Array]]] = {}
        fields = self._physics.field_names()
        for test in fields:
            for trial in fields:
                if self._derivatives.jacobian_block(test, trial) is None:
                    blocks[(test, trial)] = None
                    continue
                blocks[(test, trial)] = PointwiseJacobianBlock(
                    *(out.get((g, test, trial)) for g in ("g0", "g1", "g2", "g3"))
                )
        return blocks

    def parameter_derivatives(
        self, inputs: PointwiseInputs[Array]
    ) -> Dict[Tuple[str, str], Optional[PointwiseTerms[Array]]]:
        """Each field's integrands differentiated by each parameter, keyed
        ``(field, param)``; ``None`` where symbolically zero.

        Raises
        ------
        ValueError
            If an input the derivatives use is missing or of the wrong
            shape.
        """
        rows, npts = self._rows(inputs, self._param_derivatives.variables())
        out = self._param_derivatives(rows, npts)
        terms: Dict[Tuple[str, str], Optional[PointwiseTerms[Array]]] = {}
        for test in self._physics.field_names():
            for param in self._physics.params():
                if self._derivatives.parameter_derivative(test, param) is None:
                    terms[(test, param)] = None
                    continue
                terms[(test, param)] = PointwiseTerms(
                    out.get(("f0", test, param)), out.get(("f1", test, param))
                )
        return terms

    def __reduce__(
        self,
    ) -> Tuple[type, Tuple[PointwisePhysics, Backend[Array]]]:
        # The generated functions do not pickle; the physics and backend
        # do, and the kernel is rebuilt from them.
        return (type(self), (self._physics, self._bkd))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._physics!r}, bkd={type(self._bkd).__name__})"
        )
