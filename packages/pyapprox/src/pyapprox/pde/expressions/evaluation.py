"""PDE expressions at points, on a backend.

A convenience for stacks built on :class:`~pyapprox.util.backends.
protocols.Backend`; nothing symbolic requires it. Other stacks bind the
same symbols with their own tools (``sympy.lambdify`` to another module,
``sympy.ccode``, expression strings for a runtime parser).

Points are ``(ndim, npts)``; a stack with another layout reshapes at its
own boundary. Only the symbols the expressions use are passed to the
backend evaluator, so declaring many jets costs nothing per call.
"""

from typing import Dict, Generic, Mapping, Optional, Sequence

import sympy as sp

from pyapprox.pde.expressions.symbols import coordinates, normal, time
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.expressions.evaluation import ExpressionEvaluator
from pyapprox.util.expressions.names import Names
from pyapprox.util.expressions.parsing import Value, parse


class PointExpressions(Generic[Array]):
    """PDE expressions evaluated at points with ``bkd``.

    Parameters
    ----------
    expressions : Sequence
        The expressions, as text or sympy.
    names : Names
        What they may use, from
        :func:`~pyapprox.pde.expressions.symbols.pde_names`.
    ndim : int
        The spatial dimension.
    bkd : Backend
        The backend of the points and values.

    Raises
    ------
    ValueError
        If an expression fails to parse or uses a function the backend
        cannot evaluate, or ``names`` lacks the coordinates, time or
        normals of ``ndim`` dimensions.
    """

    def __init__(
        self,
        expressions: Sequence[Value],
        names: Names,
        ndim: int,
        bkd: Backend[Array],
    ) -> None:
        self._coordinates = coordinates(ndim)
        self._normal = normal(ndim)
        declared = set(names.variables())
        geometry = set(self._coordinates) | set(self._normal) | {time()}
        if not geometry <= declared:
            raise ValueError(
                f"names must declare the coordinates, t and normals of {ndim} "
                "dimensions; build them with pde_names"
            )
        parsed = [parse(expression, names) for expression in expressions]
        used: "set[sp.Basic]" = set()
        for expression in parsed:
            used |= expression.free_symbols
        self._variables = [s for s in names.variables() if s in used]
        self._states = [s for s in self._variables if s not in geometry]
        constants = {s.name: v for s, v in names.constants().items()}
        self._evaluator = ExpressionEvaluator(
            parsed,
            Names([s.name for s in self._variables], constants, names.functions()),
            bkd,
        )
        self._ndim, self._bkd = ndim, bkd

    def nexpressions(self) -> int:
        """The number of expressions."""
        return self._evaluator.nexpressions()

    def bkd(self) -> Backend[Array]:
        """The backend of the points and values."""
        return self._bkd

    def needs_normal(self) -> bool:
        """Whether an expression uses the outward normal."""
        return any(s in self._variables for s in self._normal)

    def states(self) -> Sequence[str]:
        """The field jets the expressions use, by name."""
        return [s.name for s in self._states]

    def __call__(
        self,
        coords: Array,
        normals: Optional[Array] = None,
        t: float = 0.0,
        state: Optional[Mapping[str, Array]] = None,
    ) -> Array:
        """The expressions at points.

        Parameters
        ----------
        coords : Array
            Shape ``(ndim, npts)``.
        normals : Array, optional
            Outward unit normals, shape ``(ndim, npts)``; required when an
            expression uses them.
        t : float
            The time.
        state : Mapping[str, Array], optional
            Each jet in :meth:`states` at the points, shape ``(npts,)``.

        Returns
        -------
        Array
            Shape ``(nexpressions, npts)``.

        Raises
        ------
        ValueError
            If ``coords`` is not ``(ndim, npts)``, or ``normals`` or a jet
            is needed and missing or of the wrong shape.
        """
        if coords.ndim != 2 or coords.shape[0] != self._ndim:
            raise ValueError(
                f"coords must have shape ({self._ndim}, npts), got "
                f"{tuple(coords.shape)}"
            )
        npts = coords.shape[1]
        rows: Dict[sp.Symbol, Array] = {
            x: coords[ii] for ii, x in enumerate(self._coordinates)
        }
        # Time as an array too, so backend methods never see a bare float.
        rows[time()] = self._bkd.zeros((npts,)) + t
        if self.needs_normal():
            if normals is None:
                raise ValueError("these expressions need the outward normals")
            if tuple(normals.shape) != tuple(coords.shape):
                raise ValueError(
                    f"normals must have shape {tuple(coords.shape)}, got "
                    f"{tuple(normals.shape)}"
                )
            rows.update({n: normals[ii] for ii, n in enumerate(self._normal)})
        state = {} if state is None else state
        for symbol in self._states:
            if symbol.name not in state:
                raise ValueError(f"these expressions need the state {symbol.name}")
            value = state[symbol.name]
            if tuple(value.shape) != (npts,):
                raise ValueError(
                    f"{symbol.name} must have shape ({npts},), got {tuple(value.shape)}"
                )
            rows[symbol] = value
        if not self._variables:
            values = self._bkd.zeros((0, npts))
        else:
            values = self._bkd.stack([rows[s] for s in self._variables], axis=0)
        return self._evaluator(values)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nexpressions={self.nexpressions()}, "
            f"ndim={self._ndim})"
        )
