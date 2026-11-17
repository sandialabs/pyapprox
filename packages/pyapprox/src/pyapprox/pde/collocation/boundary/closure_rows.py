r"""A collocation system's closure rows: natural boundary rows that replace
the collocated PDE with a relation :math:`c(u, t; p) = 0`.

Unlike an essential row, which reads :math:`u_i - g_i` and so has identity
Jacobian rows and data that never see the state, a closure row is an
equation in the state (a Robin row involves the normal derivative). The
declaration does nothing to a residual; the system composing it with a
physics replaces the rows.

Every array is full length with the declaration's rows filled and the
other rows zero, so parts sharing a row (neighboring sides share corner
nodes) resolve by order: a later part wins a shared row, its value and its
derivatives.
"""

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    StatedZero,
    backend_state_space,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    BoundaryDataInput,
    ParameterizedBoundaryDataProtocol,
    component_state_rows,
)
from pyapprox.pde.collocation.physics.coefficients import (
    NodalCoefficientProtocol,
    as_nodal_coefficient,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet

ClosureParamFn = Callable[[Array, float], Array]
r""":math:`(u, t) \mapsto \partial c/\partial p` with zero rows outside the
declaration. Shape: (nstates, nparams)"""
ClosureWeightedFn = Callable[[Array, float, Array], Array]
r""":math:`(u, t, w) \mapsto \sum_i w_i\, \partial^2 c_i` for a weight
vector :math:`w` of shape (nstates,); entries outside the declaration's
rows are ignored."""


@dataclass(frozen=True)
class ClosureRowsDerivatives(Generic[Array]):
    r"""The optional derivatives of closure rows. The state Jacobian is
    always supplied, by the rows themselves.

    Every field is a function, ``STATED_ZERO`` (the rows do not depend on
    that variable that way), or None (not known here: a consumer needing
    it refuses or supplies it).

    Attributes
    ----------
    param_jacobian : ClosureParamFn, STATED_ZERO or None
        :math:`\partial c/\partial p`. Shape: (nstates, nparams)
    state_curvature : ClosureWeightedFn, STATED_ZERO or None
        :math:`\sum_i w_i\, \partial^2 c_i/\partial u^2`.
        Shape: (nstates, nstates)
    mixed : ClosureWeightedFn, STATED_ZERO or None
        :math:`\sum_i w_i\, \partial^2 c_i/\partial u\, \partial p`.
        Shape: (nstates, nparams)
    param_curvature : ClosureWeightedFn, STATED_ZERO or None
        :math:`\sum_i w_i\, \partial^2 c_i/\partial p^2`.
        Shape: (nparams, nparams)
    """

    param_jacobian: Union[ClosureParamFn[Array], StatedZero, None] = None
    state_curvature: Union[ClosureWeightedFn[Array], StatedZero, None] = None
    mixed: Union[ClosureWeightedFn[Array], StatedZero, None] = None
    param_curvature: Union[ClosureWeightedFn[Array], StatedZero, None] = None


@runtime_checkable
class ClosureRowsProtocol(Protocol, Generic[Array]):
    """Closure rows of a collocation system, as the system replacing them
    reads them."""

    def rows(self) -> BackendRowSet[Array]:
        """The replaced rows of the component-stacked state."""
        ...

    def residual(self, state: Array, time: float) -> Array:
        """:math:`c(u, t)` on the rows, zero elsewhere. Shape: (nstates,)"""
        ...

    def jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial c/\partial u` on the rows, zero elsewhere.
        Shape: (nstates, nstates)"""
        ...

    def derivatives(self) -> ClosureRowsDerivatives[Array]:
        """The rows' optional derivatives; which fields are present is
        fixed for the object's life."""
        ...

    def is_time_invariant(self) -> bool:
        """Whether the rows' Jacobian and data are time-invariant."""
        ...


@runtime_checkable
class RobinNormalTermProtocol(Protocol, Generic[Array]):
    """The normal term :math:`N(u)` of a Robin row, as :class:`RobinRows`
    reads it. The collocation normal operators satisfy it."""

    def __call__(self, state: Array, time: float) -> Array:
        """:math:`N(u)` at the part's nodes. Shape: (nboundary,)"""
        ...

    def jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial N/\partial u`. Shape: (nboundary, nstates)"""
        ...

    def has_coefficient_dependence(self) -> bool:
        """Whether :math:`N` reads a physics coefficient, through which it
        may depend on the parameters."""
        ...

    def is_linear_in_state(self) -> bool:
        """Whether :math:`N` is declared linear in the state, so its second
        state derivative is zero. False means not known, not nonlinear."""
        ...


@dataclass(frozen=True)
class RobinPart(Generic[Array]):
    r"""One part of a Robin declaration:
    :math:`\alpha u + \beta N(u) = g` for a state component on a set of
    boundary nodes. Neumann is :math:`\alpha = 0`.

    Attributes
    ----------
    boundary_indices : Array
        Mesh node indices of the part. Shape: (nboundary,)
    normal_term : RobinNormalTermProtocol
        :math:`N(u)` at those nodes.
    alpha, beta, data : float, NodalCoefficientProtocol, callable or
        ParameterizedBoundaryDataProtocol
        :math:`\alpha`, :math:`\beta` and :math:`g` at those nodes, in the
        forms ``DirichletPart.data`` takes; each may depend on the
        parameters.
    component : int
        The state component in the :math:`\alpha u` term (0 for a scalar
        physics).
    """

    boundary_indices: Array
    normal_term: RobinNormalTermProtocol[Array]
    alpha: BoundaryDataInput[Array]
    beta: BoundaryDataInput[Array]
    data: BoundaryDataInput[Array]
    component: int = 0


class _PartCoefficient(Generic[Array]):
    """One of a Robin part's coefficients, and its parameter dependence if
    it has one."""

    def __init__(
        self, value: BoundaryDataInput[Array], nboundary: int, bkd: Backend[Array]
    ) -> None:
        self.parameterized: Optional[ParameterizedBoundaryDataProtocol[Array]]
        self.nodal: NodalCoefficientProtocol[Array]
        if isinstance(value, ParameterizedBoundaryDataProtocol):
            self.parameterized = value
            self.nodal = value
        else:
            self.parameterized = None
            self.nodal = as_nodal_coefficient(value, nboundary, bkd)
        self._nboundary = nboundary

    def values(self, time: float) -> Array:
        values = self.nodal.values(time)
        if values.shape != (self._nboundary,):
            raise ValueError(
                f"Robin coefficients must have shape ({self._nboundary},), got "
                f"{tuple(values.shape)}"
            )
        return values


class _RobinPartRows(Generic[Array]):
    def __init__(
        self,
        part: RobinPart[Array],
        rows: List[int],
        nstates: int,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(part.normal_term, RobinNormalTermProtocol):
            raise TypeError(
                "normal_term must satisfy RobinNormalTermProtocol, got "
                f"{type(part.normal_term).__name__}"
            )
        nboundary = len(rows)
        self.rows = bkd.asarray(rows, dtype=bkd.int64_dtype())
        self.normal_term = part.normal_term
        self.alpha = _PartCoefficient(part.alpha, nboundary, bkd)
        self.beta = _PartCoefficient(part.beta, nboundary, bkd)
        self.data = _PartCoefficient(part.data, nboundary, bkd)
        # Selects the part's state entries: (nboundary, nstates)
        self.selection = bkd.eye(nstates)[self.rows]

    def coefficients(self) -> Tuple[_PartCoefficient[Array], ...]:
        return (self.alpha, self.beta, self.data)

    def has_curvature(self) -> bool:
        """Whether every parameterized coefficient supplies its curvature."""
        return all(
            c.parameterized.param_derivatives().curvature is not None
            for c in self.coefficients()
            if c.parameterized is not None
        )


class _RobinParamJacobian(Generic[Array]):
    def __init__(self, rows: "RobinRows[Array]") -> None:
        self._rows = rows

    def __call__(self, state: Array, time: float) -> Array:
        return self._rows._param_jacobian(state, time)


class _RobinMixed(Generic[Array]):
    def __init__(self, rows: "RobinRows[Array]") -> None:
        self._rows = rows

    def __call__(self, state: Array, time: float, weights: Array) -> Array:
        return self._rows._mixed(state, time, weights)


class _RobinParamCurvature(Generic[Array]):
    def __init__(self, rows: "RobinRows[Array]") -> None:
        self._rows = rows

    def __call__(self, state: Array, time: float, weights: Array) -> Array:
        return self._rows._param_curvature(state, time, weights)


class RobinRows(Generic[Array]):
    r"""Robin closure rows :math:`c = \alpha u + \beta N(u) - g`, as
    :class:`ClosureRowsProtocol` declares them.

    :math:`\partial c/\partial p
    = \operatorname{diag}(u)\,\alpha_p + \operatorname{diag}(N(u))\,\beta_p
    - g_p` on each part's rows, from the coefficients that depend on the
    parameters. It is stated zero when nothing does, and not available
    (None) when a part's normal term reads a physics coefficient: the
    coefficient's dependence on the parameters is not known here, so the
    system supplies that term. The second derivatives follow
    :meth:`derivatives`; a weight on a shared row counts only for the part
    supplying it.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    npts : int
        The number of collocation nodes.
    ncomponents : int
        The number of state components.
    parts : sequence of RobinPart
        The parts, in the order that settles shared rows.

    Raises
    ------
    ValueError
        If there are no parts, a component or boundary index is out of
        range, or parameterized coefficients disagree on the number of
        parameters.
    TypeError
        If a part's normal term does not satisfy
        :class:`RobinNormalTermProtocol`.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        npts: int,
        ncomponents: int,
        parts: Sequence[RobinPart[Array]],
    ) -> None:
        if not parts:
            raise ValueError("a Robin declaration needs at least one part")
        self._bkd = bkd
        self._nstates = npts * ncomponents
        row_lists = [
            component_state_rows(
                bkd, part.boundary_indices, part.component, npts, ncomponents
            )
            for part in parts
        ]
        self._parts = [
            _RobinPartRows(part, rows, self._nstates, bkd)
            for part, rows in zip(parts, row_lists)
        ]
        counts = {
            coefficient.parameterized.nparams()
            for part in self._parts
            for coefficient in part.coefficients()
            if coefficient.parameterized is not None
        }
        if len(counts) > 1:
            raise ValueError(
                "parameterized Robin coefficients disagree on the number of "
                f"parameters: {sorted(counts)}"
            )
        self._nparams: Optional[int] = counts.pop() if counts else None
        # 1 where the part supplies the row, 0 where a later part does
        self._wins = [
            bkd.asarray(
                [
                    0.0 if any(row in later for later in row_lists[k + 1 :]) else 1.0
                    for row in rows
                ]
            )
            for k, rows in enumerate(row_lists)
        ]
        self._rows = BackendRowSet(
            backend_state_space(bkd, self._nstates).descriptor(),
            [row for rows in row_lists for row in rows],
            bkd,
        )

    def rows(self) -> BackendRowSet[Array]:
        """The replaced rows of the component-stacked state."""
        return self._rows

    def _check_state(self, state: Array) -> None:
        if state.shape != (self._nstates,):
            raise ValueError(
                f"state must have shape ({self._nstates},), got {tuple(state.shape)}"
            )

    def residual(self, state: Array, time: float) -> Array:
        r""":math:`\alpha u + \beta N(u) - g` on the rows, zero elsewhere.
        Shape: (nstates,)"""
        self._check_state(state)
        bkd = self._bkd
        column = bkd.zeros((self._nstates,))
        for part in self._parts:
            values = (
                part.alpha.values(time) * state[part.rows]
                + part.beta.values(time) * part.normal_term(state, time)
                - part.data.values(time)
            )
            column = bkd.index_update(column, part.rows, values)
        return column

    def jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\operatorname{diag}(\alpha) E + \operatorname{diag}(\beta)
        N_u` on the rows, zero elsewhere, with :math:`E` selecting the
        part's state entries. Shape: (nstates, nstates)

        Raises
        ------
        ValueError
            If a normal term's Jacobian is not (nboundary, nstates).
        """
        self._check_state(state)
        bkd = self._bkd
        matrix = bkd.zeros((self._nstates, self._nstates))
        for part in self._parts:
            normal_jacobian = part.normal_term.jacobian(state, time)
            if normal_jacobian.shape != part.selection.shape:
                raise ValueError(
                    "the normal term's Jacobian must have shape "
                    f"{tuple(part.selection.shape)}, got "
                    f"{tuple(normal_jacobian.shape)}"
                )
            block = (
                part.alpha.values(time)[:, None] * part.selection
                + part.beta.values(time)[:, None] * normal_jacobian
            )
            matrix = bkd.index_update(matrix, part.rows, block)
        return matrix

    def derivatives(self) -> ClosureRowsDerivatives[Array]:
        r"""The optional derivatives:

        - ``state_curvature``: stated zero when every normal term is
          declared linear in the state, else None;
        - when a part's normal term reads a physics coefficient, every
          parameter derivative is None (the system supplies them);
        - otherwise ``param_jacobian`` and ``param_curvature`` are stated
          zero when nothing depends on the parameters; ``mixed`` is
          stated zero when neither :math:`\alpha` nor :math:`\beta` does
          (:math:`g` does not enter it); ``param_curvature`` is None when
          a parameterized coefficient supplies no curvature.
        """
        state_curvature: Union[ClosureWeightedFn[Array], StatedZero, None] = (
            STATED_ZERO
            if all(part.normal_term.is_linear_in_state() for part in self._parts)
            else None
        )
        if any(part.normal_term.has_coefficient_dependence() for part in self._parts):
            return ClosureRowsDerivatives(state_curvature=state_curvature)
        if self._nparams is None:
            return ClosureRowsDerivatives(
                STATED_ZERO, state_curvature, STATED_ZERO, STATED_ZERO
            )
        coefficients_in_mixed = any(
            c.parameterized is not None
            for part in self._parts
            for c in (part.alpha, part.beta)
        )
        return ClosureRowsDerivatives(
            _RobinParamJacobian(self),
            state_curvature,
            _RobinMixed(self) if coefficients_in_mixed else STATED_ZERO,
            (
                _RobinParamCurvature(self)
                if all(part.has_curvature() for part in self._parts)
                else None
            ),
        )

    def _part_weights(self, position: int, weights: Array) -> Array:
        """The part's weights on the rows it supplies."""
        part = self._parts[position]
        return weights[part.rows] * self._wins[position]

    def _check_weights(self, weights: Array) -> None:
        if weights.shape != (self._nstates,):
            raise ValueError(
                f"weights must have shape ({self._nstates},), got "
                f"{tuple(weights.shape)}"
            )

    def _mixed(self, state: Array, time: float, weights: Array) -> Array:
        r""":math:`\sum_i w_i\, \partial^2 c_i/\partial u\, \partial p
        = E^\top \operatorname{diag}(w)\, \alpha_p
        + N_u^\top \operatorname{diag}(w)\, \beta_p`, per part on the rows
        it supplies."""
        self._check_state(state)
        self._check_weights(weights)
        nparams = self._require_nparams()
        bkd = self._bkd
        matrix = bkd.zeros((self._nstates, nparams))
        for position, part in enumerate(self._parts):
            part_weights = self._part_weights(position, weights)[:, None]
            factors = (part.selection, part.normal_term.jacobian(state, time))
            for coefficient, factor in zip((part.alpha, part.beta), factors):
                if coefficient.parameterized is not None:
                    jacobian = coefficient.parameterized.param_derivatives().jacobian(
                        time
                    )
                    matrix = matrix + factor.T @ (part_weights * jacobian)
        return matrix

    def _param_curvature(self, state: Array, time: float, weights: Array) -> Array:
        r""":math:`\sum_i w_i\, \partial^2 c_i/\partial p^2`, the curvatures
        of :math:`\alpha`, :math:`\beta` and :math:`g` weighted by
        :math:`w u`, :math:`w N(u)` and :math:`-w`, per part on the rows
        it supplies."""
        self._check_state(state)
        self._check_weights(weights)
        nparams = self._require_nparams()
        bkd = self._bkd
        matrix = bkd.zeros((nparams, nparams))
        for position, part in enumerate(self._parts):
            part_weights = self._part_weights(position, weights)
            factors = (
                state[part.rows],
                part.normal_term(state, time),
                -bkd.ones((part.rows.shape[0],)),
            )
            for coefficient, factor in zip(part.coefficients(), factors):
                if coefficient.parameterized is None:
                    continue
                curvature = coefficient.parameterized.param_derivatives().curvature
                if curvature is None:
                    raise RuntimeError("a parameterized coefficient has no curvature")
                matrix = matrix + curvature(time, part_weights * factor)
        return matrix

    def _require_nparams(self) -> int:
        if self._nparams is None:
            raise RuntimeError("no Robin coefficient depends on the parameters")
        return self._nparams

    def _param_jacobian(self, state: Array, time: float) -> Array:
        self._check_state(state)
        bkd, nparams = self._bkd, self._require_nparams()
        matrix = bkd.zeros((self._nstates, nparams))
        for part in self._parts:
            block = bkd.zeros((part.rows.shape[0], nparams))
            factors = (
                state[part.rows],
                part.normal_term(state, time),
                -bkd.ones((part.rows.shape[0],)),
            )
            for coefficient, factor in zip(part.coefficients(), factors):
                if coefficient.parameterized is not None:
                    jacobian = coefficient.parameterized.param_derivatives().jacobian(
                        time
                    )
                    block = block + factor[:, None] * jacobian
            # Every part writes its rows, so a later part wins a shared row
            matrix = bkd.index_update(matrix, part.rows, block)
        return matrix

    def is_time_invariant(self) -> bool:
        """Whether every coefficient is declared time-invariant and no
        normal term reads a physics coefficient (which may change with
        time)."""
        return not any(
            part.normal_term.has_coefficient_dependence()
            or any(c.nodal.is_time_dependent() for c in part.coefficients())
            for part in self._parts
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nparts={len(self._parts)}, "
            f"nrows={self._rows.count()})"
        )
