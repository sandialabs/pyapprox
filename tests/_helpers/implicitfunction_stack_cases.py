r"""The pinned quadratic residual on any linear-algebra stack.

:math:`F_i(u; p) = (A u)_i + p_0 u_i^2 + p_1^2 c_i` on 3 states with
:math:`A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]` and :math:`c = (1, -2,
3)`, at :math:`p = (0.7, -0.4)` (:math:`p_0 = 0` for the affine case);
the rows :math:`\{0, 2\}` pinned to :math:`g = (p_0^2 + 1, \cdot, p_0
p_1)`. Every derivative is analytic, and the data read :math:`F`'s
parameters, so one ``set_param`` moves both.

The residual is a producer written against a stack case
(``TypedStackCase``): its state-space work uses only the stack's
operations (stored operators, ``from_diagonal``, the space's ``axpby``
and ``dot``; :math:`u^2 = \mathrm{diag}(u)\, u`), so it reads no state
entry. Parameters enter state arithmetic as floats, the scalars the
space's operations take, which cuts a Torch graph through :math:`p`;
the derivatives are analytic, so nothing here differentiates through it.

The one adapter is :class:`ParameterVectors`: parameter-space vectors
to and from backend arrays, which the parameter-derivative operators
need (applying one reads a parameter direction's entries; its transpose
makes a parameter vector from inner products). The operator layer has
no such access on an opaque stack; a replicated parameter space with
collective ``from_array`` / ``to_array`` would replace this adapter.
"""

from typing import Generic, List, Optional, Sequence

import numpy as np

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import (
    PinnedValuesParamDerivatives,
    RowPin,
)
from pyapprox.implicitfunction.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    ResidualDerivatives,
    ResidualJacobian,
)
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    InnerProductSpaceProtocol,
    LinearizationPoint,
    LinearizedOperator,
    MatrixFree,
    OperatorProperties,
    RowSetProtocol,
    ScaledSum,
    SpaceDescriptor,
)
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V

QUADRATIC_PARAM = [0.7, -0.4]
STATES = SpaceDescriptor("state", 3)
PARAMS = SpaceDescriptor("param", 2)


class ParameterVectors(Generic[V, F, A]):
    """The adapter: parameter vectors of a stack to and from backend
    arrays, through the case's conversions."""

    def __init__(self, stack: TypedStackCase[V, F, A]) -> None:
        self._stack = stack
        self._space = stack.space(PARAMS)

    def space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._space

    def entries(self, x: V) -> A:
        """A parameter block's entries, shape ``(nparams, ncols)``."""
        return self._stack.array(x)

    def vector(self, entries: A) -> V:
        """A parameter block holding ``entries``."""
        return self._stack.vector(entries, PARAMS)


class _Columns(Generic[V, F, A]):
    r"""Parameters to states, :math:`x \mapsto \sum_j x_j a_j` for state
    columns :math:`a_j`; matrix-free."""

    def __init__(
        self,
        columns: Sequence[V],
        states: InnerProductSpaceProtocol[V, A],
        params: ParameterVectors[V, F, A],
        stack: TypedStackCase[V, F, A],
    ) -> None:
        self._columns, self._states, self._params = list(columns), states, params
        self._bkd = stack.bkd

    def domain(self) -> SpaceDescriptor:
        return PARAMS

    def range(self) -> SpaceDescriptor:
        return STATES

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        entries = self._params.entries(x)
        blocks = []
        for k in range(self._params.space().ncols(x)):
            acc = self._states.zeros(1)
            for j, column in enumerate(self._columns):
                coefficient = self._bkd.to_float(entries[j, k])
                acc = self._states.axpby(coefficient, column, 1.0, acc)
            blocks.append(acc)
        return self._states.concatenate(blocks)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        rows = [self._states.dot(column, y) for column in self._columns]
        return self._params.vector(self._bkd.vstack(rows))


class _ParameterMatrix(Generic[V, F, A]):
    """Parameters to parameters by a small backend matrix; matrix-free."""

    def __init__(self, matrix: A, params: ParameterVectors[V, F, A]) -> None:
        self._matrix, self._params = matrix, params

    def domain(self) -> SpaceDescriptor:
        return PARAMS

    def range(self) -> SpaceDescriptor:
        return PARAMS

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._params.vector(self._matrix @ self._params.entries(x))

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        return self._params.vector(self._matrix.T @ self._params.entries(y))


class QuadraticResidual(Generic[V, F, A]):
    r""":math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i` on ``stack`` and its
    bundle: state-only (``order=0``), first (1) or second order (2);
    ``affine`` sets :math:`p_0 = 0` and declares the Jacobian constant."""

    def __init__(
        self, stack: TypedStackCase[V, F, A], order: int = 2, affine: bool = False
    ) -> None:
        bkd = stack.bkd
        self.stack, self.bkd = stack, bkd
        self.states = stack.space(STATES)
        self.parameters: ParameterVectors[V, F, A] = ParameterVectors(stack)
        self._matrix = stack.build(
            np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]), STATES
        )
        self._c = stack.vector(bkd.asarray([[1.0], [-2.0], [3.0]]), STATES)
        # p_0 = 0 makes F affine in the state.
        self.p = bkd.reshape(
            bkd.asarray([0.0, QUADRATIC_PARAM[1]] if affine else QUADRATIC_PARAM),
            (2, 1),
        )
        self._version = 0
        jacobian: ResidualJacobian[V, F] = (
            ResidualJacobian.constant(self._constant)
            if affine
            else ResidualJacobian.general(self._jacobian)
        )
        params = self.parameters.space()
        self._bundle: ResidualDerivatives[V, F]
        if order == 0:
            self._bundle = ResidualDerivatives.state_only(jacobian)
        elif order == 1:
            self._bundle = ResidualDerivatives.first_order(
                params, jacobian, self._param
            )
        else:
            self._bundle = ResidualDerivatives.second_order(
                self.states,
                params,
                jacobian,
                self._param,
                state_state=STATED_ZERO if affine else self._state_state,
                state_param=self._state_param,
                param_param=self._param_param,
            )

    def nparams(self) -> int:
        return 2

    def set_param(self, p: A) -> None:
        """Set :math:`p`, shape ``(2,)``; kept as the column ``self.p``."""
        if tuple(p.shape) != (2,):
            raise ValueError(f"p must have shape (2,), got {tuple(p.shape)}")
        self.p = self.bkd.reshape(p, (2, 1))
        self._version += 1

    def parameters_version(self) -> int:
        return self._version

    def coefficients(self) -> List[float]:
        """:math:`(p_0, p_1)` as the floats stack operations take."""
        return [self.bkd.to_float(self.p[0, 0]), self.bkd.to_float(self.p[1, 0])]

    def columns(self, columns: Sequence[V]) -> DerivativeOperator[V]:
        """The parameter-to-state operator with these state columns."""
        return _Columns(columns, self.states, self.parameters, self.stack)

    def parameter_matrix(self, matrix: A) -> DerivativeOperator[V]:
        """The parameter-to-parameter operator holding ``matrix``."""
        return _ParameterMatrix(matrix, self.parameters)

    def __call__(self, state: V) -> V:
        p0, p1 = self.coefficients()
        value = self._matrix.apply(state)
        value = self.states.axpby(p0, self._square(state), 1.0, value)
        return self.states.axpby(p1**2, self._c, 1.0, value)

    def derivatives(self) -> ResidualDerivatives[V, F]:
        return self._bundle

    def _diagonal(self, d: V) -> ActionOperatorProtocol[V, F]:
        return self.stack.from_diagonal(d, STATES)

    def _square(self, state: V) -> V:
        return self._diagonal(state).apply(state)

    def _jacobian(
        self, state: V, point: LinearizationPoint
    ) -> LinearizedOperator[V, F]:
        p0 = self.coefficients()[0]
        jac = ScaledSum(
            [(1.0, self._matrix), (2.0 * p0, self._diagonal(state))], self.states
        )
        return LinearizedOperator(jac, point)

    def _constant(self, point: LinearizationPoint) -> LinearizedOperator[V, F]:
        return LinearizedOperator(self._matrix, point)

    def _param(self, state: V) -> DerivativeOperator[V]:
        p1 = self.coefficients()[1]
        return self.columns([self._square(state), self.states.scale(2.0 * p1, self._c)])

    def _state_state(self, state: V, adj: V) -> DerivativeOperator[V]:
        p0 = self.coefficients()[0]
        return self._diagonal(self.states.scale(2.0 * p0, adj))

    def _state_param(self, state: V, adj: V) -> DerivativeOperator[V]:
        product = self.states.scale(2.0, self._diagonal(state).apply(adj))
        return self.columns([product, self.states.zeros(1)])

    def _param_param(self, state: V, adj: V) -> DerivativeOperator[V]:
        entry = 2.0 * self.states.dot(self._c, adj)
        zero = 0.0 * entry
        bkd = self.bkd
        matrix = bkd.vstack([bkd.hstack([zero, zero]), bkd.hstack([zero, entry])])
        return self.parameter_matrix(matrix)


class QuadraticPinnedData(Generic[V, F, A]):
    r"""Rows 0 and 2 pinned to :math:`(p_0^2 + 1, p_0 p_1)`, read from
    :math:`F`'s parameters."""

    def __init__(
        self,
        residual: QuadraticResidual[V, F, A],
        bundle: Optional[PinnedValuesParamDerivatives[V]] = None,
        rows: Optional[RowSetProtocol[V]] = None,
    ) -> None:
        self._f = residual
        stack, bkd = residual.stack, residual.bkd
        self._rows = stack.rows(STATES, [0, 2]) if rows is None else rows
        self._e0 = stack.vector(bkd.asarray([[1.0], [0.0], [0.0]]), STATES)
        self._e2 = stack.vector(bkd.asarray([[0.0], [0.0], [1.0]]), STATES)
        self._bundle = (
            PinnedValuesParamDerivatives.second_order(self.jacobian, self.curvature)
            if bundle is None
            else bundle
        )

    def rows(self) -> RowSetProtocol[V]:
        return self._rows

    def values(self) -> V:
        p0, p1 = self._f.coefficients()
        return self._f.states.axpby(p0**2 + 1.0, self._e0, p0 * p1, self._e2)

    def parameter_derivatives(self) -> PinnedValuesParamDerivatives[V]:
        return self._bundle

    def jacobian(self) -> DerivativeOperator[V]:
        p0, p1 = self._f.coefficients()
        states = self._f.states
        return self._f.columns(
            [states.axpby(2.0 * p0, self._e0, p1, self._e2), states.scale(p0, self._e2)]
        )

    def curvature(self, weights: V) -> DerivativeOperator[V]:
        states, bkd = self._f.states, self._f.bkd
        w0, w2 = states.dot(self._e0, weights), states.dot(self._e2, weights)
        matrix = bkd.vstack([bkd.hstack([2.0 * w0, w2]), bkd.hstack([w2, 0.0 * w2])])
        return self._f.parameter_matrix(matrix)


def pinned_quadratic(
    stack: TypedStackCase[V, F, A],
    order: int = 2,
    affine: bool = False,
    bundle: Optional[PinnedValuesParamDerivatives[V]] = None,
) -> "tuple[QuadraticResidual[V, F, A], PinnedResidual[V, F]]":
    """:math:`F` and its pinned residual."""
    f: QuadraticResidual[V, F, A] = QuadraticResidual(stack, order, affine)
    return f, PinnedResidual(f, RowPin(QuadraticPinnedData(f, bundle)), f.states)
