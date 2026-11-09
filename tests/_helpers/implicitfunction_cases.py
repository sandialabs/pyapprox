r"""A small nonlinear residual with rows pinned to parameter-dependent data.

:math:`F_i(u; p) = (A u)_i + p_0 u_i^2 + p_1^2 c_i` on 3 states with
:math:`A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]` and :math:`c = (1, -2,
3)`, at :math:`p = (0.7, -0.4)` (:math:`p_0 = 0` for the affine case);
the rows :math:`\{0, 2\}` pinned to :math:`g = (p_0^2 + 1, \cdot, p_0
p_1)`. Every derivative is analytic; the data read :math:`F`'s
parameters, so one ``set_param`` moves both. A functional :math:`Q = g + r`
of a state term :math:`g = \tfrac12 u^\top D u + p_1 u_1` and a
parameter term :math:`r = \tfrac{3}{20}\|p\|^2` reads them too.
"""

from typing import Generic, List, Optional

from pyapprox.implicitfunction.functional_terms import (
    BackendScalarGradient,
    ParameterTermDerivatives,
    StateTermDerivatives,
)
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
from pyapprox.implicitfunction.steady_functional import SteadyFunctional
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    LinearizedOperator,
    RowSetProtocol,
)

QUADRATIC_PARAM = [0.7, -0.4]


class QuadraticResidual(Generic[Array]):
    r""":math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i` and its bundle:
    state-only (``order=0``), first (1) or second order (2); ``affine``
    sets :math:`p_0 = 0` and declares the Jacobian constant."""

    def __init__(
        self, bkd: Backend[Array], order: int = 2, affine: bool = False
    ) -> None:
        self.bkd = bkd
        self.states, self.params = backend_spaces(bkd, 3, 2)
        self.matrix = bkd.asarray(
            [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
        )
        self.c = bkd.asarray([[1.0], [-2.0], [3.0]])
        # p_0 = 0 makes F affine in the state.
        self.p = bkd.reshape(
            bkd.asarray([0.0, QUADRATIC_PARAM[1]] if affine else QUADRATIC_PARAM),
            (2, 1),
        )
        self._version = 0
        jacobian: ResidualJacobian[Array, DenseFormat[Array]] = (
            ResidualJacobian.constant(self._constant)
            if affine
            else ResidualJacobian.general(self._jacobian)
        )
        self._bundle: ResidualDerivatives[Array, DenseFormat[Array]]
        if order == 0:
            self._bundle = ResidualDerivatives.state_only(jacobian)
        elif order == 1:
            self._bundle = ResidualDerivatives.first_order(
                self.params, jacobian, self._param
            )
        else:
            self._bundle = ResidualDerivatives.second_order(
                self.states,
                self.params,
                jacobian,
                self._param,
                state_state=STATED_ZERO if affine else self._state_state,
                state_param=self._state_param,
                param_param=self._param_param,
            )

    def set_param(self, p: Array) -> None:
        self.p = p
        self._version += 1

    def parameters_version(self) -> int:
        return self._version

    def __call__(self, state: Array) -> Array:
        p0, p1 = self.p[0, 0], self.p[1, 0]
        return self.matrix @ state + p0 * state**2 + p1**2 * self.c

    def derivatives(self) -> ResidualDerivatives[Array, DenseFormat[Array]]:
        return self._bundle

    def dense(
        self, matrix: Array, square_params: bool = False
    ) -> BackendDenseOperator[Array]:
        """A state-to-state, parameter-to-state (two columns) or, with
        ``square_params``, parameter-to-parameter operator."""
        if square_params:
            return BackendDenseOperator(matrix, self.params.descriptor(), self.bkd)
        if matrix.shape[1] == 2:
            return BackendDenseOperator(
                matrix,
                self.params.descriptor(),
                self.bkd,
                range=self.states.descriptor(),
            )
        return BackendDenseOperator(matrix, self.states.descriptor(), self.bkd)

    def _jacobian(
        self, state: Array, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        jac = self.matrix + 2.0 * self.p[0, 0] * self.bkd.diag(state[:, 0])
        return LinearizedOperator(self.dense(jac), point)

    def _constant(
        self, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        return LinearizedOperator(self.dense(self.matrix), point)

    def _param(self, state: Array) -> DerivativeOperator[Array]:
        return self.dense(self.bkd.hstack([state**2, 2.0 * self.p[1, 0] * self.c]))

    def _state_state(self, state: Array, adj: Array) -> DerivativeOperator[Array]:
        return self.dense(2.0 * self.p[0, 0] * self.bkd.diag(adj[:, 0]))

    def _state_param(self, state: Array, adj: Array) -> DerivativeOperator[Array]:
        return self.dense(self.bkd.hstack([2.0 * state * adj, self.bkd.zeros((3, 1))]))

    def _param_param(self, state: Array, adj: Array) -> DerivativeOperator[Array]:
        entry = 2.0 * self.bkd.sum(self.c * adj)
        zero = self.bkd.zeros(())
        return self.dense(
            self.bkd.stack(
                [self.bkd.stack([zero, zero]), self.bkd.stack([zero, entry])]
            ),
            square_params=True,
        )


class QuadraticPinnedData(Generic[Array]):
    r"""Rows 0 and 2 pinned to :math:`(p_0^2 + 1, p_0 p_1)`, read from
    :math:`F`'s parameters."""

    def __init__(
        self,
        residual: QuadraticResidual[Array],
        bundle: Optional[PinnedValuesParamDerivatives[Array]] = None,
        rows: Optional[RowSetProtocol[Array]] = None,
    ) -> None:
        self._f, self.bkd = residual, residual.bkd
        self._rows = (
            BackendRowSet(residual.states.descriptor(), [0, 2], self.bkd)
            if rows is None
            else rows
        )
        self._bundle = (
            PinnedValuesParamDerivatives.second_order(self.jacobian, self.curvature)
            if bundle is None
            else bundle
        )

    def _p(self) -> List[Array]:
        return [self._f.p[0, 0], self._f.p[1, 0]]

    def rows(self) -> RowSetProtocol[Array]:
        return self._rows

    def values(self) -> Array:
        p0, p1 = self._p()
        zero = self.bkd.zeros(())
        return self.bkd.reshape(self.bkd.stack([p0**2 + 1.0, zero, p0 * p1]), (3, 1))

    def parameter_derivatives(self) -> PinnedValuesParamDerivatives[Array]:
        return self._bundle

    def jacobian(self) -> DerivativeOperator[Array]:
        p0, p1 = self._p()
        zero = self.bkd.zeros(())
        matrix = self.bkd.stack(
            [
                self.bkd.stack([2.0 * p0, zero]),
                self.bkd.stack([zero, zero]),
                self.bkd.stack([p1, p0]),
            ]
        )
        return self._f.dense(matrix)

    def curvature(self, weights: Array) -> DerivativeOperator[Array]:
        w0, w2 = weights[0, 0], weights[2, 0]
        matrix = self.bkd.stack(
            [self.bkd.stack([2.0 * w0, w2]), self.bkd.stack([w2, 0.0 * w2])]
        )
        return self._f.dense(matrix, square_params=True)


class QuadraticStateTerm(Generic[Array]):
    r""":math:`g(u; p) = \tfrac12 u^\top D u + p_1 u_1`, :math:`D =
    \mathrm{diag}(1, 2, 3)`, reading :math:`F`'s parameters; second order
    unless ``order=1``, none with ``order=0``."""

    def __init__(self, residual: QuadraticResidual[Array], order: int = 2) -> None:
        self._f, self.bkd = residual, residual.bkd
        self._d = self.bkd.asarray([[1.0], [2.0], [3.0]])
        self._bundle: StateTermDerivatives[Array, Array]
        if order == 0:
            self._bundle = StateTermDerivatives.none()
        elif order == 1:
            self._bundle = StateTermDerivatives.first_order(
                self._wrt_state, self._wrt_param
            )
        else:
            self._bundle = StateTermDerivatives.second_order(
                self._wrt_state,
                self._wrt_param,
                self._state_state,
                self._state_param,
                self._param_state,
                self._param_param,
            )

    def _unit(self, n: int, k: int, entry: Array) -> Array:
        zero = self.bkd.zeros(())
        return self.bkd.reshape(
            self.bkd.stack([entry if i == k else zero for i in range(n)]), (n, 1)
        )

    def nqoi(self) -> int:
        return 1

    def value(self, state: Array) -> Array:
        return self.bkd.reshape(
            0.5 * self.bkd.sum(self._d * state**2) + self._f.p[1, 0] * state[1, 0],
            (1, 1),
        )

    def derivatives(self) -> StateTermDerivatives[Array, Array]:
        return self._bundle

    def _wrt_state(self, state: Array) -> BackendScalarGradient[Array]:
        gradient = self._d * state + self._unit(3, 1, self._f.p[1, 0])
        return BackendScalarGradient(gradient, self.bkd)

    def _wrt_param(self, state: Array) -> BackendScalarGradient[Array]:
        return BackendScalarGradient(self._unit(2, 1, state[1, 0]), self.bkd)

    def _state_state(self, state: Array, weights: Array, x: Array) -> Array:
        return weights[0, 0] * self._d * x

    def _state_param(self, state: Array, weights: Array, x: Array) -> Array:
        return self._unit(3, 1, weights[0, 0] * x[1, 0])

    def _param_state(self, state: Array, weights: Array, x: Array) -> Array:
        return self._unit(2, 1, weights[0, 0] * x[1, 0])

    def _param_param(self, state: Array, weights: Array, x: Array) -> None:
        return None


class QuadraticParameterTerm(Generic[Array]):
    r""":math:`r(p) = \tfrac{3}{20} \|p\|^2`, reading :math:`F`'s
    parameters."""

    def __init__(self, residual: QuadraticResidual[Array]) -> None:
        self._f, self.bkd = residual, residual.bkd
        self._bundle: ParameterTermDerivatives[Array, Array] = (
            ParameterTermDerivatives.second_order(self._wrt_param, self._param_param)
        )

    def nqoi(self) -> int:
        return 1

    def value(self) -> Array:
        return self.bkd.reshape(0.15 * self.bkd.sum(self._f.p**2), (1, 1))

    def derivatives(self) -> ParameterTermDerivatives[Array, Array]:
        return self._bundle

    def _wrt_param(self) -> BackendScalarGradient[Array]:
        return BackendScalarGradient(0.3 * self._f.p, self.bkd)

    def _param_param(self, weights: Array, x: Array) -> Array:
        return weights[0, 0] * 0.3 * x


def quadratic_functional(
    residual: QuadraticResidual[Array], order: int = 2
) -> SteadyFunctional[Array, Array]:
    """:math:`Q = g + r` on :math:`F`'s parameters."""
    return SteadyFunctional.from_terms(
        residual.bkd,
        1,
        parameter_terms=(QuadraticParameterTerm(residual),),
        state_terms=(QuadraticStateTerm(residual, order),),
    )


def pinned_quadratic(
    bkd: Backend[Array],
    order: int = 2,
    affine: bool = False,
    bundle: Optional[PinnedValuesParamDerivatives[Array]] = None,
) -> "tuple[QuadraticResidual[Array], PinnedResidual[Array, DenseFormat[Array]]]":
    """:math:`F` and its pinned residual."""
    f: QuadraticResidual[Array] = QuadraticResidual(bkd, order, affine)
    return f, PinnedResidual(f, RowPin(QuadraticPinnedData(f, bundle)), f.states)
