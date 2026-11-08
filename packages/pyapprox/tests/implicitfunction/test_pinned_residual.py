r"""A residual with rows pinned to data, :math:`R = P F + (I - P)(u - g)`.

:math:`F_i(u; p) = (A u)_i + p_0 u_i^2 + p_1^2 c_i` on 3 states with
:math:`A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]`, :math:`c = (1, -2, 3)`,
at :math:`p = (0.7, -0.4)` and :math:`u = (0.3, -0.8, 1.1)`; the rows
:math:`\{0, 2\}` pinned to :math:`g = (p_0^2 + 1, \cdot, p_0 p_1)`:

- Value: the pinned rows read :math:`u_R - g_R`, the other row is
  :math:`F`'s.
- Every derivative passes finite differences (``DerivativeChecker``,
  central, relative, smallest error below 1e-6): the state and parameter
  Jacobians of :math:`R`, and the adjoint-weighted second derivatives as
  the Jacobians of :math:`R_u^\top \lambda` in :math:`u` and :math:`p` and
  of :math:`R_p^\top \lambda` in :math:`p`.
- Wrong-model control: the same data declared independent of the
  parameters give a parameter Jacobian whose error stays above 1e-2.
- Levels and zeros: a constant :math:`F_u` (``p_0 = 0`` and declared
  affine) stays constant with the stated-zero state curvature; data
  without second derivatives give a first-order bundle; a state-only
  :math:`F` gives a state-only bundle.
- Refusals: data whose parameter Jacobian is unavailable, a residual, pin
  or space of the wrong type, and rows of another space.
- The parameter version is :math:`F`'s.
"""

from typing import Callable, Generic, List, Optional

import pytest

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
    is_stated_zero,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
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

_POINT = LinearizationPoint(0, ("test",))
_PARAM = [0.7, -0.4]
_STATE = [[0.3], [-0.8], [1.1]]
_ADJ = [[0.9], [-0.5], [0.6]]
_FD_TOL = 1e-6


class _Quadratic(Generic[Array]):
    r""":math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i` and its bundle."""

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
            bkd.asarray([0.0, _PARAM[1]] if affine else _PARAM), (2, 1)
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

    def _dense(
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
        return LinearizedOperator(self._dense(jac), point)

    def _constant(
        self, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        return LinearizedOperator(self._dense(self.matrix), point)

    def _param(self, state: Array) -> DerivativeOperator[Array]:
        return self._dense(self.bkd.hstack([state**2, 2.0 * self.p[1, 0] * self.c]))

    def _state_state(self, state: Array, adj: Array) -> DerivativeOperator[Array]:
        return self._dense(2.0 * self.p[0, 0] * self.bkd.diag(adj[:, 0]))

    def _state_param(self, state: Array, adj: Array) -> DerivativeOperator[Array]:
        return self._dense(self.bkd.hstack([2.0 * state * adj, self.bkd.zeros((3, 1))]))

    def _param_param(self, state: Array, adj: Array) -> DerivativeOperator[Array]:
        entry = 2.0 * self.bkd.sum(self.c * adj)
        zero = self.bkd.zeros(())
        return self._dense(
            self.bkd.stack(
                [self.bkd.stack([zero, zero]), self.bkd.stack([zero, entry])]
            ),
            square_params=True,
        )


class _Data(Generic[Array]):
    r"""Rows 0 and 2 pinned to :math:`(p_0^2 + 1, p_0 p_1)`, read from
    :math:`F`'s parameters."""

    def __init__(
        self,
        residual: _Quadratic[Array],
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
        return self._f._dense(matrix)

    def curvature(self, weights: Array) -> DerivativeOperator[Array]:
        w0, w2 = weights[0, 0], weights[2, 0]
        matrix = self.bkd.stack(
            [self.bkd.stack([2.0 * w0, w2]), self.bkd.stack([w2, 0.0 * w2])]
        )
        return self._f._dense(matrix, square_params=True)


def _pinned(
    bkd: Backend[Array],
    order: int = 2,
    affine: bool = False,
    bundle: Optional[PinnedValuesParamDerivatives[Array]] = None,
) -> "tuple[_Quadratic[Array], PinnedResidual[Array, DenseFormat[Array]]]":
    f: _Quadratic[Array] = _Quadratic(bkd, order, affine)
    return f, PinnedResidual(f, RowPin(_Data(f, bundle)), f.states)


def _min_error(
    bkd: Backend[Array],
    nqoi: int,
    nvars: int,
    fun: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    sample: Array,
) -> float:
    errors = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=nqoi, nvars=nvars, fun=fun, jacobian=jacobian, bkd=bkd
        )
    ).check_derivatives(sample, relative=True, central=True)[0]
    return bkd.to_float(bkd.min(errors))


def _dense(op: DerivativeOperator[Array], ncols: int, bkd: Backend[Array]) -> Array:
    return bkd.hstack([op.apply(bkd.eye(ncols)[:, j : j + 1]) for j in range(ncols)])


class TestTheValue:
    def test_pinned_rows_read_the_data(self, bkd: Backend[Array]) -> None:
        f, pinned = _pinned(bkd)
        state = bkd.asarray(_STATE)
        p0, p1 = _PARAM
        value = pinned(state)
        bkd.assert_allclose(value[0, 0], bkd.asarray(0.3 - (p0**2 + 1.0)))
        bkd.assert_allclose(value[2, 0], bkd.asarray(1.1 - p0 * p1))
        bkd.assert_allclose(value[1, 0], f(state)[1, 0])
        assert pinned.parameters_version() == 0
        f.set_param(bkd.reshape(bkd.asarray(_PARAM), (2, 1)))
        assert pinned.parameters_version() == 1


class TestTheDerivatives:
    """Finite differences of the value and of the first derivatives."""

    def _at_param(
        self, f: _Quadratic[Array], fn: Callable[[], Array]
    ) -> Callable[[Array], Array]:
        def at(p: Array) -> Array:
            saved = f.p
            f.p = p
            try:
                return fn()
            finally:
                f.p = saved

        return at

    def test_the_state_jacobian(self, bkd: Backend[Array]) -> None:
        _, pinned = _pinned(bkd)
        jacobian = pinned.derivatives().jacobian
        assert jacobian is not None and not jacobian.affine()
        error = _min_error(
            bkd,
            3,
            3,
            pinned,
            lambda u: _dense(jacobian(u, _POINT), 3, bkd),
            bkd.asarray(_STATE),
        )
        assert error < _FD_TOL

    def test_the_parameter_jacobian(self, bkd: Backend[Array]) -> None:
        f, pinned = _pinned(bkd)
        param = pinned.derivatives().param
        assert param is not None
        state = bkd.asarray(_STATE)
        error = _min_error(
            bkd,
            3,
            2,
            self._at_param(f, lambda: pinned(state)),
            self._at_param(f, lambda: _dense(param(state), 2, bkd)),
            f.p,
        )
        assert error < _FD_TOL

    def test_the_second_derivatives(self, bkd: Backend[Array]) -> None:
        f, pinned = _pinned(bkd)
        bundle = pinned.derivatives()
        jacobian, param = bundle.jacobian, bundle.param
        state_state, state_param = bundle.state_state, bundle.state_param
        param_param = bundle.param_param
        assert jacobian is not None and param is not None
        assert state_state is not None and state_param is not None
        assert param_param is not None
        state, adj = bkd.asarray(_STATE), bkd.asarray(_ADJ)

        def ju(u: Array) -> Array:
            return jacobian(u, _POINT).apply_transpose(adj)

        def jp(u: Array) -> Array:
            return param(u).apply_transpose(adj)

        errors = [
            _min_error(
                bkd, 3, 3, ju, lambda u: _dense(state_state(u, adj), 3, bkd), state
            ),
            _min_error(
                bkd,
                3,
                2,
                self._at_param(f, lambda: ju(state)),
                self._at_param(f, lambda: _dense(state_param(state, adj), 2, bkd)),
                f.p,
            ),
            _min_error(
                bkd,
                2,
                2,
                self._at_param(f, lambda: jp(state)),
                self._at_param(f, lambda: _dense(param_param(state, adj), 2, bkd)),
                f.p,
            ),
        ]
        assert max(errors) < _FD_TOL

    def test_wrong_model_control(self, bkd: Backend[Array]) -> None:
        """Data declared independent of the parameters miss
        :math:`-(I - P) g_p`."""
        f, pinned = _pinned(bkd, bundle=PinnedValuesParamDerivatives.independent())
        param = pinned.derivatives().param
        assert param is not None
        state = bkd.asarray(_STATE)
        error = _min_error(
            bkd,
            3,
            2,
            self._at_param(f, lambda: pinned(state)),
            self._at_param(f, lambda: _dense(param(state), 2, bkd)),
            f.p,
        )
        assert error > 1e-2


class TestLevelsAndZeros:
    def test_a_constant_jacobian_stays_constant(self, bkd: Backend[Array]) -> None:
        _, pinned = _pinned(bkd, affine=True)
        bundle = pinned.derivatives()
        assert bundle.jacobian is not None and bundle.jacobian.affine()
        assert is_stated_zero(bundle.state_state)
        assert not is_stated_zero(bundle.state_param)
        expected = bkd.asarray([[1.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 1.0]])
        bkd.assert_allclose(
            _dense(bundle.jacobian(bkd.asarray(_STATE), _POINT), 3, bkd), expected
        )

    def test_data_without_curvature_give_first_order(self, bkd: Backend[Array]) -> None:
        f: _Quadratic[Array] = _Quadratic(bkd)
        data = _Data(f)
        first = PinnedValuesParamDerivatives.first_order(data.jacobian)
        pinned = PinnedResidual(f, RowPin(_Data(f, first)), f.states)
        bundle = pinned.derivatives()
        assert bundle.param is not None and bundle.state_state is None

    def test_a_state_only_residual_stays_state_only(self, bkd: Backend[Array]) -> None:
        _, pinned = _pinned(bkd, order=0)
        bundle = pinned.derivatives()
        assert bundle.jacobian is not None and bundle.param is None


class TestRefusals:
    def test_unavailable_data_jacobian(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="does not declare the parameter"):
            _pinned(bkd, bundle=PinnedValuesParamDerivatives.unavailable())

    def test_wrong_types_and_spaces(self, bkd: Backend[Array]) -> None:
        f: _Quadratic[Array] = _Quadratic(bkd)
        pin = RowPin(_Data(f))
        with pytest.raises(TypeError, match="ResidualProtocol"):
            PinnedResidual(3.0, pin, f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="RowPin"):
            PinnedResidual(f, 3.0, f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="VectorSpaceProtocol"):
            PinnedResidual(f, pin, 3.0)  # type: ignore[arg-type]
        other, _ = backend_spaces(bkd, 4, 2)
        rows = BackendRowSet(other.descriptor(), [0, 2], bkd)
        with pytest.raises(ValueError):
            PinnedResidual(f, RowPin(_Data(f, rows=rows)), f.states)
