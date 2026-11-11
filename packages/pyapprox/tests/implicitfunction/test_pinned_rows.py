r"""Rows of a time-free residual pinned to data, :math:`u_R = g(p)`.

A 3-state residual with 2 parameters, the rows :math:`R = \{0, 2\}`
pinned to :math:`g = (5, 99, 7)` (row 1 ignored), whose parameter
Jacobian is :math:`G = [[1, 2], [3, 4], [5, 6]]` and whose weighted
curvature is :math:`w \mapsto [[w_0, w_1], [w_1, w_2]]`:

- The row algebra: the operator :math:`A = [[1, 2, 3], [4, 5, 6], [7, 8,
  9]]` gets identity rows :math:`R` as a block's own operator, zero rows
  :math:`R` as a coupling, in both actions; an adjoint block loses its
  rows :math:`R` without the input changing.
- The data: a residual block's rows :math:`R` read :math:`u_R - g_R` and a
  right-hand side's read :math:`g_R`, the other rows unchanged; the data's
  parameter Jacobian on the rows is :math:`(I - P) G` in both actions, and
  the curvature sees the weights zeroed outside :math:`R`.
- Stated zeros give zero operators between the right spaces; an
  unavailable derivative gives None, which ``require_data_derivative``
  refuses, naming the declaration; first-order data have no curvature; a
  declaration not satisfying the protocol is refused.
"""

from typing import Generic, List, Optional

import pytest

from pyapprox.implicitfunction.pinned_rows import (
    PinnedValuesParamDerivatives,
    RowPin,
    RowReplacement,
    require_data_derivative,
)
from pyapprox.implicitfunction.residual_derivatives import DerivativeOperator
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator, BackendRowSet
from pyapprox.util.linalg.operators import RowSetProtocol


class _Declaration(Generic[Array]):
    """The rows 0 and 2 of a 3-state space pinned to (5, 7)."""

    def __init__(
        self,
        bkd: Backend[Array],
        bundle: Optional[PinnedValuesParamDerivatives[Array]] = None,
    ) -> None:
        self.bkd = bkd
        self.states, self.params = backend_spaces(bkd, 3, 2)
        self._rows = BackendRowSet(self.states.descriptor(), [0, 2], bkd)
        self.seen: List[Array] = []
        self._bundle = (
            PinnedValuesParamDerivatives.second_order(self.jacobian, self.curvature)
            if bundle is None
            else bundle
        )

    def rows(self) -> RowSetProtocol[Array]:
        return self._rows

    def values(self) -> Array:
        return self.bkd.asarray([[5.0], [99.0], [7.0]])

    def parameter_derivatives(self) -> PinnedValuesParamDerivatives[Array]:
        return self._bundle

    def jacobian(self) -> DerivativeOperator[Array]:
        return BackendDenseOperator(
            self.bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            self.params.descriptor(),
            self.bkd,
            range=self.states.descriptor(),
        )

    def curvature(self, weights: Array) -> DerivativeOperator[Array]:
        self.seen.append(weights)
        w = [self.bkd.to_float(weights[i, 0]) for i in range(3)]
        return BackendDenseOperator(
            self.bkd.asarray([[w[0], w[1]], [w[1], w[2]]]),
            self.params.descriptor(),
            self.bkd,
        )


def _matrix(decl: _Declaration[Array]) -> BackendDenseOperator[Array]:
    return BackendDenseOperator(
        decl.bkd.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        decl.states.descriptor(),
        decl.bkd,
    )


def _dense(op: DerivativeOperator[Array], ncols: int, bkd: Backend[Array]) -> Array:
    """The operator's matrix, column by column."""
    return bkd.hstack([op.apply(bkd.eye(ncols)[:, j : j + 1]) for j in range(ncols)])


class TestTheRowAlgebra:
    def test_own_rows_become_identity_rows(self, bkd: Backend[Array]) -> None:
        decl: _Declaration[Array] = _Declaration(bkd)
        own = RowReplacement(decl.rows()).own(_matrix(decl))
        expected = bkd.asarray([[1.0, 0.0, 0.0], [4.0, 5.0, 6.0], [0.0, 0.0, 1.0]])
        bkd.assert_allclose(_dense(own, 3, bkd), expected)
        y = bkd.asarray([[1.0], [-2.0], [0.5]])
        bkd.assert_allclose(own.apply_transpose(y), expected.T @ y)

    def test_couplings_lose_the_rows(self, bkd: Backend[Array]) -> None:
        decl: _Declaration[Array] = _Declaration(bkd)
        coupled = RowReplacement(decl.rows()).coupled(_matrix(decl))
        expected = bkd.asarray([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])
        bkd.assert_allclose(_dense(coupled, 3, bkd), expected)
        y = bkd.asarray([[1.0], [-2.0], [0.5]])
        bkd.assert_allclose(coupled.apply_transpose(y), expected.T @ y)

    def test_an_adjoint_loses_the_rows_in_a_copy(self, bkd: Backend[Array]) -> None:
        decl: _Declaration[Array] = _Declaration(bkd)
        block = bkd.asarray([[1.0], [2.0], [3.0]])
        zeroed = RowReplacement(decl.rows()).adjoint(block, decl.states)
        bkd.assert_allclose(zeroed, bkd.asarray([[0.0], [2.0], [0.0]]))
        bkd.assert_allclose(block, bkd.asarray([[1.0], [2.0], [3.0]]))
        assert repr(RowReplacement(decl.rows())) == "RowReplacement(nrows=2)"


class TestTheData:
    def test_the_rows_read_the_data(self, bkd: Backend[Array]) -> None:
        decl: _Declaration[Array] = _Declaration(bkd)
        pin = RowPin(decl)
        ones = bkd.ones((3, 1))
        state = bkd.asarray([[2.0], [3.0], [4.0]])
        bkd.assert_allclose(
            pin.residual(ones, state, decl.states),
            bkd.asarray([[-3.0], [1.0], [-3.0]]),
        )
        bkd.assert_allclose(
            pin.right_hand_side(ones), bkd.asarray([[5.0], [1.0], [7.0]])
        )
        bkd.assert_allclose(ones, bkd.ones((3, 1)))
        assert repr(pin) == "RowPin(nrows=2)"

    def test_the_parameter_jacobian_on_the_rows(self, bkd: Backend[Array]) -> None:
        decl: _Declaration[Array] = _Declaration(bkd)
        op = RowPin(decl).data_param_jacobian(decl.states, decl.params)
        assert op is not None
        expected = bkd.asarray([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]])
        bkd.assert_allclose(_dense(op, 2, bkd), expected)
        y = bkd.asarray([[1.0], [-2.0], [0.5]])
        bkd.assert_allclose(op.apply_transpose(y), expected.T @ y)

    def test_the_curvature_sees_the_weights_on_the_rows(
        self, bkd: Backend[Array]
    ) -> None:
        decl: _Declaration[Array] = _Declaration(bkd)
        weights = bkd.asarray([[1.0], [10.0], [2.0]])
        op = RowPin(decl).data_param_curvature(weights, decl.states, decl.params)
        assert op is not None
        bkd.assert_allclose(decl.seen[-1], bkd.asarray([[1.0], [0.0], [2.0]]))
        bkd.assert_allclose(_dense(op, 2, bkd), bkd.asarray([[1.0, 0.0], [0.0, 2.0]]))


class TestStatedZerosAndRefusals:
    def test_independent_data_give_zero_operators(self, bkd: Backend[Array]) -> None:
        decl: _Declaration[Array] = _Declaration(
            bkd, PinnedValuesParamDerivatives.independent()
        )
        pin = RowPin(decl)
        jacobian = pin.data_param_jacobian(decl.states, decl.params)
        assert jacobian is not None and jacobian.properties().zero
        assert jacobian.domain() == decl.params.descriptor()
        assert jacobian.range() == decl.states.descriptor()
        bkd.assert_allclose(_dense(jacobian, 2, bkd), bkd.zeros((3, 2)))
        curvature = pin.data_param_curvature(bkd.ones((3, 1)), decl.states, decl.params)
        assert curvature is not None and curvature.properties().zero
        assert curvature.domain() == decl.params.descriptor()
        assert curvature.range() == decl.params.descriptor()

    def test_unavailable_derivatives_are_none_and_refused(
        self, bkd: Backend[Array]
    ) -> None:
        decl: _Declaration[Array] = _Declaration(
            bkd, PinnedValuesParamDerivatives.unavailable()
        )
        pin = RowPin(decl)
        jacobian = pin.data_param_jacobian(decl.states, decl.params)
        assert jacobian is None
        weights = bkd.ones((3, 1))
        assert pin.data_param_curvature(weights, decl.states, decl.params) is None
        with pytest.raises(ValueError, match="_Declaration does not declare the"):
            require_data_derivative(jacobian, "the parameter Jacobian", decl)

    def test_first_order_data_have_no_curvature(self, bkd: Backend[Array]) -> None:
        probe: _Declaration[Array] = _Declaration(bkd)
        decl: _Declaration[Array] = _Declaration(
            bkd, PinnedValuesParamDerivatives.first_order(probe.jacobian)
        )
        pin = RowPin(decl)
        assert pin.data_param_jacobian(decl.states, decl.params) is not None
        weights = bkd.ones((3, 1))
        assert pin.data_param_curvature(weights, decl.states, decl.params) is None

    def test_a_declaration_must_satisfy_the_protocol(self) -> None:
        with pytest.raises(TypeError, match="PinnedValuesProtocol"):
            RowPin(3.0)  # type: ignore[arg-type]
